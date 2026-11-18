import os
from pathlib import Path

import pytest
import torch

FIXTURE = Path(__file__).parent / "fixtures" / "pusht_parity_2batches.pt"
P1_BAR = 1e-5
P2_BAR = 1e-4


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        pytest.skip("needs the LeWAM/lewam-pusht checkpoint from the Hub")
    from huggingface_hub import hf_hub_download

    from lerobot_policy_lewam_jepa.convert import convert

    data = torch.load(FIXTURE, weights_only=False)
    try:
        ckpt = hf_hub_download(data["checkpoint_repo"], "lewam_best.pt", revision=data["checkpoint_revision"])
        cfg = hf_hub_download(data["checkpoint_repo"], "lewam_config.json", revision=data["checkpoint_revision"])
    except Exception as err:
        pytest.skip(f"Hub download failed: {err}")
    out = tmp_path_factory.mktemp("converted")
    policy = convert(ckpt, cfg, out)
    policy.eval()
    return policy, out, data


def test_converted_config_matches_upstream(converted):
    policy, _, _ = converted
    cfg = policy.config
    assert (cfg.history_len, cfg.history_stride, cfg.frameskip, cfg.chunk_size) == (2, 5, 5, 10)
    assert (cfg.n_flow_steps, cfg.goal_type, cfg.image_size, cfg.crop_size) == (8, "sampled", 224, 202)


def test_parity_with_upstream_on_two_stored_batches(converted):
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    policy, out, data = converted
    post = PolicyProcessorPipeline.from_pretrained(
        out,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    with torch.no_grad():
        for batch in data["batches"]:
            b = batch["history"].shape[0]
            z_history = policy.encode(batch["history"].flatten(0, 1).float() / 255.0).view(b, 2, -1)
            z_goal = policy.encode(batch["goal"].float() / 255.0)
            states = policy.model.predict_states(z_history, batch["history_pad"], batch["action_target"])
            actions = policy.model.predict_actions(
                z_history, batch["history_pad"], z_goal=z_goal, h_norm=batch["h_norm"], action_noise=batch["noise"]
            )
            assert (states - batch["state_pred"]).abs().max() <= P1_BAR
            assert (actions - batch["actions"]).abs().max() <= P2_BAR
            assert (post(actions) - batch["actions_raw"]).abs().max() <= P2_BAR
