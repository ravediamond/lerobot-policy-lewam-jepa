import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.utils.constants import ACTION

from lerobot_policy_lewam_jepa import LeWAMJEPAConfig, LeWAMJEPAPolicy, make_lewam_jepa_pre_post_processors

IMG = "observation.image"
SIDE = 96


def make_config(**overrides) -> LeWAMJEPAConfig:
    kwargs = dict(
        input_features={IMG: PolicyFeature(type=FeatureType.VISUAL, shape=(3, SIDE, SIDE))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        device="cpu",
        d_model=32,
        n_heads=2,
        depth=1,
        z_dim=16,
        proj_hidden=32,
        sigreg_num_proj=64,
    )
    kwargs.update(overrides)
    return LeWAMJEPAConfig(**kwargs)


def stats() -> dict:
    return {ACTION: {"mean": torch.tensor([0.5, -1.0]), "std": torch.tensor([2.0, 0.25])}}


def training_batch(cfg: LeWAMJEPAConfig, b: int = 3) -> dict:
    t = len(cfg.observation_delta_indices)
    is_pad = torch.zeros(b, t, dtype=torch.bool)
    is_pad[0, 0] = True
    is_pad[1, -1] = True
    action_is_pad = torch.zeros(b, cfg.chunk_size, dtype=torch.bool)
    action_is_pad[1, 6:] = True
    return {
        IMG: torch.rand(b, t, 3, SIDE, SIDE),
        f"{IMG}_is_pad": is_pad,
        ACTION: torch.randn(b, cfg.chunk_size, 2),
        f"{ACTION}_is_pad": action_is_pad,
    }


def test_registered_through_plugin_discovery():
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    assert "lewam_jepa" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("lewam_jepa") is LeWAMJEPAConfig
    assert LeWAMJEPAPolicy.name == "lewam_jepa"


def test_delta_indices_follow_history_and_state_target():
    cfg = make_config()
    assert cfg.observation_delta_indices == [-1, 0, 5]
    assert cfg.action_delta_indices == list(range(10))
    assert cfg.reward_delta_indices is None
    assert make_config(history_stride=5).observation_delta_indices == [-5, 0, 5]


def test_select_action_and_chunk_shapes():
    torch.manual_seed(0)
    policy = LeWAMJEPAPolicy(make_config())
    obs = {IMG: torch.rand(3, 3, SIDE, SIDE)}
    policy.reset()
    chunk = policy.predict_action_chunk(obs)
    assert chunk.shape == (3, 10, 2)
    assert torch.isfinite(chunk).all()
    policy.reset()
    for _ in range(12):
        action = policy.select_action({IMG: torch.rand(3, 3, SIDE, SIDE)})
        assert action.shape == (3, 2)
    assert policy._step == 12
    assert sorted(policy._frames) == [10, 11]


def test_first_call_after_reset_uses_padded_history():
    torch.manual_seed(0)
    policy = LeWAMJEPAPolicy(make_config())
    obs = {IMG: torch.rand(2, 3, SIDE, SIDE)}
    noise = torch.randn(2, 10, 2)
    policy.reset()
    first = policy.predict_action_chunk(obs, noise=noise)
    z = policy.encode(obs[IMG])
    history = torch.stack([torch.zeros_like(z), z], dim=1)
    pad = torch.tensor([[True, False], [True, False]])
    expected = policy.model.predict_actions(history, pad, action_noise=noise)
    torch.testing.assert_close(first, expected)


def test_history_hand_off_reuses_cached_latents():
    torch.manual_seed(0)
    policy = LeWAMJEPAPolicy(make_config())
    frames = [torch.rand(1, 3, SIDE, SIDE) for _ in range(6)]
    policy.reset()
    for f in frames:
        policy.select_action({IMG: f})
    policy._record({IMG: torch.rand(1, 3, SIDE, SIDE)})
    history, pad = policy._history()
    assert not pad.any()
    torch.testing.assert_close(history[:, 0], policy.encode(frames[-1]))


def test_save_load_round_trip(tmp_path):
    torch.manual_seed(0)
    policy = LeWAMJEPAPolicy(make_config())
    policy.save_pretrained(tmp_path)
    loaded = LeWAMJEPAPolicy.from_pretrained(tmp_path)
    for (k1, v1), (k2, v2) in zip(policy.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)
    obs = {IMG: torch.rand(2, 3, SIDE, SIDE)}
    noise = torch.randn(2, 10, 2)
    policy.reset()
    loaded.reset()
    torch.testing.assert_close(
        policy.predict_action_chunk(obs, noise=noise), loaded.predict_action_chunk(obs, noise=noise)
    )
    assert loaded.config.observation_delta_indices == policy.config.observation_delta_indices


def test_forward_returns_loss_and_backpropagates():
    torch.manual_seed(0)
    cfg = make_config()
    policy = LeWAMJEPAPolicy(cfg)
    policy.train()
    loss, out = policy.forward(training_batch(cfg))
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert {"loss_action", "loss_state", "loss_sigreg", "latent_std"} <= set(out)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    per_sample, _ = policy.forward(training_batch(cfg), reduction="none")
    assert per_sample.shape == (3,)


def test_padded_actions_do_not_change_the_loss():
    cfg = make_config()
    policy = LeWAMJEPAPolicy(cfg)
    policy.train()
    batch = training_batch(cfg)
    other = {k: v.clone() for k, v in batch.items()}
    other[ACTION][batch[f"{ACTION}_is_pad"]] = 1e3
    for reduction in ("mean", "none"):
        torch.manual_seed(1)
        a, _ = policy.forward(batch, reduction=reduction)
        torch.manual_seed(1)
        b, _ = policy.forward(other, reduction=reduction)
        torch.testing.assert_close(a, b)
    torch.manual_seed(1)
    a, _ = policy.forward(batch)
    changed = {k: v.clone() for k, v in batch.items()}
    changed[ACTION][0, 0] += 1.0
    torch.manual_seed(1)
    b, _ = policy.forward(changed)
    assert not torch.allclose(a, b)


def test_processor_round_trip(tmp_path):
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    cfg = make_config()
    pre, post = make_lewam_jepa_pre_post_processors(cfg, stats())
    action = torch.tensor([[1.5, -0.5]])
    image = torch.rand(3, SIDE, SIDE)
    out = pre({IMG: image, ACTION: action})
    torch.testing.assert_close(out[IMG][0], image)
    torch.testing.assert_close(out[ACTION], torch.tensor([[0.5, 2.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(post(out[ACTION]), action, atol=1e-6, rtol=0)
    post.save_pretrained(tmp_path)
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    torch.testing.assert_close(loaded(out[ACTION]), action, atol=1e-6, rtol=0)


def test_goal_conditioned_checkpoint_needs_goal_key():
    with pytest.raises(ValueError):
        LeWAMJEPAPolicy(make_config(goal_type="sampled", history_stride=5))
    cfg = make_config(goal_type="sampled", history_stride=5, goal_image_key="observation.images.goal")
    policy = LeWAMJEPAPolicy(cfg)
    with pytest.raises(NotImplementedError):
        policy.forward(training_batch(cfg))
    obs = {IMG: torch.rand(1, 3, SIDE, SIDE), "observation.images.goal": torch.rand(1, 3, SIDE, SIDE)}
    assert policy.select_action(obs).shape == (1, 2)
