import argparse
import json
import sys
import types
from pathlib import Path

import torch

ROLL_STEPS = 50


def stub_stable_worldmodel():
    class BasePolicy:
        def __init__(self, **kwargs):
            self.env = None

        def _prepare_info(self, info_dict):
            return dict(info_dict)

    swm = types.ModuleType("stable_worldmodel")
    pol = types.ModuleType("stable_worldmodel.policy")
    pol.BasePolicy = BasePolicy
    swm.policy = pol
    sys.modules.setdefault("stable_worldmodel", swm)
    sys.modules.setdefault("stable_worldmodel.policy", pol)


def batch_noise(b: int, batch_size: int, chunk: int, dim: int) -> torch.Tensor:
    return torch.randn(batch_size, chunk, dim, generator=torch.Generator().manual_seed(1000 + b))


def run_upstream(a):
    sys.path.insert(0, a.upstream_src)
    stub_stable_worldmodel()
    from lewam.eval.policy import LeWAMPolicy
    from lewam.models.lewam import build_model
    from lewam.train.utils import IMAGENET_MEAN, IMAGENET_STD

    torch.manual_seed(0)
    cfg = json.loads(Path(a.config).read_text())
    model = build_model(cfg)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    data = torch.load(a.batches, weights_only=False)

    def norm(x):
        return (x.float() / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

    out = {"batches": []}
    with torch.no_grad():
        for b, batch in enumerate(data["batches"]):
            bsz = batch["history"].shape[0]
            z_history = model.encode(norm(batch["history"].flatten(0, 1))).view(bsz, cfg["history_len"], -1)
            state_target = model.encode(norm(batch["state_frames"].flatten(0, 1))).view(bsz, cfg["num_states_pred"], -1)
            z_goal = model.encode(norm(batch["goal"]))
            noise = batch_noise(b, bsz, cfg["num_actions_pred"], cfg["action_raw_dim"])
            states = model.predict_states(z_history, batch["history_pad"], batch["action_target"])
            actions = model.predict_actions(
                z_history, batch["history_pad"], z_goal=z_goal, h_norm=batch["h_norm"], action_noise=noise
            )
            mean, std = torch.tensor(cfg["action_mean"]), torch.tensor(cfg["action_std"]).clamp_min(1e-6)
            out["batches"].append(
                {
                    "z_history": z_history,
                    "state_target": state_target,
                    "z_goal": z_goal,
                    "state_pred": states,
                    "actions": actions,
                    "actions_raw": actions * std + mean,
                }
            )
        roll = data["rollout"]
        env = types.SimpleNamespace(num_envs=1, action_space=types.SimpleNamespace(shape=(1, cfg["action_raw_dim"])))
        policy = LeWAMPolicy(model, cfg, frameskip=cfg["fs"], action_dim=cfg["action_raw_dim"], flow_seed=0)
        policy.set_env(env)
        goal = norm(roll["goal"][None])
        episodes = []
        for episode in range(2):
            if episode:
                policy._flush_env(0)
            acts = []
            for t in range(ROLL_STEPS):
                info = {"pixels": norm(roll["frames"][t][None]), "goal": goal}
                acts.append(torch.from_numpy(policy.get_action(info)).reshape(-1))
            episodes.append(torch.stack(acts))
        out["rollout_actions"] = torch.stack(episodes)
        h = torch.zeros(1, cfg["history_len"], model.z_dim)
        pad = torch.tensor([[True] * (cfg["history_len"] - 1) + [False]])
        h[0, -1] = model.encode(norm(roll["frames"][0][None]))[0]
        g = torch.Generator().manual_seed(0)
        chunk = model.predict_actions(h, pad, z_goal=model.encode(goal), h_norm=torch.ones(1), generator=g)
        out["first_chunk_raw"] = (chunk * std + mean)[0]
    torch.save(out, a.out)
    print(f"wrote {a.out}")


def max_abs(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.double() - y.double()).abs().max())


def run_plugin(a):
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )

    from lerobot_policy_lewam_jepa import LeWAMJEPAPolicy

    torch.manual_seed(0)
    policy = LeWAMJEPAPolicy.from_pretrained(a.converted)
    policy.eval()
    policy.config.flow_seed = 0
    pre = PolicyProcessorPipeline.from_pretrained(
        a.converted,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    post = PolicyProcessorPipeline.from_pretrained(
        a.converted,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    cfg = policy.config
    image_key = cfg.camera_keys[0]
    data = torch.load(a.batches, weights_only=False)
    ref = torch.load(a.reference, weights_only=False)
    rows = []
    p1 = p2 = p2n = enc = 0.0
    with torch.no_grad():
        for b, (batch, r) in enumerate(zip(data["batches"], ref["batches"], strict=True)):
            bsz = batch["history"].shape[0]
            z_history = policy.encode(batch["history"].flatten(0, 1).float() / 255.0).view(bsz, cfg.history_len, -1)
            z_goal = policy.encode(batch["goal"].float() / 255.0)
            states = policy.model.predict_states(z_history, batch["history_pad"], batch["action_target"])
            noise = batch_noise(b, bsz, cfg.chunk_size, cfg.action_feature.shape[0])
            actions = policy.model.predict_actions(
                z_history, batch["history_pad"], z_goal=z_goal, h_norm=batch["h_norm"], action_noise=noise
            )
            raw = post(actions)
            row = {
                "batch": b,
                "encoder": max_abs(z_history, r["z_history"]),
                "P1": max_abs(states, r["state_pred"]),
                "P2_raw": max_abs(raw, r["actions_raw"]),
                "P2_norm": max_abs(actions, r["actions"]),
            }
            rows.append(row)
            enc, p1 = max(enc, row["encoder"]), max(p1, row["P1"])
            p2, p2n = max(p2, row["P2_raw"]), max(p2n, row["P2_norm"])
        roll = data["rollout"]
        goal = roll["goal"][None].float() / 255.0
        episodes = []
        policy.reset()
        policy._flow_rng = None
        for episode in range(2):
            if episode:
                policy.reset()
            acts = []
            for t in range(ROLL_STEPS):
                obs = pre({image_key: roll["frames"][t].float() / 255.0, cfg.goal_image_key: goal[0]})
                acts.append(post(policy.select_action(obs)).reshape(-1))
            episodes.append(torch.stack(acts))
        p3_roll = max_abs(torch.stack(episodes), ref["rollout_actions"])
        policy.reset()
        policy._flow_rng = None
        obs = pre({image_key: roll["frames"][0].float() / 255.0, cfg.goal_image_key: goal[0]})
        first = post(policy.predict_action_chunk(obs))[0]
        p3_first = max_abs(first, ref["first_chunk_raw"])
    result = {
        "P1_max_abs_latent_pred": p1,
        "P2_max_abs_action_raw": p2,
        "P2_max_abs_action_normalised": p2n,
        "encoder_max_abs": enc,
        "P3_rollout_max_abs": p3_roll,
        "P3_first_call_max_abs": p3_first,
        "P1_pass": p1 <= 1e-5,
        "P2_pass": p2 <= 1e-4 and p2n <= 1e-4,
        "P3_pass": p3_roll <= 1e-4 and p3_first <= 1e-4,
        "n_batches": len(rows),
        "batch_size": int(data["batch_size"]),
        "rollout_steps_per_episode": ROLL_STEPS,
        "rollout_episodes": 2,
        "torch": torch.__version__,
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)
    u = sub.add_parser("upstream")
    u.add_argument("--upstream-src", required=True)
    u.add_argument("--checkpoint", required=True)
    u.add_argument("--config", required=True)
    u.add_argument("--batches", required=True)
    u.add_argument("--out", required=True)
    q = sub.add_parser("plugin")
    q.add_argument("--converted", required=True)
    q.add_argument("--batches", required=True)
    q.add_argument("--reference", required=True)
    q.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    (run_upstream if a.stage == "upstream" else run_plugin)(a)


if __name__ == "__main__":
    main()
