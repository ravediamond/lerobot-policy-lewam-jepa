import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from lerobot_policy_lewam_jepa import LeWAMJEPAPolicy


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def observations(policy, frames: torch.Tensor, goal: torch.Tensor, n: int, device: str) -> list[dict]:
    key = policy.config.camera_keys[0]
    obs = []
    for t in range(n):
        o = {key: (frames[t % len(frames)].float() / 255.0)[None].to(device)}
        if policy.config.goal_image_key:
            o[policy.config.goal_image_key] = (goal.float() / 255.0)[None].to(device)
        obs.append(o)
    return obs


def measure(policy, obs: list[dict], warmup: int, calls: int) -> dict:
    policy.reset()
    torch.cuda.synchronize()
    for o in obs[:warmup]:
        policy.select_action(o)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, replans = [], []
    for o in obs[warmup : warmup + calls]:
        replan = len(policy._queue) == 0
        t0 = time.perf_counter()
        policy.select_action(o)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        times.append(dt)
        if replan:
            replans.append(dt)
    return {
        "select_action_ms_median": statistics.median(times),
        "select_action_ms_p95": percentile(times, 0.95),
        "replan_calls": len(replans),
        "replan_ms_median": statistics.median(replans),
        "replan_ms_p95": percentile(replans, 0.95),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--calls", type=int, default=200)
    a = p.parse_args()
    device = "cuda"
    data = torch.load(a.frames, weights_only=False)["rollout"]
    policy = LeWAMJEPAPolicy.from_pretrained(a.checkpoint).to(device).eval()
    obs = observations(policy, data["frames"], data["goal"], a.warmup + a.calls, device)
    result = {"eager": measure(policy, obs, a.warmup, a.calls)}
    policy.model.encoder = torch.compile(policy.model.encoder)
    policy.model.forward_mot = torch.compile(policy.model.forward_mot)
    result["compiled"] = measure(policy, obs, a.warmup, a.calls)
    result.update(
        {
            "warmup_calls": a.warmup,
            "timed_calls": a.calls,
            "batch_size": 1,
            "dtype": "float32",
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "n_action_steps": policy.config.n_action_steps,
            "n_flow_steps": policy.config.n_flow_steps,
        }
    )
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
