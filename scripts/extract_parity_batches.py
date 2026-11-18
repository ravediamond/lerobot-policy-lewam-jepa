import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zst_sparse

URL = "https://huggingface.co/datasets/LeWAM/lewam-pusht/resolve/main/pusht.h5.zst"


def resolve(root, size, fn, rounds=12):
    for _ in range(rounds):
        sf = zst_sparse.SparseFile(root, size)
        with h5py.File(sf, "r") as h:
            out = fn(h)
        if not sf.misses:
            return out
        print(f"fetching {len(sf.misses)} missing spans", flush=True)
        zst_sparse.fetch(URL, root, sf.misses, size)
    raise RuntimeError("h5 reads still missing after the fetch rounds")


class RowRecorder:
    def __init__(self, n, frame_shape):
        self.rows = set()
        self.shape = (n, *frame_shape)
        self.dtype = torch.uint8

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        idx = torch.as_tensor(idx).reshape(-1).tolist() if not isinstance(idx, int) else [idx]
        self.rows.update(int(i) for i in idx)
        return torch.zeros((len(idx), *self.shape[1:]), dtype=torch.uint8)


class SparseFrames:
    def __init__(self, h, n):
        self.col = h["pixels"]
        self.shape = (n, 3, 224, 224)
        self.dtype = torch.uint8

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, int) or np.ndim(idx) == 0:
            return torch.from_numpy(self.col[int(idx)]).permute(2, 0, 1).contiguous()
        rows = np.asarray(idx).reshape(-1)
        return torch.stack([torch.from_numpy(self.col[int(r)]) for r in rows]).permute(0, 3, 1, 2).contiguous()


def chunk_spans(h, name, rows):
    ds = h[name]
    step = ds.chunks[0]
    spans = []
    for c in sorted({r // step for r in rows}):
        coord = (c * step,) + (0,) * (ds.ndim - 1)
        info = ds.id.get_chunk_info_by_coord(coord)
        spans.append([info.byte_offset, info.byte_offset + info.size])
    return spans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_batches", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--rollout_steps", type=int, default=60)
    p.add_argument("--plan_only", action="store_true")
    a = p.parse_args()
    sys.path.insert(0, a.upstream)
    from lewam.train.datasets import TrajectoryDataset

    cfg = json.load(open(a.config))
    size = json.load(open(os.path.join(a.root, "meta.json")))["size"]

    def meta(h):
        return np.asarray(h["ep_offset"][:]).reshape(-1), np.asarray(h["ep_len"][:]).reshape(-1), h["pixels"].shape[0]

    ep_offset, ep_len, n_rows = resolve(a.root, size, meta)
    starts_ep, first = np.unique(ep_offset, return_index=True)
    lengths = ep_len[first]
    frame_starts, ep_starts, to_term = [], [], []
    fs = cfg["fs"]
    for s, ln in zip(starts_ep, lengths, strict=True):
        end = ln - 1
        t = np.arange(max(0, end - fs + 1))
        frame_starts.append(s + t)
        ep_starts.append(np.full(t.size, s))
        to_term.append(end - t)
    frame_starts, ep_starts, to_term = (
        torch.from_numpy(np.concatenate(x).astype(np.int64)) for x in (frame_starts, ep_starts, to_term)
    )
    n_starts = frame_starts.shape[0]
    possible = torch.randperm(n_starts, generator=torch.Generator().manual_seed(42))
    n_val = int(round((1 - 0.9) * possible.numel()))
    val = possible[:n_val]
    pick = torch.randperm(n_val, generator=torch.Generator().manual_seed(0))[: a.n_batches * a.batch_size]

    def dataset(frames):
        return TrajectoryDataset(
            frames,
            torch.zeros(n_rows, cfg["action_raw_dim"]),
            frame_starts,
            ep_starts,
            to_term,
            val,
            cfg["history_len"],
            cfg["history_stride"],
            cfg["num_actions_pred"],
            cfg["num_states_pred"],
            fs,
            cfg["goal_type"],
            cfg["H_max"],
        )

    rec = RowRecorder(n_rows, (3, 224, 224))
    torch.manual_seed(0)
    ds = dataset(rec)
    for i in pick.tolist():
        ds[i]
    first_val_ep = int(ep_starts[val[0]])
    roll_rows = list(range(first_val_ep, first_val_ep + a.rollout_steps + 1))
    ep_idx = int(np.searchsorted(starts_ep, first_val_ep))
    goal_row = first_val_ep + int(lengths[ep_idx]) - 1
    pixel_rows = sorted(rec.rows | set(roll_rows) | {goal_row})
    action_rows = sorted(
        {r + k for i in pick.tolist() for r in [int(frame_starts[val[i]])] for k in range(cfg["num_actions_pred"])}
        | set(roll_rows)
    )
    action_rows = [r for r in action_rows if r < n_rows]

    def spans(h):
        return chunk_spans(h, "pixels", pixel_rows) + chunk_spans(h, "action", action_rows)

    wanted = resolve(a.root, size, spans)
    total = sum(e - s for s, e in wanted)
    print(
        f"samples {len(pick)} pixel rows {len(pixel_rows)} chunks {len(wanted)} bytes {total / 1e9:.2f} GB "
        f"last offset {max(e for _, e in wanted) / 1e9:.2f} GB",
        flush=True,
    )
    if a.plan_only:
        return
    zst_sparse.fetch(URL, a.root, wanted, size)

    def build(h):
        frames = SparseFrames(h, n_rows)
        raw = h["action"]
        actions = torch.zeros(n_rows, cfg["action_raw_dim"])
        for r in action_rows:
            actions[r] = torch.from_numpy(raw[r])
        mean, std = torch.tensor(cfg["action_mean"]), torch.tensor(cfg["action_std"])
        z = ((actions - mean) / std).half()
        torch.manual_seed(0)
        d = TrajectoryDataset(
            frames,
            z,
            frame_starts,
            ep_starts,
            to_term,
            val,
            cfg["history_len"],
            cfg["history_stride"],
            cfg["num_actions_pred"],
            cfg["num_states_pred"],
            fs,
            cfg["goal_type"],
            cfg["H_max"],
        )
        samples = [d[i] for i in pick.tolist()]
        keys = [
            "history",
            "history_pad",
            "action_target",
            "action_valid",
            "state_frames",
            "state_valid",
            "goal",
            "h_norm",
        ]
        batches = [
            {
                k: torch.stack([s[j] for s in samples[b * a.batch_size : (b + 1) * a.batch_size]])
                for j, k in enumerate(keys)
            }
            for b in range(a.n_batches)
        ]
        roll = {
            "frames": frames[np.asarray(roll_rows)],
            "actions_raw": torch.stack([torch.from_numpy(raw[r]) for r in roll_rows]),
            "goal": frames[goal_row],
            "episode_first_row": first_val_ep,
            "goal_row": goal_row,
        }
        return batches, roll

    batches, roll = resolve(a.root, size, build)
    torch.save(
        {
            "batches": batches,
            "rollout": roll,
            "pick": pick,
            "n_val": n_val,
            "seed_split": 42,
            "batch_size": a.batch_size,
        },
        a.out,
    )
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
