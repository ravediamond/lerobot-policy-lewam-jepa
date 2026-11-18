# Parity with upstream LeWAM (2026-11-18)

Bars set before the run: P1 ≤ 1e-5 on predicted latents, P2 ≤ 1e-4 on action chunks, P3 the policy loop within P2.
Numbers from `docs/results/parity_result.json`.

| bar | measure | limit | result | verdict |
|---|---|---|---|---|
| P1 | max \|Δ\| of the predicted next-state latents (`predict_states`, 32 batches × 8) | ≤ 1e-5 | 9.305775165557861e-06 | pass |
| P2 | max \|Δ\| of the action chunks, after the plugin's postprocessor (raw action units) | ≤ 1e-4 | 1.9729137420654297e-05 | pass |
| P2 | the same chunks before unnormalisation (z-scored) | ≤ 1e-4 | 9.5367431640625e-05 | pass |
| P3 | first `predict_action_chunk` after `reset()` vs upstream's first replan (10 actions) | within P2 | 1.4901161193847656e-07 | pass |
| P3 | `select_action` over 2 episodes × 50 steps (10 replans each, `reset()` between) vs upstream `LeWAMPolicy.get_action` | within P2 | 2.8312206268310547e-07 | pass |

Encoder latents of the history frames differ by at most 9.506940841674805e-06 (same file, `encoder_max_abs`).

## Same-build control

The two sides above run on two torch builds: upstream in its own virtual environment (torch 2.11.0+cpu, PyPI
aarch64 wheel), the plugin in the `lerobot-jetson:0.7.0-jp6.2` container (torch 2.11.0, Jetson build), both on CPU in
fp32. Running the upstream stage inside the container too gives 0.0 on every measure
(`docs/results/parity_result_samebuild.json`): the plugin's
arithmetic is identical to upstream's, and the margins in the table come from the two CPU builds. P1 and the z-scored
P2 sit within 7 % and 5 % of their limits (worst batches 22 and 21; batch medians 5.4836273193359375e-06 and
4.76837158203125e-06).

## Inputs

- Checkpoint `LeWAM/lewam-pusht` at `d8e7039c6e3d8eaec320ba0d0122a5e9c9ce23c2`, converted with
  `scripts/convert_upstream_checkpoint.py` (plugin repo). It is a goal-reaching (`gr`) run: `goal_type: sampled`,
  history stride 5 . Parity runs upstream's policy mode, flow head, no planner.
- Data: upstream's `TrajectoryDataset` (commit 83c4703) over the validation split of `pusht.h5` (seed 42 split,
  224,331 validation starts), 256 starts drawn with seed 0, collated into 32 batches of 8; goals and horizons sampled by
  the dataset under `torch.manual_seed(0)`. 10 of the 32 batches contain a padded history frame. The rollout is the
  first 51 frames of the validation episode starting at row 191341, goal = its last frame (row 191477).
- Streaming: `pusht.h5` is 46.3 GB and the disk had 30 GB free, so the 13.8 GB `.zst` was decompressed as a stream
  and only the h5 chunks the 256 samples and the rollout need were kept (0.52 GB in 578 chunks,
  `scripts/zst_sparse.py`, `scripts/extract_parity_batches.py`). The loader code and its sampling are upstream's; only the file access differs.
- Batch size 8 is this run's choice; the bar text fixes 32 batches and does not fix the size.
- Flow noise: one fixed `torch.randn` tensor per batch (generator seed 1000 + batch index), passed to both sides; the
  rollout uses upstream's own generator (`flow_seed=0`) and the plugin's `flow_seed=0`.

## Commands

```bash
<upstream-venv>/bin/python -I scripts/parity.py upstream \
  --upstream-src <clone of MinghaoFu/lewam at 83c4703> --checkpoint ckpt/lewam_best.pt --config ckpt/lewam_config.json \
  --batches parity_batches.pt --out parity_upstream.pt
docker run --rm --memory=8g --memory-swap=8g -v $PWD:/work/plugin \
  -v <data dir>:/work/s -w /work/s lerobot-jetson:0.7.0-jp6.2 bash -c \
  'pip install -q --no-deps --no-build-isolation -e /work/plugin &&
   python /work/plugin/scripts/convert_upstream_checkpoint.py --checkpoint ckpt/lewam_best.pt \
     --config ckpt/lewam_config.json --out converted_pusht &&
   python /work/plugin/scripts/parity.py plugin --converted converted_pusht --batches parity_batches.pt \
     --reference parity_upstream.pt --out parity_result.json'
```
