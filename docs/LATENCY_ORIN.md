# Latency on the Jetson Orin NX 16 GB (2026-11-18)

Numbers from `docs/results/latency_orin.json`; power mode from `nvpmodel -q` (25W, mode 3).

Checkpoint: `LeWAM/lewam-pusht` converted (224×224 input, 202 crop, goal image encoded at every replan, 8 Euler
steps, chunk 10, 5 actions executed per replan). Batch 1, fp32, CUDA, torch 2.11.0 in `lerobot-jetson:0.7.0-jp6.2`.
20 warm-up calls, then 200 timed `select_action` calls with `torch.cuda.synchronize()` after each; frames from the
parity rollout episode. 40 of the 200 calls replan; the other 160 pop the action queue.

| mode | `select_action` median (ms) | `select_action` p95 (ms) | replan median (ms) | replan p95 (ms) | peak GPU memory (MB) |
|---|---|---|---|---|---|
| eager | 0.9864154853858054 | 179.24433037405834 | 178.98417596006766 | 179.83783408999443 | 337.302734375 |
| `torch.compile` (encoder and `forward_mot`) | 1.0081594809889793 | 105.84106107708067 | 105.51958699943498 | 107.34982646536082 | 346.11865234375 |

The median of `select_action` is the queue pop; the replan columns are the cost of one new chunk (encode the new frame
and the goal, 8 MoT passes). At 5 actions per replan, a control loop at 10 Hz has 500 ms per replan.

Command:

```bash
docker run --rm --runtime nvidia --memory=11g --memory-swap=11g -v $PWD:/work/plugin \
  -v <data dir>:/work/s -w /work/s lerobot-jetson:0.7.0-jp6.2 bash -c \
  'pip install -q --no-deps --no-build-isolation -e /work/plugin &&
   python /work/plugin/scripts/latency.py --checkpoint converted_pusht --frames parity_batches.pt --out latency_orin.json'
```
