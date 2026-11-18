# lerobot_policy_lewam_jepa

LeWAM ([arXiv:2610.12818](https://arxiv.org/abs/2610.12818), "End-to-End World and Action Modeling with JEPAs") as a
[LeRobot](https://github.com/huggingface/lerobot) 0.7.0 policy plugin, policy type `lewam_jepa`. One image encoder
(ResNet18 with GroupNorm and a 32-keypoint spatial softmax, trained from scratch with the SIGReg regulariser) feeds a
Mixture-of-Transformers predictor with two streams: a latent dynamics stream that predicts the next frame's latent, and
a flow-matching action stream that predicts a 10-action chunk from a 2-frame history. The Push-T model has 15,962,402
parameters, 11,292,576 of them in the encoder.

The model code is vendored from [MinghaoFu/lewam](https://github.com/MinghaoFu/lewam) (MIT, commit `83c4703`) with
import changes only (see [NOTICE](NOTICE)); nothing from stable-worldmodel, stable-pretraining or le-wm is imported at
run time. Version 0.1 ports upstream's policy mode. Planning (`lewam_plan`) is not ported.

| check | result | file |
|---|---|---|
| parity with upstream, predicted latents (32 batches × 8, Push-T validation split) | max \|Δ\| 9.3e-06 (bar 1e-5) | [PARITY](docs/PARITY.md) |
| parity, action chunks in action units | max \|Δ\| 2.0e-05 (bar 1e-4) | same |
| parity, `select_action` over 2 × 50 steps with `reset()` between | max \|Δ\| 2.8e-07 | same |
| same, with upstream and the plugin on one torch build | 0.0 on every measure | same |
| Jetson Orin NX 16 GB, 25 W, one new chunk (224 px, fp32) | 179 ms eager, 106 ms `torch.compile` | [LATENCY](docs/LATENCY_ORIN.md) |
| `lerobot-train` 200 steps + `lerobot-eval` 2 episodes on `lerobot/pusht`, Orin | both run to the end | smoke run, no score |

No success rate is claimed yet. A Push-T benchmark with `lerobot-eval` and an SO-101 test come in later releases.

## Install

Requires Python ≥ 3.12 and LeRobot 0.7.0. Not on PyPI yet:

```bash
pip install "lerobot_policy_lewam_jepa @ git+https://github.com/ravediamond/lerobot-policy-lewam-jepa@v0.1.0"
```

LeRobot finds the policy by the distribution name prefix `lerobot_policy_`; there is no entry point. Check discovery:

```bash
python -c "
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.configs import PreTrainedConfig
register_third_party_plugins()
assert 'lewam_jepa' in PreTrainedConfig.get_known_choices()
print('ok')"
```

On a Jetson, or any torch build pip cannot fetch again (a JetPack wheel, ROCm, a source build), install with
`--no-deps` so pip does not replace your torch with a PyPI CUDA wheel:

```bash
pip install --no-deps "lerobot_policy_lewam_jepa @ git+https://github.com/ravediamond/lerobot-policy-lewam-jepa@v0.1.0"
python -c "import torch; print(torch.__version__)"
```

This needs LeRobot 0.7.0 already installed in that environment (the Orin numbers above come from a container with
LeRobot 0.7.0 and torch 2.11.0 built for JetPack 6.2).

## Train

```bash
lerobot-train --policy.type=lewam_jepa --dataset.repo_id=lerobot/pusht --env.type=pusht \
  --batch_size=64 --steps=100000 --policy.push_to_hub=false --output_dir=outputs/lewam_jepa_pusht
lerobot-eval --policy.path=outputs/lewam_jepa_pusht/checkpoints/last/pretrained_model --env.type=pusht \
  --eval.n_episodes=50
```

The defaults are upstream's `tc` (behaviour cloning) configuration: history of 2 frames 1 step apart, chunk of 10
actions, 5 executed per replan, one predicted next-frame latent 5 steps ahead, 8 Euler steps, AdamW lr 1e-4 with weight
decay 1e-4, gradient clipping at 1.0, loss = action flow loss + latent loss + 0.04 × SIGReg. The image crop defaults
to 84 px for LeRobot's 96 px Push-T frames; upstream trains on 224 px frames with a 202 px crop (`--policy.image_size=224
--policy.crop_size=202`). Only `observation.image*` features are read; `observation.state` is not an input, as in
upstream. Several cameras of the same size are supported; each is one view.

The dataset loader provides frames at offsets `[-1, 0, 5]` (two history frames and the latent target) and 10 actions;
`drop_n_last_frames=5` matches upstream's rule that a sample starts at least 5 frames before the episode's end.

Differences from upstream's training script, all in the training loop and not in the model:

| upstream | here |
|---|---|
| epochs: 120 with 10 warm-up epochs, cosine to 0 | steps: cosine with 10,000 warm-up steps over 120,000 (`scheduler_warmup_steps`, `scheduler_decay_steps`) |
| actions z-scored with the dataset's mean and sample std, stored in fp16 | LeRobot's `MEAN_STD` normaliser (population std, fp32) |
| a history frame before the episode start repeats the first frame inside the window | LeRobot repeats the episode's first frame; the frame is masked out of attention either way, but it still enters SIGReg and the projector's BatchNorm statistics |
| padded actions are zeros | padded actions are set to zero after normalisation, so the clean-action tokens match |

## Use an upstream checkpoint

```bash
python scripts/convert_upstream_checkpoint.py --repo-id LeWAM/lewam-pusht --out converted/lewam_pusht
```

The converter reads `lewam_best.pt` and `lewam_config.json` and writes `model.safetensors`, `config.json`, the
preprocessor and postprocessor (action mean and std from the checkpoint) and `conversion.json` with the source
revision. Single-view ResNet checkpoints convert; the ViT and DINOv3 encoders are not ported.

`LeWAM/lewam-pusht` is upstream's goal-reaching run (`goal_type: sampled`, history stride 5). The converted policy
reads a goal image from `observation.images.goal` (set `--goal-image-key` to change it) and counts the goal horizon
down from `h_max` = 10 chunks as upstream's `LeWAMPolicy` does. Its inputs are upstream's own Push-T environment
(stable-worldmodel `swm/PushT-v1`), whose frames, goals and action space differ from LeRobot's `gym_pusht`; it is
there for parity and for comparing against upstream's evaluation, not for `lerobot-eval --env.type=pusht`. Training
with a goal is not supported in 0.1 (`forward` raises for `goal_type != "none"`).

## Scope of 0.1 and what blocks the rest

- **Planning.** Upstream's planners (gradient, CEM, steering) optimise action chunks against a goal latent through the
  dynamics stream. They need a goal image and several model calls per step, and `select_action` has no place for
  them. Not ported.
- **Real-time chunking (RTC).** RTC adds a guidance term at every denoising step so the new chunk agrees with the
  actions already committed. The vendored `predict_actions` runs a fixed 8-step Euler loop with no hook for that term,
  and 0.1 keeps the vendored code identical to upstream for parity. `supports_rtc` stays false.
- **Async inference.** The policy keeps its own frame history, one frame per `select_action` call, and picks the
  history frames by step offset (1 step in `tc`, 5 in the Push-T checkpoint). The policy server calls
  `predict_action_chunk` once per observation it receives, at chunk boundaries rather than every control step, so the
  step offsets would not match real time. The server would have to send frames at the history stride (temporal
  observation sampling).
- **Precision.** Measured in fp32 only; `dtype` is left at LeRobot's default.

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

Registration through plugin discovery, delta indices, `select_action` and `predict_action_chunk` shapes, the first
call after `reset()`, the history hand-off between calls, save and load, the loss and its backward pass, that padded
actions do not change the loss (`reduction="mean"` and `"none"`), the processor round trip, and parity on 2 stored
Push-T batches (`tests/fixtures/pusht_parity_2batches.pt`, 1.7 MB; downloads `LeWAM/lewam-pusht` and skips when
`HF_HUB_OFFLINE=1`). CI runs on Python 3.12 and 3.13.

`scripts/parity.py` and `scripts/latency.py` reproduce the full parity run and the latency table; `scripts/zst_sparse.py` and
`scripts/extract_parity_batches.py` rebuild the parity batches from upstream's 13.8 GB Push-T archive without storing
it whole. The commands are in [docs/PARITY.md](docs/PARITY.md) and [docs/LATENCY_ORIN.md](docs/LATENCY_ORIN.md).

## Citation

```bibtex
@article{fu2026lewam,
  title={LeWAM: End-to-End World and Action Modeling with JEPAs},
  author={Fu, Minghao and Siebert, Tavis and Halicki, Eryk and Balestriero, Randall},
  journal={arXiv preprint arXiv:2610.12818},
  year={2026}
}
```

The name `lewam_jepa` keeps this policy apart from two other 2026 papers called LeWAM (arXiv:2610.12407 and
arXiv:2609.27455) and from LaWAM, which is already in LeRobot.

## License

Apache-2.0 for the code in this repository. The vendored model code under `src/lerobot_policy_lewam_jepa/_vendor/` is
MIT ([third_party/LICENSE-lewam-MIT](third_party/LICENSE-lewam-MIT)); see [NOTICE](NOTICE).
