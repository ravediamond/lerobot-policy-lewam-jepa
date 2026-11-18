import json
from pathlib import Path

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION

from .configuration_lewam_jepa import LeWAMJEPAConfig
from .modeling_lewam_jepa import LeWAMJEPAPolicy
from .processor_lewam_jepa import make_lewam_jepa_pre_post_processors

UPSTREAM_KEYS = (
    "fs",
    "action_raw_dim",
    "img_size",
    "encoder_backbone",
    "z_dim",
    "proj_hidden",
    "d_model",
    "n_heads",
    "depth",
    "dropout",
    "n_flow_steps",
    "num_actions_pred",
    "num_states_pred",
    "history_len",
    "history_stride",
    "goal_type",
    "action_head",
    "state_head",
    "action_mean",
    "action_std",
)


def config_from_upstream(up: dict, image_key: str, goal_image_key: str | None, device: str) -> LeWAMJEPAConfig:
    missing = [k for k in UPSTREAM_KEYS if k not in up]
    if missing:
        raise ValueError(f"lewam_config.json lacks {missing}")
    if int(up.get("num_views", 1)) != 1:
        raise ValueError("the converter handles single-view checkpoints only")
    if up["encoder_backbone"] not in ("resnet18dp", "resnet18"):
        raise ValueError(f"encoder {up['encoder_backbone']!r} is not ported")
    if up.get("input_norm", "imagenet") != "imagenet" and "input_norm_mean" not in up:
        raise ValueError("unknown input normalisation")
    goal = up["goal_type"]
    side = int(up["img_size"])
    return LeWAMJEPAConfig(
        input_features={image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, side, side))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(int(up["action_raw_dim"]),))},
        device=device,
        history_len=int(up["history_len"]),
        history_stride=int(up["history_stride"]),
        frameskip=int(up["fs"]),
        chunk_size=int(up["num_actions_pred"]),
        n_action_steps=int(up["fs"]),
        num_states_pred=int(up["num_states_pred"]),
        drop_n_last_frames=int(up["fs"]),
        goal_type=goal,
        h_max=int(up.get("H_max", 10)),
        goal_image_key=None if goal == "none" else goal_image_key,
        encoder_backbone=up["encoder_backbone"],
        image_size=side,
        crop_size=202 if up["encoder_backbone"] == "resnet18dp" else None,
        input_norm_mean=tuple(float(x) for x in up.get("input_norm_mean", (0.485, 0.456, 0.406))),
        input_norm_std=tuple(float(x) for x in up.get("input_norm_std", (0.229, 0.224, 0.225))),
        z_dim=int(up["z_dim"]),
        proj_hidden=int(up["proj_hidden"]),
        d_model=int(up["d_model"]),
        n_heads=int(up["n_heads"]),
        depth=int(up["depth"]),
        dropout=float(up["dropout"]),
        action_head=up["action_head"],
        state_head=up["state_head"],
        n_flow_steps=int(up["n_flow_steps"]),
        flow_alpha_act=float(up.get("flow_alpha_act", 1.0)),
        flow_alpha_state=float(up.get("flow_alpha_state", up.get("flow_alpha_act", 1.0))),
        sigreg_grouped=bool(up.get("sigreg_grouped", False)),
        sigreg_proj_dim=int(up.get("sigreg_proj_dim", 0)),
    )


def action_stats(up: dict) -> dict[str, dict[str, torch.Tensor]]:
    mean = torch.tensor(up["action_mean"], dtype=torch.float32)
    std = torch.tensor(up["action_std"], dtype=torch.float32).clamp_min(1e-6)
    return {ACTION: {"mean": mean, "std": std}}


def convert(
    checkpoint: str | Path,
    upstream_config: str | Path,
    out_dir: str | Path,
    image_key: str = "observation.image",
    goal_image_key: str = "observation.images.goal",
    device: str = "cpu",
    provenance: dict | None = None,
) -> LeWAMJEPAPolicy:
    up = json.loads(Path(upstream_config).read_text())
    cfg = config_from_upstream(up, image_key, goal_image_key, device)
    policy = LeWAMJEPAPolicy(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    missing, unexpected = policy.model.load_state_dict(state, strict=True)
    pre, post = make_lewam_jepa_pre_post_processors(cfg, action_stats(up))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    pre.save_pretrained(out_dir)
    post.save_pretrained(out_dir)
    record = {"upstream_config": up, "source_checkpoint": str(checkpoint), **(provenance or {})}
    (out_dir / "conversion.json").write_text(json.dumps(record, indent=1))
    return policy
