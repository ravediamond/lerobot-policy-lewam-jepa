from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@PreTrainedConfig.register_subclass("lewam_jepa")
@dataclass
class LeWAMJEPAConfig(PreTrainedConfig):
    history_len: int = 2
    history_stride: int = 1
    frameskip: int = 5
    chunk_size: int = 10
    n_action_steps: int = 5
    num_states_pred: int = 1

    goal_type: str = "none"
    h_max: int = 10
    goal_h0: float | None = None
    goal_image_key: str | None = None

    encoder_backbone: str = "resnet18dp"
    image_size: int | None = None
    crop_size: int | None = 84
    input_norm_mean: tuple[float, float, float] = IMAGENET_MEAN
    input_norm_std: tuple[float, float, float] = IMAGENET_STD

    z_dim: int = 192
    proj_hidden: int = 384
    d_model: int = 192
    n_heads: int = 4
    depth: int = 4
    dropout: float = 0.1

    action_head: str = "flow"
    state_head: str = "mse"
    n_flow_steps: int = 8
    flow_alpha_act: float = 1.0
    flow_alpha_state: float = 1.0
    flow_seed: int | None = None

    sigreg_grouped: bool = True
    sigreg_proj_dim: int = 0
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    w_act: float = 1.0
    w_dyn: float = 1.0
    w_reg: float = 0.04

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    drop_n_last_frames: int = 5

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 10_000
    scheduler_decay_steps: int = 120_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.goal_type not in ("none", "terminal", "sampled"):
            raise ValueError(f"goal_type must be none, terminal or sampled, got {self.goal_type!r}")
        if self.action_head not in ("flow", "mse") or self.state_head not in ("flow", "mse"):
            raise ValueError("action_head and state_head must be 'flow' or 'mse'")
        if self.encoder_backbone not in ("resnet18dp", "resnet18"):
            raise ValueError(f"encoder_backbone must be resnet18dp or resnet18, got {self.encoder_backbone!r}")
        if self.chunk_size < self.num_states_pred * self.frameskip:
            raise ValueError("chunk_size must cover num_states_pred * frameskip actions")
        if not 0 < self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be in 1..chunk_size")
        if not 1 <= self.history_stride <= self.frameskip:
            raise ValueError("history_stride must be in 1..frameskip")
        if self.history_len < 1:
            raise ValueError("history_len must be at least 1")

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("lewam_jepa needs at least one image feature")
        if self.action_feature is None:
            raise ValueError("lewam_jepa needs 'action' in output_features")
        shapes = {tuple(ft.shape) for ft in self.image_features.values()}
        if len(shapes) != 1:
            raise ValueError(f"every camera must have the same shape, got {sorted(shapes)}")
        ((c, h, w),) = shapes
        side = self.image_size or min(h, w)
        if self.crop_size is not None and self.crop_size > side:
            raise ValueError(f"crop_size {self.crop_size} does not fit images of side {side}")
        if self.goal_type != "none" and self.goal_image_key is None:
            raise ValueError("a goal-conditioned checkpoint needs goal_image_key")

    @property
    def camera_keys(self) -> list[str]:
        return [k for k in self.image_features if k != self.goal_image_key]

    @property
    def history_offsets(self) -> list[int]:
        return [-k * self.history_stride for k in range(self.history_len - 1, -1, -1)]

    @property
    def state_offsets(self) -> list[int]:
        return [q * self.frameskip for q in range(1, self.num_states_pred + 1)]

    @property
    def observation_delta_indices(self) -> list[int]:
        return self.history_offsets + self.state_offsets

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def model_kwargs(self) -> dict:
        return dict(
            fs=self.frameskip,
            action_raw_dim=self.action_feature.shape[0],
            img_size=self.image_size or next(iter(self.image_features.values())).shape[-1],
            encoder_backbone=self.encoder_backbone,
            encoder_size="tiny",
            encoder_ckpt=None,
            crop_size=self.crop_size,
            z_dim=self.z_dim,
            proj_hidden=self.proj_hidden,
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            dropout=self.dropout,
            n_flow_steps=self.n_flow_steps,
            num_actions_pred=self.chunk_size,
            num_states_pred=self.num_states_pred,
            history_len=self.history_len,
            num_views=len(self.camera_keys),
            goal_type=self.goal_type,
            goal_view_indices=[0],
            action_head=self.action_head,
            state_head=self.state_head,
            flow_alpha_act=self.flow_alpha_act,
            flow_alpha_state=self.flow_alpha_state,
            sigreg_grouped=self.sigreg_grouped,
            sigreg_proj_dim=self.sigreg_proj_dim if self.w_reg > 0 else 0,
        )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )
