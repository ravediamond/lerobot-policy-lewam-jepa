from collections import deque

import torch
import torch.nn.functional as F
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from torch import Tensor

from ._vendor.lewam import LeWAM
from ._vendor.module import SIGReg
from .configuration_lewam_jepa import LeWAMJEPAConfig


def sigreg_loss(
    sigreg: SIGReg, model: LeWAM, z_history: Tensor, state_target: Tensor, z_goal: Tensor | None, grouped: bool
) -> Tensor:
    project = model.sigreg_proj if model.sigreg_proj_dim > 0 else (lambda x: x)
    if not grouped:
        pooled = torch.cat([z_history[:, -1], state_target.reshape(-1, state_target.shape[-1])])
        return sigreg(project(pooled).unsqueeze(0))
    groups = [z_history[:, k] for k in range(z_history.shape[1])]
    groups += [state_target[:, q] for q in range(state_target.shape[1])]
    if z_goal is not None:
        groups += [z_goal] if z_goal.dim() == 2 else [z_goal[:, g] for g in range(z_goal.shape[1])]
    return torch.stack([sigreg(project(group).unsqueeze(0)) for group in groups]).mean()


def view_major(latents: Tensor, batch: int, frames: int, views: int) -> Tensor:
    return latents.reshape(batch, frames, views, -1).transpose(1, 2).reshape(batch, views * frames, -1)


class LeWAMJEPAPolicy(PreTrainedPolicy):
    config_class = LeWAMJEPAConfig
    name = "lewam_jepa"

    def __init__(self, config: LeWAMJEPAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = LeWAM(config.model_kwargs())
        self.sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_num_proj)
        self.register_buffer("pixel_mean", torch.tensor(config.input_norm_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(config.input_norm_std).view(1, 3, 1, 1), persistent=False)
        self._flow_rng = None
        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._queue = deque(maxlen=self.config.n_action_steps)
        self._frames = {}
        self._latents = {}
        self._step = 0
        self._steps_left = None

    def preprocess_pixels(self, images: Tensor) -> Tensor:
        side = self.config.image_size
        if side is not None and tuple(images.shape[-2:]) != (side, side):
            images = F.interpolate(images, size=(side, side), mode="bilinear", align_corners=False, antialias=True)
        return (images - self.pixel_mean) / self.pixel_std

    def encode(self, images: Tensor) -> Tensor:
        return self.model.encode(self.preprocess_pixels(images))

    def _flow_generator(self, device: torch.device) -> torch.Generator | None:
        if self.config.flow_seed is None:
            return None
        if self._flow_rng is None or self._flow_rng.device != device:
            self._flow_rng = torch.Generator(device=device)
            self._flow_rng.manual_seed(self.config.flow_seed)
        return self._flow_rng

    def _current_frames(self, batch: dict[str, Tensor]) -> Tensor:
        frames = []
        for key in self.config.camera_keys:
            img = batch[key]
            frames.append(img[:, -1] if img.dim() == 5 else img)
        return torch.stack(frames, dim=1)

    def _record(self, batch: dict[str, Tensor]) -> None:
        window = (self.config.history_len - 1) * self.config.history_stride
        self._frames[self._step] = self._current_frames(batch)
        self._frames = {s: f for s, f in self._frames.items() if s >= self._step - window}
        self._latents = {s: z for s, z in self._latents.items() if s >= self._step - window}

    def _history(self) -> tuple[Tensor, Tensor]:
        steps = [self._step + o for o in self.config.history_offsets]
        present = [s for s in steps if s >= 0]
        missing = [s for s in present if s not in self._latents]
        if missing:
            pixels = torch.stack([self._frames[s] for s in missing])
            m, b, v = pixels.shape[:3]
            latents = self.encode(pixels.flatten(0, 2)).view(m, b, v, -1)
            for s, z in zip(missing, latents, strict=True):
                self._latents[s] = z
        first = self._latents[present[-1]]
        b, v, d = first.shape
        h = self.config.history_len
        history = first.new_zeros(b, v, h, d)
        history_pad = torch.ones(b, h, dtype=torch.bool, device=first.device)
        buf = torch.stack([self._latents[s] for s in present], dim=2)
        history[:, :, h - len(present) :] = buf
        history_pad[:, h - len(present) :] = False
        if v == 1:
            history = history[:, 0]
        return history, history_pad

    def _goal(self, batch: dict[str, Tensor], b: int, device: torch.device) -> dict:
        if self.config.goal_type == "none":
            return {}
        goal = batch[self.config.goal_image_key]
        goal = goal[:, -1] if goal.dim() == 5 else goal
        z_goal = self.encode(goal)
        h_norm = None
        if self.config.goal_type == "sampled":
            if self._steps_left is None:
                h0 = self.config.goal_h0 if self.config.goal_h0 is not None else float(self.config.h_max)
                self._steps_left = torch.full((b,), h0, dtype=torch.float64)
            steps = self._steps_left.clamp(min=1.0)
            h_norm = (steps.clamp(max=self.config.h_max) / self.config.h_max).to(device=device, dtype=torch.float32)
        return dict(z_goal=z_goal, h_norm=h_norm)

    def _count_down(self, executed: int) -> None:
        if self._steps_left is not None:
            self._steps_left = (self._steps_left - executed / self.config.frameskip).clamp(min=1.0)

    @torch.no_grad()
    def _predict_from_buffer(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        history, history_pad = self._history()
        goal = self._goal(batch, history.shape[0], history.device)
        return self.model.predict_actions(
            history, history_pad, action_noise=noise, generator=self._flow_generator(history.device), **goal
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        self._record(batch)
        chunk = self._predict_from_buffer(batch, kwargs.get("noise"))
        self._count_down(self.config.n_action_steps)
        return chunk

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        self._record(batch)
        if len(self._queue) == 0:
            chunk = self._predict_from_buffer(batch, kwargs.get("noise"))
            self._count_down(self.config.n_action_steps)
            self._queue.extend(chunk[:, : self.config.n_action_steps].transpose(0, 1))
        self._step += 1
        return self._queue.popleft()

    def _encode_training_batch(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        keys = cfg.camera_keys
        images = torch.stack([batch[k] for k in keys], dim=2)
        b, t, v = images.shape[:3]
        h = cfg.history_len
        if t != h + cfg.num_states_pred:
            raise ValueError(f"expected {h + cfg.num_states_pred} frames per sample, got {t}")
        history_pixels = images[:, :h].flatten(0, 2)
        state_pixels = images[:, h:].flatten(0, 2)
        z = self.encode(torch.cat([history_pixels, state_pixels]))
        z_history = view_major(z[: b * h * v], b, h, v)
        state_target = view_major(z[b * h * v :], b, cfg.num_states_pred, v)
        pad_key = f"{keys[0]}_is_pad"
        is_pad = batch.get(pad_key)
        if is_pad is None:
            is_pad = torch.zeros(b, t, dtype=torch.bool, device=images.device)
        history_pad = is_pad[:, :h].bool()
        state_valid = (~is_pad[:, h:].bool()).float()
        if v > 1:
            state_valid = state_valid.repeat(1, v)
        return z_history, history_pad, state_target, state_valid

    def _action_target(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        actions = batch[ACTION].float()
        pad = batch.get(f"{ACTION}_is_pad")
        if pad is None:
            return actions, torch.ones(actions.shape[:2], device=actions.device)
        pad = pad.bool()
        return actions.masked_fill(pad[..., None], 0.0), (~pad).float()

    @staticmethod
    def _per_sample_mse(pred: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        per_token = ((pred - target) ** 2).mean(-1) * valid
        return per_token.sum(-1) / valid.sum(-1).clamp(min=1)

    def _per_sample_losses(self, z_history, history_pad, action_target, action_valid, state_target, state_valid):
        m = self.model
        b, device = z_history.shape[0], z_history.device
        if m.action_head == "flow":
            tau_a = m._sample_flow_time(b, m.flow_alpha_act, device)
            action_noise = torch.randn_like(action_target)
            noisy_actions = torch.lerp(action_noise, action_target, tau_a[:, None, None])
        else:
            tau_a = action_noise = noisy_actions = None
        if m.state_head == "flow":
            tau_s = m._sample_flow_time(b, m.flow_alpha_state, device)
            state_noise = torch.randn_like(state_target)
            noisy_states = torch.lerp(state_noise, state_target, tau_s[:, None, None])
        else:
            tau_s = state_noise = noisy_states = None
        clean = action_target[:, : m.n_clean_actions]
        action_pred, state_pred = m.forward_mot(
            z_history, history_pad, clean, noisy_actions, noisy_states, tau_a, tau_s
        )
        act_target = action_target if m.action_head == "mse" else action_target - action_noise
        loss_action = self._per_sample_mse(action_pred, act_target, action_valid)
        if m.num_states_pred == 0:
            return loss_action, torch.zeros_like(loss_action)
        st_target = state_target if m.state_head == "mse" else state_target - state_noise
        return loss_action, self._per_sample_mse(state_pred, st_target, state_valid)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        cfg = self.config
        if cfg.goal_type != "none":
            raise NotImplementedError("lewam_jepa 0.1 trains goal-blind policies only (goal_type='none')")
        z_history, history_pad, state_target, state_valid = self._encode_training_batch(batch)
        action_target, action_valid = self._action_target(batch)
        if reduction == "mean":
            loss_action, loss_state = self.model.loss(
                z_history, history_pad, action_target, action_valid, state_target, state_valid
            )
        elif reduction == "none":
            loss_action, loss_state = self._per_sample_losses(
                z_history, history_pad, action_target, action_valid, state_target, state_valid
            )
        else:
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        loss = cfg.w_act * loss_action + cfg.w_dyn * loss_state
        out = {"loss_action": loss_action.mean().item(), "loss_state": loss_state.mean().item()}
        if cfg.w_reg > 0:
            loss_reg = sigreg_loss(self.sigreg, self.model, z_history, state_target, None, cfg.sigreg_grouped)
            loss = loss + cfg.w_reg * loss_reg
            out["loss_sigreg"] = loss_reg.item()
        if cfg.num_states_pred:
            out["latent_std"] = state_target.reshape(-1, state_target.shape[-1]).std(0).mean().item()
        return loss, out
