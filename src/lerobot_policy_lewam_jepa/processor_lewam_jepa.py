from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_lewam_jepa import LeWAMJEPAConfig


def make_lewam_jepa_pre_post_processors(
    config: LeWAMJEPAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_default_pre_post_processors(config, dataset_stats)
