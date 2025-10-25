"""Configuration for on-policy runner."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Optional

from cross_gym.utils.configclass import configclass
from . import OnPolicyRunner


@configclass
class OnPolicyRunnerCfg:
    """Configuration for on-policy runner (PPO, A2C, etc.).
    
    Note: This does NOT contain env_cfg or algorithm_cfg.
    Those are in TaskCfg. This only has runner-specific settings.
    """

    class_type: type = OnPolicyRunner

    # ========== Training ==========
    max_iterations: int = MISSING
    """Maximum number of training iterations."""

    num_steps_per_update: int = 24
    """Number of environment steps to collect before each policy update."""

    # ========== Logging ==========
    log_interval: int = 1
    """Log every N iterations."""

    save_interval: int = 100
    """Save checkpoint every N iterations."""

    logger_backend: str = "tensorboard"
    """Logger backend ('tensorboard', 'wandb', or None)."""

    log_dir: str = "logs"
    """Root directory for logs."""

    project_name: str = MISSING
    """Project name for organizing logs."""

    experiment_name: str = MISSING
    """Experiment name/ID."""

    # ========== Resume Training ==========
    resume_path: Optional[str] = None
    """Path to checkpoint to resume from (None = start from scratch)."""

    load_optimizer: bool = True
    """Whether to load optimizer state when resuming."""


__all__ = ["OnPolicyRunnerCfg"]
