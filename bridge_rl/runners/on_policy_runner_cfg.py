"""Configuration for on-policy runner."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Optional

from cross_gym.envs import ManagerBasedRLEnvCfg
from bridge_rl.algorithms.ppo import PPOCfg
from cross_gym.utils.configclass import configclass
from . import OnPolicyRunner


@configclass
class OnPolicyRunnerCfg:
    """Configuration for on-policy runner (PPO, A2C, etc.)."""
    
    class_type: type = OnPolicyRunner
    
    # ========== Environment ==========
    env_cfg: ManagerBasedRLEnvCfg = MISSING
    """Environment configuration."""
    
    # ========== Algorithm ==========
    algorithm_cfg: PPOCfg = MISSING
    """Algorithm configuration (PPO, PPO_AMP, etc.)."""
    
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
    
    def __post_init__(self):
        """Post-initialization to link configs."""
        # Set num_steps_per_update in algorithm config
        self.algorithm_cfg.num_steps_per_update = self.num_steps_per_update


__all__ = ["OnPolicyRunnerCfg"]

