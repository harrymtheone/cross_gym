"""Configuration for PPO algorithm."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal, Optional

from cross_gym.utils.configclass import configclass
from . import PPO


@configclass
class PPOCfg:
    """Configuration for PPO algorithm."""
    
    class_type: type = PPO
    
    # ========== RL Parameters ==========
    gamma: float = 0.99
    """Discount factor."""
    
    lam: float = 0.95
    """GAE lambda parameter."""
    
    # ========== PPO Parameters ==========
    clip_param: float = 0.2
    """PPO clipping parameter."""
    
    num_mini_batches: int = 4
    """Number of mini-batches for PPO update."""
    
    num_learning_epochs: int = 5
    """Number of epochs to train on each rollout."""
    
    value_loss_coef: float = 1.0
    """Coefficient for value loss."""
    
    entropy_coef: float = 0.01
    """Coefficient for entropy bonus."""
    
    use_clipped_value_loss: bool = True
    """Whether to use clipped value loss."""
    
    # ========== Network Parameters ==========
    actor_hidden_dims: list = [256, 256, 128]
    """Hidden layer dimensions for actor network."""
    
    critic_hidden_dims: list = [256, 256, 128]
    """Hidden layer dimensions for critic network."""
    
    activation: str = 'elu'
    """Activation function name."""
    
    # ========== Action Noise ==========
    init_noise_std: float = 1.0
    """Initial standard deviation for action noise."""
    
    noise_std_range: tuple = (0.3, 1.5)
    """Range to clip action noise std."""
    
    # ========== Learning Rate ==========
    learning_rate: float = 1e-3
    """Learning rate for optimizer."""
    
    learning_rate_schedule: Literal["adaptive", "fixed"] = "adaptive"
    """Learning rate schedule type."""
    
    desired_kl: Optional[float] = 0.01
    """Desired KL divergence for adaptive learning rate (None = disabled)."""
    
    # ========== Training Settings ==========
    max_grad_norm: Optional[float] = 1.0
    """Maximum gradient norm for clipping (None = no clipping)."""
    
    use_amp: bool = False
    """Whether to use automatic mixed precision."""
    
    # ========== This will be set by runner ==========
    num_steps_per_update: int = MISSING
    """Number of steps to collect before each update (set by runner)."""

