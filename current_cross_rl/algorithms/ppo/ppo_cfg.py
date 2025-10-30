"""Configuration for PPO algorithm."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from cross_gym.utils import configclass
from cross_rl.algorithms import AlgorithmBaseCfg
from . import PPO, ActorCriticCfg


@configclass
class PPOCfg(AlgorithmBaseCfg):
    """Configuration for PPO algorithm."""

    class_type: type = PPO

    # ========== Network Parameters ==========
    actor_critic: ActorCriticCfg = ActorCriticCfg()

    noise_std_range: tuple[float, float] = (0.3, 1.0)
    """Range to clip action noise std."""

    # ========== PPO Parameters ==========
    gamma: float = 0.99
    """Discount factor."""

    lam: float = 0.95
    """GAE lambda parameter."""

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

    # ========== Learning Rate ==========
    learning_rate: float = 1e-3
    """Learning rate for optimizer."""

    learning_rate_schedule: Literal["adaptive", "fixed"] = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float | None = 0.01
    """Desired KL divergence for adaptive cross_rl rate (None = disabled)."""

    # ========== Training Settings ==========
    max_grad_norm: float | None = 1.0
    """Maximum gradient norm for clipping (None = no clipping)."""

    use_amp: bool = False
    """Whether to use automatic mixed precision."""

    num_steps_per_update: int = MISSING
    """Number of steps to collect before each update (set by runner)."""
