"""Neural networks for PPO algorithm."""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from cross_gym.utils import configclass
from cross_rl.modules import make_mlp

# Disable validation for faster sampling
Normal.set_default_validate_args = False


@configclass
class ActorCriticCfg:
    """Configuration for Actor-Critic network."""

    actor_input_size: int = MISSING
    """Observation shapes. Dict mapping obs name to shape tuple."""

    actor_hidden_dims: Sequence[int] = [512, 256, 128]
    """Hidden dimensions for actor MLP."""

    action_size: int = MISSING
    """Action dimension."""

    critic_obs_shape: tuple[int, int] = MISSING
    """Critic observation shape."""

    scan_shape: tuple[int, int] = MISSING
    """Scan observation shape."""

    critic_hidden_dims: Sequence[int] = [512, 256, 128]
    """Hidden dimensions for critic MLP."""

    activation: str = "elu"
    """Activation function."""

    init_noise_std: float = 1.0
    """Initial action noise standard deviation."""


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, cfg: ActorCriticCfg):
        """Initialize Actor-Critic network.
        
        Args:
            cfg: Actor-Critic configuration containing:
                - obs_shape: Dictionary of observation shapes
                - action_dim: Action dimension
                - actor_hidden_dims: Hidden layer sizes for actor
                - critic_hidden_dims: Hidden layer sizes for critic
                - activation: Activation function name
                - init_noise_std: Initial action noise std
        """
        super().__init__()

        # Build actor network (outputs action mean)
        self.actor = make_mlp(
            input_dim=cfg.actor_input_size,
            hidden_dims=cfg.actor_hidden_dims,
            output_dim=cfg.action_size,
            activation=cfg.activation,
        )

        # Build critic network (outputs state value)
        critic_input_size = math.prod(cfg.critic_obs_shape)
        self.priv_enc = make_mlp(
            input_dim=critic_input_size,
            hidden_dims=(256, 128),
            output_dim=64,
            activation=cfg.activation,
            output_activation=cfg.activation,
        )

        scan_size = math.prod(cfg.scan_shape)
        self.scan_enc = make_mlp(
            input_dim=scan_size,
            hidden_dims=(256, 128),
            output_dim=64,
            activation=cfg.activation,
            output_activation=cfg.activation,
        )
        self.edge_enc = make_mlp(
            input_dim=scan_size,
            hidden_dims=(256, 128),
            output_dim=64,
            activation=cfg.activation,
            output_activation=cfg.activation,
        )

        self.critic = make_mlp(
            input_dim=64 + 64 + 64,
            hidden_dims=cfg.critic_hidden_dims,
            output_dim=1,
            activation=cfg.activation,
        )

        # Action noise (learnable std)
        self.log_std = nn.Parameter(torch.zeros(cfg.action_size))

        # Reset noise
        self.reset_std(cfg.init_noise_std)

        # Current distribution (set during act())
        self.distribution: Normal | None = None

    def act(self, x: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        """Generate actions from observations.
        
        Args:
            x: Input tensor
            eval_mode: If True, return deterministic actions (mean)
            
        Returns:
            Actions (num_envs, action_dim)
        """
        action_mean = self.actor(x)

        if eval_mode:
            return action_mean

        self.distribution = Normal(action_mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def evaluate(self, priv: torch.Tensor, scan: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """Evaluate value function.
        
        Args:
            priv: Private observation
            scan: Scan observation
            edge: Edge observation
            
        Returns:
            State values (num_envs, 1)
        """
        priv_enc = self.priv_enc(priv)
        scan_enc = self.scan_enc(scan)
        edge_enc = self.edge_enc(edge)

        return self.critic(torch.cat([priv_enc, scan_enc, edge_enc], dim=-1))

    def reset_std(self, std: float):
        """Reset action standard deviation.
        
        Args:
            std: New standard deviation value
        """
        std = torch.full(self.log_std.shape, std)
        self.log_std.data = torch.log(std)

    def clip_std(self, min_std: float, max_std: float):
        """Clip action standard deviation to range.
        
        Args:
            min_std: Minimum std
            max_std: Maximum std
        """
        self.log_std.data = torch.clamp(
            self.log_std.data, math.log(min_std), math.log(max_std)
        )

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions under current distribution.
        
        Args:
            actions: Actions tensor
            
        Returns:
            Log probabilities (num_envs, 1)
        """
        if self.distribution is None:
            raise RuntimeError("Must call act() before get_actions_log_prob()")

        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    @property
    def action_mean(self) -> torch.Tensor:
        """Get mean of current action distribution."""
        if self.distribution is None:
            raise RuntimeError("Must call act() before accessing action_mean")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Get std of current action distribution."""
        if self.distribution is None:
            raise RuntimeError("Must call act() before accessing action_std")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Get entropy of current action distribution."""
        if self.distribution is None:
            raise RuntimeError("Must call act() before accessing entropy")
        return self.distribution.entropy().sum(dim=-1, keepdim=True)
