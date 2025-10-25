"""Neural networks for PPO algorithm."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from cross_gym.utils.configclass import configclass
from learning.modules import make_mlp

if TYPE_CHECKING:
    pass

# Disable validation for faster sampling
Normal.set_default_validate_args = False


@configclass
class ActorCriticCfg:
    """Configuration for Actor-Critic network."""

    actor_hidden_dims: list = [256, 256, 128]
    """Hidden dimensions for actor MLP."""

    critic_hidden_dims: list = [256, 256, 128]
    """Hidden dimensions for critic MLP."""

    activation: str = 'elu'
    """Activation function."""

    init_noise_std: float = 1.0
    """Initial action noise standard deviation."""


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO.
    
    This is a simple MLP-based actor-critic with Gaussian policy.
    """

    is_recurrent: bool = False

    def __init__(
            self,
            obs_shape: Dict[str, tuple],
            action_dim: int,
            actor_hidden_dims: List[int] = [256, 256, 128],
            critic_hidden_dims: List[int] = [256, 256, 128],
            activation: str = 'elu',
            init_noise_std: float = 1.0,
    ):
        """Initialize Actor-Critic network.
        
        Args:
            obs_shape: Dictionary of observation shapes
            action_dim: Action dimension
            actor_hidden_dims: Hidden layer sizes for actor
            critic_hidden_dims: Hidden layer sizes for critic
            activation: Activation function name
            init_noise_std: Initial action noise std
        """
        super().__init__()

        # Compute input dimensions
        # Assume 'policy' observation group is used
        if 'policy' in obs_shape:
            if isinstance(obs_shape['policy'], tuple):
                actor_input_dim = obs_shape['policy'][0]
            else:
                actor_input_dim = obs_shape['policy']
        else:
            # Fall back to first observation group
            first_key = list(obs_shape.keys())[0]
            if isinstance(obs_shape[first_key], tuple):
                actor_input_dim = obs_shape[first_key][0]
            else:
                actor_input_dim = obs_shape[first_key]

        # Critic uses same observations as actor by default
        critic_input_dim = actor_input_dim

        # Build actor network (outputs action mean)
        self.actor = make_mlp(
            input_dim=actor_input_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=action_dim,
            activation=activation,
        )

        # Build critic network (outputs state value)
        self.critic = make_mlp(
            input_dim=critic_input_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation,
        )

        # Action noise (learnable std)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Reset noise
        self.reset_std(init_noise_std)

        # Current distribution (set during act())
        self.distribution: Optional[Normal] = None

    def reset_std(self, std: float):
        """Reset action standard deviation.
        
        Args:
            std: New standard deviation value
        """
        self.log_std.data = torch.log(torch.tensor(std)).repeat(self.log_std.shape)

    def clip_std(self, min_std: float, max_std: float):
        """Clip action standard deviation to range.
        
        Args:
            min_std: Minimum std
            max_std: Maximum std
        """
        self.log_std.data = torch.clamp(
            self.log_std.data,
            math.log(min_std),
            math.log(max_std)
        )

    def act(self, observations: Dict[str, torch.Tensor], eval_mode: bool = False) -> torch.Tensor:
        """Generate actions from observations.
        
        Args:
            observations: Dictionary of observations
            eval_mode: If True, return deterministic actions (mean)
            
        Returns:
            Actions (num_envs, action_dim)
        """
        # Get policy observations
        if 'policy' in observations:
            obs = observations['policy']
        else:
            obs = observations[list(observations.keys())[0]]

        # Forward through actor
        action_mean = self.actor(obs)

        # Create distribution
        std = torch.exp(self.log_std)
        self.distribution = Normal(action_mean, std)

        # Sample or return mean
        if eval_mode:
            return action_mean
        else:
            return self.distribution.sample()

    def evaluate(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate value function.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            State values (num_envs, 1)
        """
        # Get critic observations (same as policy for now)
        if 'policy' in observations:
            obs = observations['policy']
        else:
            obs = observations[list(observations.keys())[0]]

        return self.critic(obs)

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


__all__ = ["ActorCritic", "ActorCriticCfg"]
