"""Base class for all RL algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from . import AlgorithmBaseCfg
    from cross_gym.envs import VecEnv


class AlgorithmBase(ABC):
    """Abstract base class for RL algorithms.
    
    All algorithms (PPO, SAC, PPO_AMP, DreamWaQ, etc.) inherit from this class.
    """

    def __init__(self, cfg: AlgorithmBaseCfg, env: VecEnv):
        """Initialize algorithm.
        
        Args:
            cfg: Algorithm configuration
            env: Environment instance
        """
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

    @abstractmethod
    def act(self, observations: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Generate actions from observations.
        
        Args:
            observations: Dictionary of observations
            **kwargs: Additional arguments
            
        Returns:
            Actions tensor (num_envs, action_dim)
        """
        pass

    @abstractmethod
    def process_env_step(
            self,
            rewards: torch.Tensor,
            terminated: torch.Tensor,
            truncated: torch.Tensor,
            infos: dict[str, Any],
            **kwargs
    ):
        """Process one environment step and store transition.
        
        Args:
            rewards: Reward tensor (num_envs,)
            terminated: Termination flags (num_envs,)
            truncated: Truncation flags (num_envs,)
            infos: Additional information dictionary
            **kwargs: Additional arguments
        """
        pass

    @abstractmethod
    def compute_returns(self, last_observations: dict[str, torch.Tensor]):
        """Compute returns and advantages for collected rollouts.
        
        Args:
            last_observations: Observations from last step
        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> dict[str, float]:
        """Update policy using collected rollouts.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of losses/metrics
        """
        pass

    @abstractmethod
    def reset(self, env_ids: torch.Tensor):
        """Reset algorithm state for specified environments.
        
        Args:
            env_ids: Environment IDs to reset
        """
        pass

    @abstractmethod
    def train(self):
        """Set algorithm to training mode."""
        pass

    @abstractmethod
    def eval(self):
        """Set algorithm to evaluation mode."""
        pass

    @abstractmethod
    def export(self) -> dict:
        """export algorithm state dict."""
        pass

    @abstractmethod
    def load(self, state_dict: dict):
        """Load algorithm state.
        
        Args:
            state_dict: State dict to load from.
        """
        pass
