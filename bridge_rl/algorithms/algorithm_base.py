"""Base class for all RL algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any

import torch

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedRLEnv


class AlgorithmBase(ABC):
    """Abstract base class for RL algorithms.
    
    All algorithms (PPO, SAC, PPO_AMP, DreamWaQ, etc.) inherit from this class.
    """
    
    def __init__(self, cfg: Any, env: ManagerBasedRLEnv):
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
    def act(self, observations: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
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
        infos: Dict[str, Any],
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
    def compute_returns(self, last_observations: Dict[str, torch.Tensor]):
        """Compute returns and advantages for collected rollouts.
        
        Args:
            last_observations: Observations from last step
        """
        pass
    
    @abstractmethod
    def update(self, **kwargs) -> Dict[str, float]:
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
    
    def train(self):
        """Set algorithm to training mode."""
        if hasattr(self, 'actor_critic'):
            self.actor_critic.train()
        if hasattr(self, 'actor'):
            self.actor.train()
        if hasattr(self, 'critic'):
            self.critic.train()
    
    def eval(self):
        """Set algorithm to evaluation mode."""
        if hasattr(self, 'actor_critic'):
            self.actor_critic.eval()
        if hasattr(self, 'actor'):
            self.actor.eval()
        if hasattr(self, 'critic'):
            self.critic.eval()
    
    def save(self, path: str):
        """Save algorithm state.
        
        Args:
            path: Path to save checkpoint
        """
        save_dict = {
            'cfg': self.cfg,
        }
        
        if hasattr(self, 'actor_critic'):
            save_dict['actor_critic'] = self.actor_critic.state_dict()
        if hasattr(self, 'actor'):
            save_dict['actor'] = self.actor.state_dict()
        if hasattr(self, 'critic'):
            save_dict['critic'] = self.critic.state_dict()
        if hasattr(self, 'optimizer'):
            save_dict['optimizer'] = self.optimizer.state_dict()
        
        torch.save(save_dict, path)
    
    def load(self, path: str, load_optimizer: bool = True):
        """Load algorithm state.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if hasattr(self, 'actor_critic') and 'actor_critic' in checkpoint:
            self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        if hasattr(self, 'actor') and 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
        if hasattr(self, 'critic') and 'critic' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic'])
        if load_optimizer and hasattr(self, 'optimizer') and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

