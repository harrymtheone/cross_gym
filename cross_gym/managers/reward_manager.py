"""Reward manager for computing rewards."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING, Any

import torch

from . import ManagerBase, ManagerTermCfg

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


class RewardManager(ManagerBase):
    """Manager for computing rewards.
    
    The reward manager computes the total reward as a weighted sum of individual reward terms:
        total_reward = sum(weight_i * reward_i)
    
    Each reward term is a function that takes the environment and returns a reward tensor.
    """

    def __init__(self, cfg: Any, env: ManagerBasedEnv):
        """Initialize reward manager.
        
        Args:
            cfg: Configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Reward terms
        self.reward_terms: dict[str, tuple[Callable, dict, float]] = {}

        # Parse configuration
        self._prepare_terms()

        # Logging buffers
        self.episode_sums: dict[str, torch.Tensor] = {}
        for name in self.reward_terms.keys():
            self.episode_sums[name] = torch.zeros(env.num_envs, device=env.device)

    def _prepare_terms(self):
        """Parse configuration and create reward terms."""
        for attr_name in dir(self.cfg):
            if attr_name.startswith("_"):
                continue

            attr_value = getattr(self.cfg, attr_name)

            if isinstance(attr_value, ManagerTermCfg):
                func = attr_value.func
                params = attr_value.params if hasattr(attr_value, 'params') else {}
                weight = attr_value.weight if hasattr(attr_value, 'weight') else 1.0

                self.reward_terms[attr_name] = (func, params, weight)
                self.active_terms[attr_name] = attr_value

    def compute(self, dt: float) -> torch.Tensor:
        """Compute total reward.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Total reward for each environment (num_envs,)
        """
        total_reward = torch.zeros(self._env.num_envs, device=self._env.device)

        for name, (func, params, weight) in self.reward_terms.items():
            # Compute individual reward
            reward = func(self._env, **params)

            # Apply weight
            weighted_reward = weight * reward

            # Accumulate
            total_reward += weighted_reward

            # Log episode sum
            self.episode_sums[name] += weighted_reward

        return total_reward

    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        """Reset reward manager.
        
        Args:
            env_ids: Environment IDs to reset
            
        Returns:
            Dictionary of episode reward sums (for logging)
        """
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)

        # Collect episode sums for logging
        info = {}
        for name, episode_sum in self.episode_sums.items():
            # Only log for reset environments
            if len(env_ids) > 0:
                info[f"Episode_Reward/{name}"] = episode_sum[env_ids].mean().item()

            # Reset episode sums
            episode_sum[env_ids] = 0.0

        return info
