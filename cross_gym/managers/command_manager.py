"""Command manager for generating goal commands."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from cross_gym.utils.configclass import configclass
from .manager_base import ManagerBase

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


@configclass
class CommandManagerCfg:
    """Configuration for command manager.
    
    Commands are goals/references that the policy should track
    (e.g., desired velocity, target pose).
    """
    pass  # Command generators are added as attributes


class CommandManager(ManagerBase):
    """Manager for generating and managing commands.
    
    Commands are goal references for the policy to track, such as:
    - Desired base velocity (for locomotion)
    - Target end-effector pose (for manipulation)
    - Trajectory waypoints
    
    The command manager generates new commands at specified intervals
    and provides them to the policy through observations.
    """
    
    def __init__(self, cfg: CommandManagerCfg, env: ManagerBasedEnv):
        """Initialize command manager.
        
        Args:
            cfg: Configuration
            env: Environment instance
        """
        super().__init__(cfg, env)
        
        # Command buffers (to be filled by subclasses/terms)
        # For now, just a placeholder
        self.has_commands = False
    
    def compute(self, dt: float):
        """Update commands.
        
        This is called every environment step to potentially generate new commands.
        
        Args:
            dt: Time step in seconds
        """
        # Placeholder - will be filled when we implement specific command generators
        pass
    
    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        """Reset commands for specified environments.
        
        Args:
            env_ids: Environment IDs to reset
            
        Returns:
            Dictionary of reset information
        """
        # Placeholder
        return {}

