"""Manager-based RL environment with Gym interface."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

import numpy as np
import torch

from cross_gym.managers import RewardManager, TerminationManager, CommandManager
from .manager_based_env import ManagerBasedEnv
from .vec_env import VecEnvStepReturn

if TYPE_CHECKING:
    from . import ManagerBasedRLEnvCfg


class ManagerBasedRLEnv(ManagerBasedEnv):
    """Manager-based RL environment with Gymnasium interface.
    
    This class extends ManagerBasedEnv with RL-specific functionality:
    - Reward computation (via RewardManager)
    - Termination checking (via TerminationManager)
    - Command generation (via CommandManager)
    - Gymnasium-compatible interface
    
    The environment is vectorized - it runs multiple environment instances in parallel.
    """

    is_vector_env: bool = True
    """Whether this is a vectorized environment."""

    metadata: Dict[str, Any] = {
        "render_modes": [None, "human"],
    }
    """Environment metadata."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the RL environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: Optional[str] = None, **kwargs):
        """Initialize the RL environment.
        
        Args:
            cfg: Environment configuration
            render_mode: Rendering mode (None or "human")
            **kwargs: Additional arguments
        """
        # Initialize base environment
        super().__init__(cfg)

        # Store render mode
        self.render_mode = render_mode

        # Set up RL-specific managers
        self._setup_rl_managers()

        # Reset buffers
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Reward buffer
        self.reward_buf = torch.zeros(self.num_envs, device=self.device)

        # Observation buffer
        self.obs_buf: Dict[str, torch.Tensor] = {}

        print("[INFO] RL Environment setup complete!")

    def _setup_rl_managers(self):
        """Set up RL-specific managers."""
        # Reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print(f"[INFO] Reward Manager: {len(self.reward_manager.active_terms)} terms")

        # Termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print(f"[INFO] Termination Manager: {len(self.termination_manager.active_terms)} terms")

        # Command manager (optional)
        if self.cfg.commands is not None:
            self.command_manager = CommandManager(self.cfg.commands, self)
            print(f"[INFO] Command Manager initialized")
        else:
            self.command_manager = None

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        """Execute one environment step and handle resets.
        
        Args:
            actions: Actions to apply (num_envs, action_dim)
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # Call parent step (processes actions, steps sim, computes obs)
        self.obs_buf = super().step(actions)

        # Compute rewards
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # Check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_truncated = self.termination_manager.time_outs

        # Update commands
        if self.command_manager is not None:
            self.command_manager.compute(dt=self.step_dt)

        # Reset environments that terminated
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # Return Gym-style outputs
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_truncated, self.extras

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: Random seed (optional)
            options: Additional options (optional)
            
        Returns:
            Tuple of (observations, info)
        """
        # Set seed if provided
        if seed is not None:
            self._set_seed(seed)

        # Reset all environments
        self.obs_buf = super().reset(env_ids=None)

        # Reset episode length
        self.episode_length_buf.zero_()

        # Reset buffers
        self.reset_buf.zero_()
        self.reset_terminated.zero_()
        self.reset_truncated.zero_()
        self.reward_buf.zero_()

        # Return observations and info
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments (internal).
        
        Args:
            env_ids: Environment IDs to reset
        """
        # Call parent reset
        super().reset(env_ids)

        # Reset RL-specific managers
        reward_info = self.reward_manager.reset(env_ids)
        self.termination_manager.reset(env_ids)
        if self.command_manager is not None:
            self.command_manager.reset(env_ids)

        # Store reward info in extras for logging
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(reward_info)

        # Compute new observations for reset environments
        self.obs_buf = self.observation_manager.compute()

    def render(self) -> np.ndarray | None:
        """Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", otherwise None
        """
        if self.render_mode == "human" or self.render_mode is None:
            super().render()
            return None
        elif self.render_mode == "rgb_array":
            # TODO: Implement image capture
            raise NotImplementedError("rgb_array rendering not yet implemented")
        else:
            raise NotImplementedError(f"Render mode '{self.render_mode}' not supported")

    def close(self):
        """Clean up resources."""
        super().close()
