"""Direct RL environment template (minimal base class)."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from cross_gym.scene import InteractiveScene
from cross_gym.sim import SimulationContext
from . import VecEnv, VecEnvStepReturn

if TYPE_CHECKING:
    from . import DirectRLEnvCfg


class DirectRLEnv(VecEnv):
    """Direct RL environment template.
    
    Minimal template for direct RL environments. Provides:
    - Scene management
    - Basic Gym interface (step, reset)
    - Abstract methods for user to implement
    
    Users inherit and implement:
    - compute_observations()
    - compute_rewards()
    - check_terminations()
    - process_actions() (optional)
    """

    def __init__(self, cfg: DirectRLEnvCfg):
        """Initialize direct RL environment.
        
        Args:
            cfg: Environment configuration
        """
        self.cfg = cfg

        # Create simulation
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = cfg.sim.class_type(cfg.sim)
        else:
            self.sim = SimulationContext.instance()

        # Create scene
        self.scene = InteractiveScene(cfg.scene)

        # Initialize VecEnv base
        super().__init__(num_envs=cfg.scene.num_envs, device=self.sim.device)

        # Timesteps
        self.dt = cfg.decimation * cfg.sim.dt
        self.decimation = cfg.decimation
        self.max_episode_length = int(cfg.episode_length_s / self.dt)

        # Minimal buffers
        self.obs_buf: dict[str, torch.Tensor] = {}
        self.reward_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.extras = {}

    # ===== Abstract Methods (Users Implement) =====

    @abstractmethod
    def compute_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations for the policy.
        
        Returns:
            Dictionary of observations
        """
        pass

    @abstractmethod
    def compute_rewards(self) -> torch.Tensor:
        """Compute rewards.
        
        Returns:
            Reward tensor (num_envs,)
        """
        pass

    @abstractmethod
    def check_terminations(self) -> torch.Tensor:
        """Check termination conditions.
        
        Returns:
            Boolean tensor indicating which environments should reset
        """
        pass

    def process_actions(self, actions: torch.Tensor):
        """Process and apply actions (users can override).
        
        Default: Do nothing. Users override to apply torques/commands.
        
        Args:
            actions: Policy actions
        """
        pass

    # ===== Gym Interface (Implemented) =====

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment.
        
        Args:
            actions: Actions (num_envs, action_dim)
            
        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        # Process actions (user implements)
        self.process_actions(actions)

        # Step simulation
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.physics_dt)

        # Update episode length
        self.episode_length_buf.add_(1)

        # Compute MDP components (user implements)
        self.reward_buf = self.compute_rewards()
        self.reset_buf = self.check_terminations()

        # Auto-reset
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)

        # Compute observations
        self.obs_buf = self.compute_observations()

        # Gym interface
        terminated = self.reset_buf.clone()
        truncated = torch.zeros_like(self.reset_buf)

        return self.obs_buf, self.reward_buf, terminated, truncated, self.extras

    def reset(self, env_ids=None) -> tuple:
        """Reset environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, reset all.
            
        Returns:
            Tuple of (observations, info)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        self._reset_idx(env_ids)
        self.obs_buf = self.compute_observations()

        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments (users can override for custom reset).
        
        Args:
            env_ids: Environment IDs to reset
        """
        # Reset scene
        self.scene.reset(env_ids)

        # Reset counters
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False

    def render(self):
        """Render the environment."""
        self.sim.render()

    def close(self):
        """Clean up resources."""
        SimulationContext.clear_instance()
