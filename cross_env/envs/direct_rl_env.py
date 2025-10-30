"""Direct RL environment template (minimal base class)."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from cross_core.base import InteractiveScene
from .vec_env import VecEnv, VecEnvStepReturn

if TYPE_CHECKING:
    from .direct_rl_env_cfg import DirectRLEnvCfg


class DirectRLEnv(VecEnv):
    """Direct RL environment template.
    
    Minimal template for direct RL environments. Provides:
    - Scene management (owns simulation)
    - Basic Gym interface (step, reset)
    - Abstract methods for user to implement
    
    Users inherit and implement:
    - compute_observations()
    - compute_rewards()
    - check_terminations()
    - process_actions() (optional)
    """

    def __init__(self, cfg: DirectRLEnvCfg, device: torch.device):
        """Initialize direct RL environment.
        
        Args:
            cfg: Environment configuration
            device: Device to run simulation on
        """
        self.cfg = cfg

        # Create scene (scene owns simulation initialization)
        self.scene: InteractiveScene = cfg.scene.class_type(cfg.scene, device)

        # Initialize VecEnv base
        super().__init__(num_envs=cfg.scene.num_envs, device=device)

        # Timesteps
        self.dt = cfg.decimation * cfg.scene.sim.dt
        self.decimation = cfg.decimation
        self.max_episode_length = int(cfg.episode_length_s / self.dt)

        # -- counter for simulation steps
        self._sim_step_counter = 0
        # -- counter for curriculum
        self.common_step_counter = 0

        # Buffers
        self._ALL_ENVS = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.obs_buf: dict[str, torch.Tensor] = {}
        self.reward_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.extras = {}

    @property
    def step_dt(self):
        return self.cfg.decimation * self.cfg.scene.sim.dt

    @property
    def physics_dt(self):
        return self.cfg.scene.sim.dt

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
    def check_terminations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions.
        
        Returns:
            Tuple of (terminated, truncated) boolean tensors
        """
        pass

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Process actions before applying to simulation.
        
        Args:
            actions: Raw policy actions
            
        Returns:
            Processed actions (default: return as-is)
        """
        return actions

    # ===== Gym Interface =====

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment.
        
        Args:
            actions: Actions from policy (num_envs, action_dim)
            
        Returns:
            (observations, rewards, dones, infos)
        """
        # Process actions
        processed_actions = self.process_actions(actions)
        
        # Apply actions to simulation (user implements this)
        self.apply_actions(processed_actions)

        # Step simulation
        for _ in range(self.decimation):
            self.scene.step(render=False)
            self._sim_step_counter += 1
        
        # Render if needed
        if self._sim_step_counter % (self.decimation * self.cfg.scene.render_interval) == 0:
            self.scene.render()

        # Compute observations
        self.obs_buf = self.compute_observations()

        # Compute rewards
        self.reward_buf = self.compute_rewards()

        # Check terminations
        self.reset_terminated, self.reset_truncated = self.check_terminations()
        self.reset_buf = self.reset_terminated | self.reset_truncated

        # Update episode lengths
        self.episode_length_buf += 1
        
        # Reset terminated/truncated environments
        if self.reset_buf.any():
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).squeeze(-1))
        
        # Increment step counter
        self.common_step_counter += 1
        
        # Package extras
        self.extras = {
            "time_outs": self.reset_truncated,
            "episode_length": self.episode_length_buf[self.reset_buf].float().mean() if self.reset_buf.any() else 0.0,
        }
        
        # Return observation tensor (assume "policy" key)
        obs = self.obs_buf.get("policy", next(iter(self.obs_buf.values())))
        
        return obs, self.reward_buf, self.reset_buf, self.extras

    def reset(self) -> torch.Tensor:
        """Reset all environments.
            
        Returns:
            Initial observations
        """
        # Reset simulation
        self.scene.reset()
        
        # Reset buffers
        self.reset_buf.fill_(False)
        self.reset_terminated.fill_(False)
        self.reset_truncated.fill_(False)
        self.episode_length_buf.fill_(0)
        self._sim_step_counter = 0
        self.common_step_counter = 0
        
        # Compute initial observations
        self.obs_buf = self.compute_observations()

        # Return observation tensor
        return self.obs_buf.get("policy", next(iter(self.obs_buf.values())))

    def reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments.
        
        Args:
            env_ids: Environment indices to reset
        """
        if len(env_ids) == 0:
            return

        # Reset episode lengths
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.reset_terminated[env_ids] = False
        self.reset_truncated[env_ids] = False

        # User implements specific reset logic
        self.reset_env_specific(env_ids)

    @abstractmethod
    def reset_env_specific(self, env_ids: torch.Tensor):
        """Reset specific environment states (user implements).
        
        Args:
            env_ids: Environment indices to reset
        """
        pass

    @abstractmethod
    def apply_actions(self, actions: torch.Tensor):
        """Apply actions to simulation (user implements).
        
        Args:
            actions: Processed actions to apply
        """
        pass
