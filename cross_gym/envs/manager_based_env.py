"""Manager-based environment implementation."""

from __future__ import annotations

import random
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import torch

from cross_gym.managers import (
    ActionManager,
    ObservationManager,
    EventManager,
)
from cross_gym.scene import InteractiveScene
from cross_gym.sim import SimulationContext

if TYPE_CHECKING:
    from . import ManagerBasedEnvCfg


class ManagerBasedEnv:
    """Base environment using the manager-based workflow.
    
    This environment provides the core functionality for simulation-based RL:
    - Scene management (robots, objects, sensors)
    - Action processing (via ActionManager)
    - Observation computation (via ObservationManager)
    - Event handling (via EventManager)
    
    It does NOT include:
    - Reward computation (added in ManagerBasedRLEnv)
    - Termination checking (added in ManagerBasedRLEnv)
    - Gym interface (added in ManagerBasedRLEnv)
    
    This separation allows using the same environment for different purposes
    (e.g., data collection, imitation cross_gym_rl, RL).
    """

    def __init__(self, cfg: ManagerBasedEnvCfg):
        """Initialize the environment.
        
        Args:
            cfg: Environment configuration
        """
        self.cfg = cfg

        # Runtime validation
        if cfg.decimation < 1:
            raise ValueError(f"Decimation must be >= 1, got {cfg.decimation}")

        # Set random seed
        if self.cfg.seed is not None:
            self._set_seed(self.cfg.seed)

        # Create simulation context using class_type from config
        if SimulationContext.instance() is None:
            # Get the simulator class from the config
            if not hasattr(self.cfg.sim, 'class_type'):
                raise ValueError(
                    f"Simulation config must have 'class_type' attribute. "
                    f"Use IsaacGymCfg, GenesisCfg, or IsaacSimCfg."
                )

            sim_class = self.cfg.sim.class_type
            self.sim: SimulationContext = sim_class(self.cfg.sim)
        else:
            self.sim = SimulationContext.instance()

        # Store common properties
        self.device = self.sim.device
        self.num_envs = self.cfg.scene.num_envs

        # Create scene
        print("[INFO] Creating scene...")
        self.scene = InteractiveScene(self.cfg.scene)
        print(f"[INFO] Scene created with {self.num_envs} environments")

        # Create managers
        print("[INFO] Setting up managers...")
        self._setup_managers()

        # Environment counters
        self._sim_step_counter = 0
        self.common_step_counter = 0

        # Episode length tracking
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # Extras dictionary for logging
        self.extras: Dict[str, Any] = {}

        print("[INFO] Environment setup complete!")

    def _set_seed(self, seed: int) -> int:
        """Set random seed for reproducibility.
        
        Args:
            seed: Random seed
            
        Returns:
            The seed that was set
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        return seed

    def _setup_managers(self):
        """Create and initialize all managers."""
        # Action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print(f"[INFO] Action Manager: {len(self.action_manager.active_terms)} terms")

        # Observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print(f"[INFO] Observation Manager: {len(self.observation_manager.active_terms)} terms")

        # Event manager (optional)
        if self.cfg.events is not None:
            self.event_manager = EventManager(self.cfg.events, self)
            print(f"[INFO] Event Manager: {len(self.event_manager.active_terms)} terms")

            # Apply startup events
            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")
        else:
            self.event_manager = None

    @property
    def physics_dt(self) -> float:
        """Physics simulation time step."""
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """Environment step time step."""
        return self.cfg.decimation * self.cfg.sim.dt

    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute one time-step of the environment.
        
        Args:
            actions: Actions to apply (num_envs, action_dim)
            
        Returns:
            Dictionary of observations
        """
        # Process actions
        self.action_manager.process_action(actions)

        # Perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            # Apply actions
            self.action_manager.apply_action()

            # Write data to simulator
            self.scene.write_data_to_sim()

            # Step simulation
            self.sim.step(render=False)

            # Update scene (read from simulator)
            self.scene.update(self.physics_dt)

        # Update counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Apply interval events
        if self.event_manager is not None and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # Compute observations
        obs = self.observation_manager.compute()

        return obs

    def reset(self, env_ids: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Reset specified environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, reset all.
            
        Returns:
            Dictionary of observations
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # Apply reset events
        if self.event_manager is not None and "reset" in self.event_manager.available_modes:
            self.event_manager.apply(mode="reset", env_ids=env_ids)

        # Reset scene
        self.scene.reset(env_ids)

        # Reset managers
        self.action_manager.reset(env_ids)
        self.observation_manager.reset(env_ids)
        if self.event_manager is not None:
            self.event_manager.reset(env_ids)

        # Reset episode length
        self.episode_length_buf[env_ids] = 0

        # Compute observations
        obs = self.observation_manager.compute()

        return obs

    def render(self):
        """Render the environment."""
        self.sim.render()

    def close(self):
        """Clean up resources."""
        # Clear simulation context
        SimulationContext.clear_instance()

    def __del__(self):
        """Destructor."""
        self.close()
