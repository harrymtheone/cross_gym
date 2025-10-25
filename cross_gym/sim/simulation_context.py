"""Abstract base class for simulation context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from cross_gym.sim.sim_cfg_base import SimCfgBase


class SimulationContext(ABC):
    """Abstract base class for simulator contexts.
    
    This class defines the common interface that all simulator-specific contexts
    must implement. It provides a unified API for:
    - Stepping the physics simulation
    - Resetting the simulation
    - Rendering the scene
    - Managing simulation state
    
    Each simulator (IsaacGym, Genesis, IsaacSim) implements this interface
    with its own specific backend.
    """

    # Class-level instance for singleton pattern
    _instance = None

    def __init__(self, cfg: SimCfgBase):
        """Initialize the simulation context.
        
        Args:
            cfg: Configuration for the simulation (SimCfgBase subclass).
        """
        self.cfg = cfg

        # Runtime validation
        if cfg.dt <= 0:
            raise ValueError(f"Physics timestep must be positive, got {cfg.dt}")
        if cfg.render_interval < 1:
            raise ValueError(f"Render interval must be at least 1, got {cfg.render_interval}")

        # Device
        self._device = torch.device(cfg.device)

        # Set as singleton instance
        if SimulationContext._instance is not None:
            raise RuntimeError(
                "Only one SimulationContext instance can exist at a time. "
                "Call SimulationContext.clear_instance() first."
            )
        SimulationContext._instance = self

        # Simulation state
        self._is_playing = False
        self._is_stopped = True
        self._sim_step_counter = 0

    @classmethod
    def instance(cls):
        """Get the singleton instance of SimulationContext.
        
        Returns:
            The current SimulationContext instance, or None if not created.
        """
        return cls._instance

    @classmethod
    def clear_instance(cls):
        """Clear the singleton instance."""
        if cls._instance is not None:
            cls._instance = None

    # ========== Core Properties ==========

    @property
    def device(self) -> torch.device:
        """The device on which simulation runs."""
        return self._device

    @property
    def physics_dt(self) -> float:
        """The physics timestep in seconds."""
        return self.cfg.dt

    @property
    def render_dt(self) -> float:
        """The rendering timestep in seconds."""
        return self.cfg.dt * self.cfg.render_interval

    @property
    def sim_step_counter(self) -> int:
        """The number of physics steps since simulation start/reset."""
        return self._sim_step_counter

    # ========== Simulation State ==========

    def is_playing(self) -> bool:
        """Check if simulation is currently playing."""
        return self._is_playing

    def is_stopped(self) -> bool:
        """Check if simulation is stopped."""
        return self._is_stopped

    # ========== Core Abstract Methods ==========

    @abstractmethod
    def reset(self):
        """Reset the simulation to initial state.
        
        This prepares the simulation for running by:
        - Initializing physics
        - Setting up the scene
        - Resetting all state
        """
        pass

    @abstractmethod
    def step(self, render: bool = True):
        """Step the physics simulation forward by one timestep.
        
        Args:
            render: Whether to render the scene after stepping.
        """
        pass

    @abstractmethod
    def render(self):
        """Render the current scene."""
        pass

    # ========== Scene Management ==========

    @abstractmethod
    def create_articulation_view(self, prim_path: str, num_envs: int) -> Any:
        """Create a view for articulated bodies (robots).
        
        Args:
            prim_path: Path pattern to the articulation (e.g., "/World/envs/env_.*/robot")
            num_envs: Number of parallel environments
            
        Returns:
            Simulator-specific articulation view object
        """
        pass

    @abstractmethod
    def create_rigid_object_view(self, prim_path: str, num_envs: int) -> Any:
        """Create a view for rigid bodies.
        
        Args:
            prim_path: Path pattern to the rigid objects
            num_envs: Number of parallel environments
            
        Returns:
            Simulator-specific rigid object view object
        """
        pass

    # ========== Utility Methods ==========

    def has_gui(self) -> bool:
        """Check if simulation has a GUI enabled.
        
        Returns:
            True if GUI is enabled, False for headless mode.
        """
        return not self.cfg.headless

    def get_physics_handle(self) -> Any:
        """Get simulator-specific physics handle.
        
        This allows access to simulator-specific features not covered
        by the abstract interface.
        
        Returns:
            Simulator-specific physics simulation handle
        """
        raise NotImplementedError(
            "Simulator-specific physics handle not implemented. "
            "Override this method in the simulator-specific context."
        )
