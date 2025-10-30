"""Abstract base class for simulation context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scene_base import SceneConfigBase


class SimulationConfigBase(ABC):
    """Base class for simulation configuration.
    
    All simulator-specific configs should inherit from this.
    """
    
    backend: str  # Identifier: "isaacgym", "genesis", etc.


class SimulationContextBase(ABC):
    """Abstract base class for simulation context.
    
    This defines the interface that all simulator backends must implement.
    Each backend (IsaacGym, Genesis, etc.) provides its own implementation.
    """
    
    def __init__(self, cfg: SimulationConfigBase):
        """Initialize simulation context.
        
        Args:
            cfg: Simulation configuration
        """
        self.cfg = cfg
        self.device = None  # Set by implementation
        self._sim_step_counter = 0
        self._is_playing = False
        self._is_stopped = False
    
    @abstractmethod
    def build_scene(self, scene_cfg: SceneConfigBase):
        """Build the simulation scene from configuration.
        
        Args:
            scene_cfg: Scene configuration with articulations, sensors, terrain
        """
        pass
    
    @abstractmethod
    def step(self, render: bool = True):
        """Step the physics simulation forward by one timestep.
        
        Args:
            render: Whether to render the scene after stepping
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the simulation to initial state."""
        pass
    
    @abstractmethod
    def render(self):
        """Render the current scene."""
        pass
    
    @property
    def dt(self) -> float:
        """Simulation timestep."""
        return self.cfg.dt
    
    @property
    def is_playing(self) -> bool:
        """Whether simulation is currently playing."""
        return self._is_playing
    
    @property
    def is_stopped(self) -> bool:
        """Whether simulation has been stopped."""
        return self._is_stopped
    
    @property
    def sim_step_counter(self) -> int:
        """Number of simulation steps taken."""
        return self._sim_step_counter

