"""Abstract base class for interactive scene."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .articulation_base import ArticulationBase
    from .sensor_base import SensorBase


class SceneConfigBase(ABC):
    """Base class for scene configuration.
    
    All simulator-specific scene configs should inherit from this.
    """
    
    backend: str  # Identifier: "isaacgym", "genesis", etc.
    num_envs: int  # Number of parallel environments


class InteractiveSceneBase(ABC):
    """Abstract base class for interactive scene.
    
    This manages articulations, sensors, and terrain in a simulator-agnostic way.
    Each backend provides its own implementation.
    """
    
    @abstractmethod
    def get_articulation(self, name: str) -> ArticulationBase:
        """Get articulation by name.
        
        Args:
            name: Name of the articulation (from config attribute name)
            
        Returns:
            Articulation object
            
        Raises:
            ValueError: If articulation not found
        """
        pass
    
    @abstractmethod
    def get_sensor(self, name: str) -> SensorBase:
        """Get sensor by name.
        
        Args:
            name: Name of the sensor (from config attribute name)
            
        Returns:
            Sensor object
            
        Raises:
            ValueError: If sensor not found
        """
        pass
    
    @abstractmethod
    def get_terrain(self):
        """Get terrain object if configured.
        
        Returns:
            Terrain object or None
        """
        pass
    
    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of parallel environments."""
        pass

