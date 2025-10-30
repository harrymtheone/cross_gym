"""Abstract base class for interactive scene."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING
from typing import TYPE_CHECKING

from cross_core.utils import configclass

if TYPE_CHECKING:
    from . import ArticulationBase, SensorBase


@configclass
class SceneBaseCfg(ABC):
    """Base class for scene configuration.
    
    All simulator-specific scene configs should inherit from this.
    The class_type attribute should reference the scene class.
    
    Usage:
        scene = scene_cfg.class_type(scene_cfg, sim_context)
    """

    class_type: type[InteractiveSceneBase] = MISSING
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
