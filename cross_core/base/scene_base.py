"""Abstract base class for interactive scene."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from cross_core.utils import configclass

if TYPE_CHECKING:
    from . import ArticulationBase, SensorBase


@configclass
class InteractiveSceneCfg(ABC):
    """Base class for scene configuration.
    
    All simulator-specific scene configs should inherit from this.
    Scene owns simulation initialization and scene building.
    
    Usage:
        scene = scene_cfg.class_type(scene_cfg, device)
    """

    class_type: type = MISSING
    num_envs: int = MISSING


class InteractiveScene(ABC):
    """Abstract base class for interactive scene.
    
    Scene owns everything:
    - Simulator initialization
    - Scene building (terrain, assets, envs)
    - Asset management (articulations, sensors)
    - Physics control (step, reset, render)
    """

    def __init__(self, cfg: InteractiveSceneCfg, device: torch.device):
        self.cfg = cfg
        self.device = device

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

    @abstractmethod
    def step(self, render: bool = True):
        """Step physics simulation.
        
        Args:
            render: Whether to render after stepping
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the simulation."""
        pass

    @abstractmethod
    def render(self):
        """Render the scene."""
        pass

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of parallel environments."""
        pass

    @property
    @abstractmethod
    def is_stopped(self) -> bool:
        """Whether simulation has been stopped."""
        pass
