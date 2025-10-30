"""Abstract base class for interactive scene."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from cross_core.utils import configclass

if TYPE_CHECKING:
    from cross_core.base import ArticulationBase, SensorBase


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
    def step(self, render: bool = True):
        """Step physics simulation.
        
        Args:
            render: Whether to render after stepping
        """
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
