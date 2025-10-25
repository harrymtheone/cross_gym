"""Base classes for sub-terrains."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING

import numpy as np
import trimesh

from cross_gym.utils import configclass


@configclass
class SubTerrainBaseCfg:
    """Base configuration for sub-terrains.
    
    A sub-terrain is an individual terrain patch (e.g., flat, slope, stairs).
    Each sub-terrain type has its own config class with class_type.
    """

    class_type: type[SubTerrain] = MISSING
    """SubTerrain class to instantiate."""

    proportion: float = 1.0
    """Proportion of this terrain type in the mix.
    
    Used for weighted sampling when multiple terrain types are specified.
    """

    size: tuple[float, float] = (8.0, 8.0)
    """Size of the sub-terrain (width_x, length_y) in meters."""


class SubTerrain(ABC):
    """Base class for sub-terrains.
    
    Each sub-terrain type inherits from this and implements:
    - build_trimesh(): Generate the terrain mesh
    - build_origins(): Compute spawn origin
    - build_goals(): Generate goal positions (optional)
    """

    def __init__(self, cfg: SubTerrainBaseCfg, difficulty: float):
        """Initialize sub-terrain.
        
        Args:
            cfg: Sub-terrain configuration
            difficulty: Difficulty level [0, 1]
        """
        self.cfg = cfg
        self.difficulty = difficulty

        # Generate trimesh
        self.mesh: trimesh.Trimesh = self.build_trimesh(difficulty)

        # Build origin (spawn point for robots)
        self.origin: tuple[float, float, float] = self.build_origins()

        # Frame origin (bottom-left corner, set by terrain generator during merging)
        self.frame_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @abstractmethod
    def build_trimesh(self, difficulty: float) -> trimesh.Trimesh:
        """Build trimesh for this sub-terrain.
        
        Args:
            difficulty: Difficulty level [0, 1]
            
        Returns:
            trimesh.Trimesh object
        """
        pass

    @abstractmethod
    def build_origins(self) -> tuple[float, float, float]:
        """Build spawn origin for this sub-terrain.
        
        Returns:
            Origin (x, y, z) relative to sub-terrain frame (before transformation)
        """
        pass

    @abstractmethod
    def build_goals(self) -> np.ndarray | None:
        """Build goal positions for this sub-terrain (optional).
        
        Returns:
            Goal positions array or None if no goals
        """
        pass
