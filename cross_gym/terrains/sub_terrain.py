"""Base classes for sub-terrains."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING
from enum import Enum

import numpy as np
import trimesh

from cross_gym.utils import configclass


class TerrainTypeID(Enum):
    """Terrain type identifiers."""
    # Flat terrains (0-1)
    flat = 0
    rough = 1

    # Stair terrains (2-99)
    stairs_up = 2
    stairs_down = 3
    huge_stair = 4
    discrete = 5
    stepping_stone = 6
    gap = 7
    pit = 8
    slope = 9

    # Parkour terrains (100+)
    parkour_flat = 100
    parkour_hurdle = 101
    parkour_gap = 102
    parkour_box = 103
    parkour_step = 104
    parkour_stair = 105
    parkour_stair_down = 106


class TerrainCommandType(Enum):
    """Command mode for different terrain types."""
    Omni = 0  # Omnidirectional
    Heading = 1  # Heading-based
    Goal = 2  # Goal-based


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
    - terrain_type_id: Class variable with TerrainTypeID
    - command_type: Class variable with TerrainCommandType
    - build_trimesh(): Generate the terrain mesh
    - build_origins(): Compute spawn origin
    - build_goals(): Generate goal positions (optional)
    """

    # Class attributes (must be set by subclasses)
    terrain_type_id: TerrainTypeID = None
    """Terrain type identifier."""

    command_type: TerrainCommandType = None
    """Command mode (Omni, Heading, or Goal)."""

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
