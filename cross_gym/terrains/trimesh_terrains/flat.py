"""Flat terrain generation."""

from __future__ import annotations

import numpy as np
import trimesh

from cross_gym.terrains.sub_terrain import SubTerrainBaseCfg, SubTerrain
from cross_gym.terrains.utils import create_rectangle
from cross_gym.utils.configclass import configclass


class FlatTerrain(SubTerrain):
    """Flat terrain sub-terrain."""

    def build_trimesh(self, difficulty: float) -> trimesh.Trimesh:
        """Build flat terrain mesh.
        
        Args:
            difficulty: Difficulty level [0, 1] (not used for flat)
            
        Returns:
            Flat rectangular trimesh
        """
        # Use create_rectangle utility for consistency
        return create_rectangle(
            size=self.cfg.size,
            height=0.0,
            up_left_center=True  # Origin at (0, 0)
        )

    def build_origins(self) -> tuple[float, float, float]:
        """Build spawn origin (center of terrain).
        
        Returns:
            Origin (x, y, z)
        """
        return self.cfg.size[0] / 2, self.cfg.size[1] / 2, 0.0

    def build_goals(self) -> np.ndarray | None:
        return None


@configclass
class FlatTerrainCfg(SubTerrainBaseCfg):
    """Configuration for flat terrain."""

    class_type: type = FlatTerrain


__all__ = ["FlatTerrain", "FlatTerrainCfg"]
