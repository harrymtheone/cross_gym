"""Flat terrain generation."""

from __future__ import annotations

import numpy as np
import trimesh

from cross_gym.terrains.sub_terrain import SubTerrainBaseCfg, SubTerrain
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
        # Create a simple flat rectangle
        vertices = np.array([
            [0, 0, 0],
            [self.cfg.size[0], 0, 0],
            [self.cfg.size[0], self.cfg.size[1], 0],
            [0, self.cfg.size[1], 0],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.uint32)

        return trimesh.Trimesh(vertices=vertices, faces=faces)

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
