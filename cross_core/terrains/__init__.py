"""Terrain generation for Cross-Gym.

This module provides terrain generation capabilities:
- TerrainGenerator: Creates procedural terrains with curriculum
- Sub-terrains: Individual terrain types (slopes, stairs, gaps, etc.)
- Height field and trimesh support
"""

from .sub_terrain import TerrainTypeID, TerrainCommandType, SubTerrainBaseCfg, SubTerrain

from .terrain_generator import TerrainGenerator
from .terrain_generator_cfg import TerrainGeneratorCfg

from .trimesh_terrains import *
