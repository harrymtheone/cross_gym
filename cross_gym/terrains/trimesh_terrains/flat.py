from __future__ import annotations

import trimesh

from cross_gym.terrains import SubTerrain, SubTerrainBaseCfg, TerrainTypeID, TerrainCommandType
from cross_gym.terrains.utils import create_rectangle
from cross_gym.utils import configclass


class FlatTerrain(SubTerrain):
    """Flat terrain (easiest difficulty)."""

    type_id = TerrainTypeID.flat
    command_type = TerrainCommandType.Omni

    cfg: FlatCfg

    def build_trimesh(self, difficulty: float) -> trimesh.Trimesh:
        terrain_x, terrain_y = self.cfg.size
        return create_rectangle(size=(terrain_x, terrain_y))

    def build_origins(self) -> tuple[float, float, float]:
        terrain_x, terrain_y = self.cfg.size
        return terrain_x / 2, terrain_y / 2, 0.0

    def build_goals(self, **kwargs) -> None:
        return None


@configclass
class FlatCfg(SubTerrainBaseCfg):
    class_type = FlatTerrain
