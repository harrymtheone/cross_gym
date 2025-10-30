from __future__ import annotations

import numpy as np
import trimesh

from cross_gym.terrains import TerrainTypeID, TerrainCommandType, SubTerrain, SubTerrainBaseCfg
from cross_gym.terrains.utils import create_rectangle
from cross_core.utils import configclass


class ParkourFlat(SubTerrain):
    type_id = TerrainTypeID.parkour_flat
    command_type = TerrainCommandType.Goal

    cfg: ParkourFlatCfg

    def build_trimesh(self, difficulty: float) -> trimesh.Trimesh:
        terrain_x, terrain_y = self.cfg.size
        return create_rectangle(size=(terrain_x, terrain_y))

    def build_origins(self) -> tuple[float, float, float]:
        terrain_x, terrain_y = self.cfg.size
        platform_len = self.cfg.platform_len
        return platform_len / 2, terrain_y / 2, 0.0

    def build_goals(self, num_envs: int = 1, **kwargs) -> np.ndarray:
        """Build randomized waypoint goals for parkour.
        
        Generates 8 waypoints per environment with random x/y offsets.
        
        Args:
            num_envs: Number of environments (for per-env randomization)
            **kwargs: Additional parameters (unused)
            
        Returns:
            Goals array (num_envs, 8, 3) - different goals per environment
        """
        terrain_x, terrain_y = self.cfg.size
        platform_len = self.cfg.platform_len
        x_range = self.cfg.x_range
        y_range = self.cfg.y_range

        mid_y = terrain_y / 2
        goals = np.zeros((num_envs, 8, 3))  # 8 goals with x, y, z coordinates

        # First goal at start platform
        goals[:, 0, 0] = platform_len
        goals[:, 0, 1] = mid_y

        # Intermediate waypoints with random offsets
        cur_x = np.full(num_envs, platform_len)
        for i in range(6):
            rand_x = np.random.uniform(x_range[0], x_range[1], num_envs)
            rand_y = np.random.uniform(y_range[0], y_range[1], num_envs)
            cur_x += rand_x
            goals[:, i + 1, 0] = cur_x
            goals[:, i + 1, 1] = mid_y + rand_y

        # Last goal at end platform
        goals[:, -1, 0] = terrain_x - platform_len / 2
        goals[:, -1, 1] = mid_y

        return goals


@configclass
class ParkourFlatCfg(SubTerrainBaseCfg):
    class_type = ParkourFlat

    platform_len: float = 2.0
    x_range: tuple[float, float] = (1.5, 2.4)
    y_range: tuple[float, float] = (-1.0, 1.0)
