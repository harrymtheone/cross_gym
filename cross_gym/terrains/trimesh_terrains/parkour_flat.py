from __future__ import annotations

import torch
import trimesh

from cross_gym.terrains import TerrainTypeID, TerrainCommandType, SubTerrain, SubTerrainBaseCfg
from cross_gym.terrains.utils import create_rectangle
from cross_gym.utils import configclass
from cross_gym.utils.math import torch_rand_float


class ParkourFlat(SubTerrain):
    class_type = TerrainTypeID.parkour_flat

    command_type = TerrainCommandType.Goal

    cfg: ParkourFlatCfg

    def build_trimesh(self, difficulty: float) -> trimesh.Trimesh:
        terrain_x, terrain_y = self.cfg.size
        return create_rectangle(size=(terrain_x, terrain_y))

    def build_origins(self) -> tuple[float, float, float]:
        terrain_x, terrain_y = self.cfg.size
        platform_len = self.cfg.platform_len
        return platform_len / 2, terrain_y / 2, 0.0

    def build_goals(self) -> callable:
        def generator(num_envs: int, device: torch.device):
            terrain_x, terrain_y = self.cfg.size
            platform_len = self.cfg.platform_len
            x_range = self.cfg.x_range
            y_range = self.cfg.y_range

            mid_y = terrain_y / 2
            cur_x = platform_len
            goals = torch.zeros(num_envs, 8, 3, device=device)  # 8 goals with x, y, z coordinates
            goals[:, 0, 0] = platform_len
            goals[:, 0, 1] = mid_y

            for i in range(6):
                rand_x = torch_rand_float(x_range[0], x_range[1], (num_envs, 1), device).squeeze(1)
                rand_y = torch_rand_float(y_range[0], y_range[1], (num_envs, 1), device).squeeze(1)
                cur_x += rand_x
                goals[:, i + 1, 0] = cur_x
                goals[:, i + 1, 1] = mid_y + rand_y

            goals[:, -1, 0] = terrain_x - platform_len / 2
            goals[:, -1, 1] = mid_y
            return goals

        return generator


@configclass
class ParkourFlatCfg(SubTerrainBaseCfg):
    class_type = ParkourFlat

    platform_len: float = 2.0
    x_range: tuple[float, float] = (1.5, 2.4)
    y_range: tuple[float, float] = (-1.0, 1.0)
