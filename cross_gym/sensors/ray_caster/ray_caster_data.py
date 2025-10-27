"""Data container for ray caster sensor."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cross_gym.sensors import SensorBaseData


@dataclass
class RayCasterData(SensorBaseData):
    """Data container for ray caster sensor.
    
    Inherits sensor pose (pos_w, quat_w) from SensorBaseData.
    All tensors have shape (num_envs, num_rays) unless otherwise noted.
    """

    # ========== Ray Directions ==========
    ray_directions_w: torch.Tensor = None
    """Ray directions in world frame (unit vectors). Shape: (num_envs, num_rays, 3)."""

    # ========== Measurements ==========
    distances: torch.Tensor = None
    """Ray hit distances. Shape: (num_envs, num_rays).
    
    Distance to first hit along each ray. If no hit, value is max_distance.
    """

    hit_points_w: torch.Tensor = None
    """Hit point positions in world frame. Shape: (num_envs, num_rays, 3).
    
    Position where ray intersected geometry. If no hit, position is at max_distance.
    """

    hit_mask: torch.Tensor = None
    """Hit mask indicating valid hits. Shape: (num_envs, num_rays).
    
    Boolean tensor: True where ray hit geometry, False otherwise.
    """


__all__ = ["RayCasterData"]

