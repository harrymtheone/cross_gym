"""Configuration for ray caster sensor."""

from __future__ import annotations

from dataclasses import MISSING

from cross_gym.sensors import SensorBaseCfg
from cross_gym.utils import configclass
from .patterns import RayPatternCfg


@configclass
class RayCasterCfg(SensorBaseCfg):
    """Configuration for ray caster sensor.
    
    Ray caster performs geometric raycasting to measure distances
    to obstacles/terrain in the environment.
    """

    from .ray_caster import RayCaster
    class_type: type = RayCaster

    # ========== Ray Pattern ==========
    pattern: RayPatternCfg = MISSING
    """Ray pattern configuration (grid, lidar, circle, etc.)."""

    # ========== Range ==========
    max_distance: float = 10.0
    """Maximum ray distance in meters."""

    min_distance: float = 0.0
    """Minimum ray distance in meters (for filtering close hits)."""

    # ========== Mesh Configuration ==========
    mesh_prim_paths: list[str] | None = None
    """List of mesh prim paths to raycast against.
    
    If None, raycasts against all meshes in scene.
    Example: ["/World/ground", "/World/obstacles"]
    """

    # ========== Output Options ==========
    return_normals: bool = False
    """Whether to compute and return surface normals at hit points."""

    return_hit_points: bool = True
    """Whether to compute and return hit point positions."""


__all__ = ["RayCasterCfg"]

