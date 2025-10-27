"""Configuration for ray caster sensor."""

from __future__ import annotations

from dataclasses import MISSING

from cross_gym.sensors import SensorBaseCfg
from cross_gym.utils import configclass
from . import RayCaster
from .patterns import RayPatternCfg


@configclass
class RayCasterCfg(SensorBaseCfg):
    """Configuration for ray caster sensor.
    
    Ray caster performs geometric raycasting to measure distances
    to obstacles/terrain in the environment.
    """

    class_type: type = RayCaster

    # ========== Ray Pattern ==========
    pattern: RayPatternCfg = MISSING
    """Ray pattern configuration (grid, lidar, circle, etc.)."""

    # ========== Range ==========
    min_distance: float = 0.0
    """Minimum ray distance in meters (for filtering close hits)."""

    max_distance: float = 10.0
    """Maximum ray distance in meters."""

    # ========== Mesh Configuration ==========
    mesh_prim_paths: list[str] | None = None
    """List of mesh prim paths to raycast against (deprecated, use mesh_names).
    
    If None, raycasts against all meshes in scene.
    Example: ["/World/ground", "/World/obstacles"]
    
    .. deprecated::
        Use mesh_names instead for registry-based mesh access.
    """
    
    mesh_names: list[str] = ["terrain"]
    """Names of meshes to raycast against in the mesh registry.
    
    These correspond to names registered by terrain or other objects.
    Default is ["terrain"] to raycast against the main terrain mesh.
    Example: ["terrain", "obstacles"]
    """

    # ========== Output Options ==========
    return_hit_points: bool = True
    """Whether to compute and return hit point positions."""


__all__ = ["RayCasterCfg"]
