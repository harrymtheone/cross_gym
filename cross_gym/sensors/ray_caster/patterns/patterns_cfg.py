"""Ray casting pattern configurations."""

from __future__ import annotations

from dataclasses import MISSING

from cross_core.utils import configclass


@configclass
class RayPatternCfg:
    """Base configuration for ray casting patterns."""
    
    class_type: type = MISSING
    """Pattern class to instantiate."""


@configclass
class GridPatternCfg(RayPatternCfg):
    """Grid pattern for ray casting.
    
    Creates a regular grid of rays in the sensor's field of view.
    """
    
    from .patterns import GridPattern
    class_type: type = GridPattern
    
    resolution: tuple[int, int] = (10, 10)
    """Grid resolution (height, width) in number of rays."""
    
    fov: tuple[float, float] = (90.0, 90.0)
    """Field of view (vertical, horizontal) in degrees."""


@configclass
class LidarPatternCfg(RayPatternCfg):
    """LiDAR-style pattern for ray casting.
    
    Creates horizontal scan lines at different vertical angles.
    """
    
    from .patterns import LidarPattern
    class_type: type = LidarPattern
    
    num_rays_horizontal: int = 360
    """Number of rays in horizontal direction."""
    
    num_rays_vertical: int = 16
    """Number of vertical scan lines."""
    
    vertical_fov: tuple[float, float] = (-15.0, 15.0)
    """Vertical field of view (min_angle, max_angle) in degrees."""
    
    horizontal_fov: tuple[float, float] = (0.0, 360.0)
    """Horizontal field of view (min_angle, max_angle) in degrees."""


@configclass
class CirclePatternCfg(RayPatternCfg):
    """Single horizontal circle pattern for ray casting.
    
    Creates rays in a horizontal plane (like 2D lidar).
    """
    
    from .patterns import CirclePattern
    class_type: type = CirclePattern
    
    num_rays: int = 180
    """Number of rays in the circle."""
    
    fov: tuple[float, float] = (0.0, 360.0)
    """Field of view (min_angle, max_angle) in degrees."""


__all__ = [
    "RayPatternCfg",
    "GridPatternCfg",
    "LidarPatternCfg",
    "CirclePatternCfg",
]

