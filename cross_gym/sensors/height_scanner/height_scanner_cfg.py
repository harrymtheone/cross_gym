"""Configuration for height scanner sensor."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from cross_core.base import SensorBaseCfg
from cross_core.utils import configclass
from . import HeightScanner
from .patterns import ScanPatternCfg


@configclass
class HeightScannerCfg(SensorBaseCfg):
    """Configuration for height scanner sensor.
    
    Height scanner reads height values from a terrain heightmap at scan points
    around the sensor. It transforms scan points from sensor frame to world frame,
    then queries the terrain heightmap.
    """

    class_type: type = HeightScanner

    # Scan pattern
    pattern_cfg: ScanPatternCfg = MISSING
    """Scan pattern configuration defining sampling points."""

    # Alignment mode
    alignment: Literal["base", "yaw", "gravity"] = "yaw"
    """How to align scan points:
    - "base": Full 6DOF sensor orientation (roll, pitch, yaw)
    - "yaw": Only yaw alignment (ignores roll/pitch)  
    - "gravity": Project gravity-aligned (for slope-relative measurements)
    """

    # Interpolation method
    interpolation: Literal["nearest", "bilinear", "minimum"] = "bilinear"
    """Heightmap interpolation method:
    - "nearest": Nearest neighbor (fastest)
    - "bilinear": Bilinear interpolation (smooth)
    - "minimum": Minimum of 4 neighbors (conservative, good for collision)
    """

    # Height representation
    measure_relative_height: bool = True
    """If True, return heights relative to sensor position (sensor.z - height).
    If False, return absolute world heights."""

    # Advanced options
    use_guidance_map: bool = False
    """If True, use terrain guidance heightmap instead of collision heightmap.
    Guidance maps may have cleaned/simplified terrain for planning."""

    clamp_to_terrain_bounds: bool = True
    """If True, clamp scan points to terrain boundaries."""


__all__ = ["HeightScannerCfg"]
