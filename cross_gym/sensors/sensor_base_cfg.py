"""Base configuration for sensors."""

from __future__ import annotations

from dataclasses import MISSING

from cross_gym.utils import configclass


@configclass
class SensorBaseCfg:
    """Base configuration for all sensors.
    
    Defines common parameters for sensor attachment, update rate,
    history buffering, randomization, and visualization.
    """

    class_type: type = MISSING
    """Sensor class to instantiate (must inherit from SensorBase)."""

    # ========== Attachment ==========
    body_name: str = MISSING
    """Name of the body/link to attach sensor to."""

    # ========== Transform ==========
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset from body frame (x, y, z) in meters."""

    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Rotation offset (roll, pitch, yaw) in degrees."""

    # ========== Update Rate ==========
    update_period: float = 0.0
    """Update period in seconds. If 0.0, updates every simulation step."""

    # ========== History Buffer ==========
    history_length: int = 0
    """Number of past measurements to store. 0 means only current data."""

    # ========== Randomization ==========
    offset_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    """Randomize offset per environment. Format: ((x_min, x_max), (y_min, y_max), (z_min, z_max))."""

    rotation_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    """Randomize rotation per environment. Format: ((r_min, r_max), (p_min, p_max), (y_min, y_max))."""

    # ========== Delay & Noise ==========
    delay_range: tuple[float, float] | None = None
    """Measurement delay range (min, max) in seconds. None for no delay."""

    # ========== Visualization ==========
    debug_vis: bool = False
    """Enable debug visualization."""


__all__ = ["SensorBaseCfg"]

