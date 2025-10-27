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
    @configclass
    class OffsetCfg:
        """Position offset randomization ranges per axis."""
        x: tuple[float, float] | None = None
        """X-axis offset range (min, max) in meters. None means no randomization."""
        y: tuple[float, float] | None = None
        """Y-axis offset range (min, max) in meters. None means no randomization."""
        z: tuple[float, float] | None = None
        """Z-axis offset range (min, max) in meters. None means no randomization."""

    offset_range: OffsetCfg = OffsetCfg()
    """Randomize position offset per environment.
    
    Each environment gets a random offset sampled from the specified ranges.
    Leave axis as None to disable randomization for that axis.
    
    Example:
        offset_range = OffsetCfg(
            x=(-0.02, 0.02),  # Random offset ±2cm in x
            y=(-0.01, 0.01),  # Random offset ±1cm in y
            z=None,           # No randomization in z
        )
    """

    @configclass
    class RotationCfg:
        """Rotation randomization ranges per axis."""
        roll: tuple[float, float] | None = None
        """Roll angle range (min, max) in degrees. None means no randomization."""
        pitch: tuple[float, float] | None = None
        """Pitch angle range (min, max) in degrees. None means no randomization."""
        yaw: tuple[float, float] | None = None
        """Yaw angle range (min, max) in degrees. None means no randomization."""

    rotation_range: RotationCfg = RotationCfg()
    """Randomize rotation offset per environment.
    
    Each environment gets a random rotation sampled from the specified ranges.
    Leave axis as None to disable randomization for that axis.
    
    Example:
        rotation_range = RotationCfg(
            roll=(-5.0, 5.0),   # Random roll ±5 degrees
            pitch=(-3.0, 3.0),  # Random pitch ±3 degrees
            yaw=None,           # No randomization in yaw
        )
    """

    # ========== Delay & Noise ==========
    delay_range: tuple[float, float] | None = None
    """Measurement delay range (min, max) in seconds. None for no delay."""

    # ========== Visualization ==========
    debug_vis: bool = False
    """Enable debug visualization."""


__all__ = ["SensorBaseCfg"]
