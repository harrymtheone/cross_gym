"""Base sensor data container."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SensorBaseData:
    """Base data container for all sensors.
    
    This provides common sensor data that all sensors have:
    - Body attachment index
    - Sensor pose in world frame (position and orientation)
    - Offset transforms (nominal and randomized)
    
    Subclasses should inherit from this and add sensor-specific measurements.
    
    Example:
        >>> @dataclass
        >>> class ImuData(SensorBaseData):
        >>>     lin_acc_b: torch.Tensor = None  # Linear acceleration in body frame
        >>>     ang_vel_b: torch.Tensor = None  # Angular velocity in body frame
    """

    # Body attachment
    body_idx: int = None
    """Index of the body this sensor is attached to."""

    # Nominal offsets (from config)
    offset_pos: torch.Tensor = None
    """Nominal position offset from body frame. Shape: (num_envs, 3)"""
    
    offset_quat: torch.Tensor = None
    """Nominal rotation offset from body frame (w, x, y, z). Shape: (num_envs, 4)"""

    # Actual offsets (with randomization)
    offset_pos_sim: torch.Tensor = None
    """Actual position offset used in simulation (nominal + random). Shape: (num_envs, 3)"""
    
    offset_quat_sim: torch.Tensor = None
    """Actual rotation offset used in simulation (nominal + random). Shape: (num_envs, 4)"""

    # Sensor pose in world frame
    pos_w: torch.Tensor = None
    """Sensor position in world frame. Shape: (num_envs, 3)"""

    quat_w: torch.Tensor = None
    """Sensor orientation in world frame (w, x, y, z). Shape: (num_envs, 4)"""
