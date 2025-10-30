"""Data container for height scanner sensor."""

from __future__ import annotations

import torch


class HeightScannerData:
    """Data container for height scanner measurements.
    
    This class holds the measurement data from the height scanner sensor,
    including scan points and height measurements.
    """

    # Scan points in world frame
    scan_points_w: torch.Tensor = None
    """Scan point positions in world frame. Shape: (num_envs, num_points, 3)"""

    # Height measurements
    heights: torch.Tensor = None
    """Height measurements at scan points. Shape: (num_envs, num_points)"""

    # Height values relative to sensor
    heights_relative: torch.Tensor = None
    """Heights relative to sensor position. Shape: (num_envs, num_points)"""
