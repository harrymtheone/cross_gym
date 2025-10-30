"""Ray casting pattern implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .patterns_cfg import RayPatternCfg


class RayPattern(ABC):
    """Base class for ray casting patterns.
    
    Generates ray directions in sensor's local frame.
    """

    def __init__(self, cfg: RayPatternCfg):
        """Initialize ray pattern.
        
        Args:
            cfg: Pattern configuration
        """
        self.cfg = cfg
        self._directions: torch.Tensor | None = None
    
    @property
    def num_rays(self) -> int:
        """Total number of rays in pattern."""
        return self._directions.shape[0] if self._directions is not None else 0
    
    @abstractmethod
    def generate(self, device: torch.device) -> torch.Tensor:
        """Generate ray directions.
        
        Args:
            device: Device to create tensors on
            
        Returns:
            Ray directions in sensor frame. Shape: (num_rays, 3)
            Directions are unit vectors pointing in ray direction.
        """
        pass


class GridPattern(RayPattern):
    """Grid pattern for regular sampling of field of view.
    
    Creates rays in a rectangular grid pattern.
    """

    def generate(self, device: torch.device) -> torch.Tensor:
        """Generate grid of ray directions.
        
        Args:
            device: Device to create tensors on
            
        Returns:
            Ray directions. Shape: (height * width, 3)
        """
        height, width = self.cfg.resolution
        vfov, hfov = self.cfg.fov
        
        # Convert FOV to radians
        vfov_rad = np.deg2rad(vfov)
        hfov_rad = np.deg2rad(hfov)
        
        # Create grid of angles
        # Vertical angles (pitch): -vfov/2 to +vfov/2
        v_angles = np.linspace(-vfov_rad / 2, vfov_rad / 2, height)
        # Horizontal angles (yaw): -hfov/2 to +hfov/2
        h_angles = np.linspace(-hfov_rad / 2, hfov_rad / 2, width)
        
        # Create meshgrid
        v_grid, h_grid = np.meshgrid(v_angles, h_angles, indexing='ij')
        
        # Flatten grids
        v_flat = v_grid.flatten()
        h_flat = h_grid.flatten()
        
        # Convert spherical to Cartesian (sensor frame: +X forward, +Y left, +Z up)
        # x = cos(pitch) * cos(yaw)
        # y = cos(pitch) * sin(yaw)
        # z = sin(pitch)
        x = np.cos(v_flat) * np.cos(h_flat)
        y = np.cos(v_flat) * np.sin(h_flat)
        z = np.sin(v_flat)
        
        # Stack and convert to tensor
        directions = np.stack([x, y, z], axis=-1)
        self._directions = torch.tensor(directions, dtype=torch.float32, device=device)
        
        return self._directions


class LidarPattern(RayPattern):
    """LiDAR-style pattern with horizontal scan lines.
    
    Creates multiple horizontal rings at different vertical angles.
    """

    def generate(self, device: torch.device) -> torch.Tensor:
        """Generate LiDAR-style ray directions.
        
        Args:
            device: Device to create tensors on
            
        Returns:
            Ray directions. Shape: (num_rays_vertical * num_rays_horizontal, 3)
        """
        num_h = self.cfg.num_rays_horizontal
        num_v = self.cfg.num_rays_vertical
        v_min, v_max = self.cfg.vertical_fov
        h_min, h_max = self.cfg.horizontal_fov
        
        # Convert to radians
        v_min_rad = np.deg2rad(v_min)
        v_max_rad = np.deg2rad(v_max)
        h_min_rad = np.deg2rad(h_min)
        h_max_rad = np.deg2rad(h_max)
        
        # Create vertical angles (elevation)
        v_angles = np.linspace(v_min_rad, v_max_rad, num_v)
        
        # Create horizontal angles (azimuth)
        h_angles = np.linspace(h_min_rad, h_max_rad, num_h)
        
        # Create all combinations
        directions = []
        for v_angle in v_angles:
            for h_angle in h_angles:
                # Spherical to Cartesian
                x = np.cos(v_angle) * np.cos(h_angle)
                y = np.cos(v_angle) * np.sin(h_angle)
                z = np.sin(v_angle)
                directions.append([x, y, z])
        
        directions = np.array(directions, dtype=np.float32)
        self._directions = torch.tensor(directions, dtype=torch.float32, device=device)
        
        return self._directions


class CirclePattern(RayPattern):
    """Single horizontal circle pattern (2D lidar).
    
    Creates rays in a horizontal plane.
    """

    def generate(self, device: torch.device) -> torch.Tensor:
        """Generate circular ray pattern.
        
        Args:
            device: Device to create tensors on
            
        Returns:
            Ray directions. Shape: (num_rays, 3)
        """
        num_rays = self.cfg.num_rays
        min_angle, max_angle = self.cfg.fov
        
        # Convert to radians
        min_rad = np.deg2rad(min_angle)
        max_rad = np.deg2rad(max_angle)
        
        # Create angles
        angles = np.linspace(min_rad, max_rad, num_rays, endpoint=False)
        
        # Convert to Cartesian (z=0 for horizontal plane)
        x = np.cos(angles)
        y = np.sin(angles)
        z = np.zeros_like(x)
        
        directions = np.stack([x, y, z], axis=-1)
        self._directions = torch.tensor(directions, dtype=torch.float32, device=device)
        
        return self._directions


__all__ = [
    "RayPattern",
    "GridPattern",
    "LidarPattern",
    "CirclePattern",
]

