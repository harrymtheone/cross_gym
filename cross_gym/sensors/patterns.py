"""Scan pattern configurations for sensors (height scanner, ray caster, etc.)."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from cross_gym.utils import configclass


@configclass
class ScanPatternCfg:
    """Base configuration for scan patterns.
    
    Scan patterns generate sampling points in the sensor's local frame.
    Different sensors (ray caster, height scanner) can use the same patterns.
    """

    func: callable = MISSING
    """Pattern generation function."""


def grid_pattern(cfg: ScanPatternCfg, device: torch.device) -> torch.Tensor:
    """Generate a regular grid pattern in XY plane.
    
    Args:
        cfg: Pattern configuration with 'size' and 'resolution' attributes
        device: Device to create tensor on
        
    Returns:
        Points in local frame. Shape: (num_points, 3)
    """
    size = getattr(cfg, 'size', (2.0, 2.0))  # (x_size, y_size) in meters
    resolution = getattr(cfg, 'resolution', (10, 10))  # (x_res, y_res)

    x = torch.linspace(-size[0] / 2, size[0] / 2, resolution[0], device=device)
    y = torch.linspace(-size[1] / 2, size[1] / 2, resolution[1], device=device)

    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid_z = torch.zeros_like(grid_x)

    points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    return points


def circle_pattern(cfg: ScanPatternCfg, device: torch.device) -> torch.Tensor:
    """Generate a circle pattern in XY plane.
    
    Args:
        cfg: Pattern configuration with 'radius' and 'num_points' attributes
        device: Device to create tensor on
        
    Returns:
        Points in local frame. Shape: (num_points, 3)
    """
    radius = getattr(cfg, 'radius', 1.0)
    num_points = getattr(cfg, 'num_points', 36)

    angles = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    z = torch.zeros_like(x)

    points = torch.stack([x, y, z], dim=-1)
    return points


def radial_grid_pattern(cfg: ScanPatternCfg, device: torch.device) -> torch.Tensor:
    """Generate a radial grid pattern (multiple circles at different radii).
    
    Args:
        cfg: Pattern configuration with 'radii' and 'num_points_per_circle' attributes
        device: Device to create tensor on
        
    Returns:
        Points in local frame. Shape: (num_points, 3)
    """
    radii = getattr(cfg, 'radii', [0.5, 1.0, 1.5])
    num_points_per_circle = getattr(cfg, 'num_points_per_circle', 12)

    all_points = []
    for radius in radii:
        angles = torch.linspace(0, 2 * torch.pi, num_points_per_circle + 1, device=device)[:-1]
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        z = torch.zeros_like(x)
        all_points.append(torch.stack([x, y, z], dim=-1))

    points = torch.cat(all_points, dim=0)
    return points


@configclass
class GridPatternCfg(ScanPatternCfg):
    """Regular grid pattern configuration."""

    func: callable = grid_pattern

    size: tuple[float, float] = (2.0, 2.0)
    """Grid size (x, y) in meters."""

    resolution: tuple[int, int] = (10, 10)
    """Grid resolution (x, y) in number of points."""


@configclass
class CirclePatternCfg(ScanPatternCfg):
    """Circle pattern configuration."""

    func: callable = circle_pattern

    radius: float = 1.0
    """Circle radius in meters."""

    num_points: int = 36
    """Number of points around the circle."""


@configclass
class RadialGridPatternCfg(ScanPatternCfg):
    """Radial grid pattern (multiple concentric circles)."""

    func: callable = radial_grid_pattern

    radii: list[float] = None
    """List of radii for concentric circles."""

    num_points_per_circle: int = 12
    """Number of points per circle."""

    def __post_init__(self):
        if self.radii is None:
            self.radii = [0.5, 1.0, 1.5]


__all__ = [
    "ScanPatternCfg",
    "GridPatternCfg",
    "CirclePatternCfg",
    "RadialGridPatternCfg",
    "grid_pattern",
    "circle_pattern",
    "radial_grid_pattern",
]
