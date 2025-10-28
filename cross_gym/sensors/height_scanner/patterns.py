"""Scan pattern configurations for sensors (height scanner, ray caster, etc.)."""

from __future__ import annotations

import torch

from cross_gym.utils import configclass


@configclass
class ScanPatternCfg:
    """Base configuration for scan patterns.
    
    Scan patterns generate sampling points in the sensor's local frame.
    Different sensors (ray caster, height scanner) can use the same patterns.
    """

    func: callable
    """Pattern generation function. Must be set by subclass."""


def grid_pattern(cfg: GridPatternCfg, device: torch.device) -> torch.Tensor:
    """Generate a regular grid pattern in XY plane.
    
    Args:
        cfg: Grid pattern configuration
        device: Device to create tensor on
        
    Returns:
        Points in local frame. Shape: (num_points, 3)
    """
    x = torch.linspace(-cfg.size[0] / 2, cfg.size[0] / 2, cfg.resolution[0], device=device)
    y = torch.linspace(-cfg.size[1] / 2, cfg.size[1] / 2, cfg.resolution[1], device=device)

    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid_z = torch.zeros_like(grid_x)

    return torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)


def circle_pattern(cfg: CirclePatternCfg, device: torch.device) -> torch.Tensor:
    """Generate a circle pattern in XY plane.
    
    Args:
        cfg: Circle pattern configuration
        device: Device to create tensor on
        
    Returns:
        Points in local frame. Shape: (num_points, 3)
    """
    angles = torch.linspace(0, 2 * torch.pi, cfg.num_points + 1, device=device)[:-1]
    x = cfg.radius * torch.cos(angles)
    y = cfg.radius * torch.sin(angles)
    z = torch.zeros_like(x)

    return torch.stack([x, y, z], dim=-1)


def radial_grid_pattern(cfg: RadialGridPatternCfg, device: torch.device) -> torch.Tensor:
    """Generate a radial grid pattern (multiple circles at different radii).
    
    Args:
        cfg: Radial grid pattern configuration
        device: Device to create tensor on
        
    Returns:
        Points in local frame. Shape: (num_points, 3)
    """
    all_points = []
    for radius in cfg.radii:
        angles = torch.linspace(0, 2 * torch.pi, cfg.num_points_per_circle + 1, device=device)[:-1]
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        z = torch.zeros_like(x)
        all_points.append(torch.stack([x, y, z], dim=-1))

    return torch.cat(all_points, dim=0)


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
