"""Base configuration for simulation contexts."""

from __future__ import annotations

from typing import Tuple

from cross_gym.utils.configclass import configclass


@configclass
class SimulationContextCfg:
    """Base configuration for simulation contexts.
    
    Each simulator has its own config class that inherits from this.
    """

    # Common parameters across all simulators
    device: str = "cuda:0"
    """Device to run simulation on."""

    dt: float = 0.01
    """Physics timestep in seconds."""

    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    """Gravity vector."""

    headless: bool = True
    """Whether to run without GUI."""

    render_interval: int = 1
    """Render every N physics steps."""
