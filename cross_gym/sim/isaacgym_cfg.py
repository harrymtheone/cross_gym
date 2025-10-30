"""Configuration for IsaacGym simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cross_core.base import SimulationContextCfg
from cross_core.utils import configclass

if TYPE_CHECKING:
    from .isaacgym_context import IsaacGymContext


@configclass
class PhysXCfg:
    """PhysX-specific settings for IsaacGym."""

    # Solver settings
    solver_type: int = 1  # 0: PGS, 1: TGS
    num_position_iterations: int = 4
    num_velocity_iterations: int = 1
    num_threads: int = 4

    # Contact settings
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    bounce_threshold_velocity: float = 0.2
    max_depenetration_velocity: float = 1000.0

    # GPU settings
    use_gpu: bool = True
    num_subscenes: int = 4

    # Other
    enable_ccd: bool = False
    friction_offset_threshold: float = 0.04
    friction_correlation_distance: float = 0.025


@configclass
class IsaacGymCfg(SimulationContextCfg):
    """Configuration for IsaacGym simulator.
    
    This contains all the parameters specific to IsaacGym.
    
    Usage:
        cfg = IsaacGymCfg(...)
        sim = cfg.class_type(cfg)  # Creates IsaacGymContext instance
    """

    class_type: type[IsaacGymContext] = None  # Set after IsaacGymContext is defined

    # Basic simulation parameters
    dt: float = 0.005
    """Simulation timestep in seconds."""

    gravity: tuple = (0.0, 0.0, -9.81)
    """Gravity vector."""

    # IsaacGym-specific parameters
    substeps: int = 1
    """Number of physics substeps per step."""

    up_axis: str = "z"
    """Up axis ('z' or 'y')."""

    use_gpu_pipeline: bool = True
    """Whether to use GPU pipeline (auto-set based on device)."""

    # Rendering
    headless: bool = False
    """Run without viewer."""

    render_interval: int = 2
    """Render every N steps."""

    # IsaacGym-specific physics settings
    physx: PhysXCfg = PhysXCfg()
    """PhysX physics engine settings."""

