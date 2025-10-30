"""IsaacGym-specific scene configuration."""

from __future__ import annotations

from cross_core.base import InteractiveSceneCfg
from cross_core.utils import configclass
from . import IsaacGymInteractiveScene


@configclass
class PhysXCfg:
    """PhysX-specific settings."""
    solver_type: int = 1
    num_position_iterations: int = 4
    num_velocity_iterations: int = 1
    num_threads: int = 4
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    bounce_threshold_velocity: float = 0.2
    max_depenetration_velocity: float = 1000.0
    use_gpu: bool = True
    num_subscenes: int = 4
    friction_offset_threshold: float = 0.04
    friction_correlation_distance: float = 0.025


@configclass
class SimCfg:
    """Simulation parameters for IsaacGym."""
    dt: float = 0.005
    substeps: int = 1
    gravity: tuple = (0.0, 0.0, -9.81)
    up_axis: str = "z"
    use_gpu_pipeline: bool = True

    headless: bool = False
    render_interval: int = 2

    physx: PhysXCfg = PhysXCfg()


@configclass
class IsaacGymSceneCfg(InteractiveSceneCfg):
    """IsaacGym-specific scene configuration.
    
    Contains both simulation and scene parameters since scene owns everything.
    
    Usage:
        scene = scene_cfg.class_type(scene_cfg, device)
    """
    class_type = IsaacGymInteractiveScene

    num_envs: int = 1024
    env_spacing: float = 2.0

    sim: SimCfg = SimCfg()

    # Additional attributes for terrain, articulations, sensors
    # are added dynamically as needed:
    # terrain: TerrainGeneratorCfg | None = None
    # robot: ArticulationCfg = ...
    # height_scanner: HeightScannerCfg = ...
