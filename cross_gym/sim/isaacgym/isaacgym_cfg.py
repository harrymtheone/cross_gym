"""Configuration for IsaacGym simulator."""

from __future__ import annotations

from cross_gym.sim.simulation_context_cfg import SimulationContextCfg
from cross_gym.utils.configclass import configclass
from . import IsaacGymContext


@configclass
class IsaacGymCfg(SimulationContextCfg):
    """Configuration for IsaacGym simulator.
    
    This contains all the parameters specific to IsaacGym.
    """

    @configclass
    class PhysxCfg:
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

    class_type: type = IsaacGymContext

    # IsaacGym-specific physics settings
    physx: PhysxCfg = PhysxCfg()
    """PhysX physics engine settings."""

    # IsaacGym-specific parameters
    substeps: int = 1
    """Number of physics substeps per step."""

    up_axis: str = "z"
    """Up axis ('z' or 'y')."""

    use_gpu_pipeline: bool = True
    """Whether to use GPU pipeline (auto-set based on device)."""
