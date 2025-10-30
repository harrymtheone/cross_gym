"""Configuration for articulation assets."""

from __future__ import annotations

from cross_core.base import ArticulationBaseCfg
from cross_core.utils import configclass
from . import GymArticulation


@configclass
class GymArticulationCfg(ArticulationBaseCfg):
    """Configuration for articulated assets (robots).
    
    An articulation is a collection of rigid bodies (links) connected by joints.
    This is typically used for robots.
    """

    class_type: type = GymArticulation

    # Asset file
    prim_path: str = "/World/envs/env_.*/Asset"
    """Path pattern to articulation in scene."""

    file: str | None = None
    """Path to URDF/USD file."""

    # Articulation-specific initial state
    @configclass
    class InitStateCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Initial position (x, y, z) in world frame."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Initial rotation as quaternion (w, x, y, z) in world frame."""

        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Initial linear velocity (vx, vy, vz) in world frame."""

        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Initial angular velocity (wx, wy, wz) in world frame."""

        joint_pos: dict[str, float] = None
        """Default joint positions as pattern-to-value mapping."""

        joint_vel: dict[str, float] = None
        """Default joint velocities as pattern-to-value mapping."""

    init_state: InitStateCfg = InitStateCfg()

    # Collision group (-1 = global collision, 0+ = group ID)
    collision_group: int = 0

    # Asset properties
    @configclass
    class AssetOptionsCfg:
        """Asset loading options."""
        fix_base_link: bool = False
        collapse_fixed_joints: bool = True
        replace_cylinder_with_capsule: bool = True
        flip_visual_attachments: bool = False
        default_dof_drive_mode: int = 3  # 0=None, 1=Pos, 2=Vel, 3=Effort

        # Physical properties
        density: float = 0.001
        angular_damping: float = 0.0
        linear_damping: float = 0.0
        max_angular_velocity: float = 64.0
        max_linear_velocity: float = 1000.0
        armature: float = 0.0
        thickness: float = 0.01
        disable_gravity: bool = False

        # Soft limits
        use_soft_limits: bool = False
        sim_dof_limit_mul: float = 1.0

    asset_options: AssetOptionsCfg = AssetOptionsCfg()

    # Self collisions
    self_collisions: bool = False

    # Actuator groups
    actuators: dict = {}
    """Actuator models for different joint groups.
    
    Keys are group names, values are ActuatorBaseCfg instances.
    If empty, no actuator models are used (direct torque control).
    
    Example:
        actuators = {
            "legs": IdealPDActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_.*"],
                stiffness=20.0,
                damping=0.5,
            ),
        }
    """
