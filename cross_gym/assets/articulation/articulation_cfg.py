"""Configuration for articulation assets."""

from __future__ import annotations

from dataclasses import MISSING

from cross_gym.assets.asset_base import AssetBaseCfg
from cross_gym.utils.configclass import configclass
from . import Articulation


@configclass
class ArticulationCfg(AssetBaseCfg):
    """Configuration for articulated assets (robots).
    
    An articulation is a collection of rigid bodies (links) connected by joints.
    This is typically used for robots.
    """

    class_type: type = Articulation

    # Asset file
    file: str = MISSING  # Path to URDF/USD file

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
