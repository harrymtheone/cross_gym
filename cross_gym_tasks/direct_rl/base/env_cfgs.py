from __future__ import annotations

from dataclasses import MISSING

from cross_gym import DirectRLEnvCfg, ManagerTermCfg
from cross_gym.utils import configclass
from . import LocomotionEnv, ParkourEnv, HumanoidEnv


@configclass
class LocomotionEnvCfg(DirectRLEnvCfg):
    """Configuration for locomotion environment."""

    class_type: type = LocomotionEnv

    # Number of actions = number of DOF
    num_actions: int = MISSING

    # Control - PD controller settings
    @configclass
    class LocomotionControlCfg(DirectRLEnvCfg.ControlCfg):
        action_scale: float = 0.25
        clip_actions: float = 100.0

        # PD gains (adjust for your robot)
        stiffness: dict = {
            ".*": 20.0,  # Match all joints
        }
        damping: dict = {
            ".*": 0.5,
        }

    control: LocomotionControlCfg = LocomotionControlCfg()

    # Rewards
    rewards: dict = MISSING


@configclass
class ParkourEnvCfg(LocomotionEnvCfg):
    """Configuration for parkour environment."""

    class_type: type = ParkourEnv

    # ========== Terrain Curriculum ==========
    terrain_curriculum: bool = True
    """Enable terrain curriculum learning."""

    max_init_terrain_level: int = 0
    """Maximum initial terrain level (row). Increases with curriculum."""

    terrain_size: tuple[float, float] = (8.0, 8.0)
    """Size of each terrain patch (for curriculum distance thresholds)."""

    # ========== Goal Navigation ==========
    next_goal_threshold: float = 0.5
    """Distance threshold to consider goal reached (meters)."""

    reach_goal_delay: float = 0.5
    """Time delay before moving to next goal (seconds)."""

    # ========== Commands ==========
    @configclass
    class CommandsCfg:
        """Command configuration for different terrain types."""
        lin_vel_clip: float = 0.1
        """Minimum linear velocity to be considered non-zero."""

        ang_vel_clip: float = 0.1
        """Minimum angular velocity to be considered non-zero."""

        # Flat terrain commands (omnidirectional)
        @configclass
        class FlatRangesCfg:
            lin_vel_x: tuple[float, float] = (-1.0, 1.5)
            lin_vel_y: tuple[float, float] = (-0.5, 0.5)
            ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

        flat_ranges: FlatRangesCfg = FlatRangesCfg()

        # Stair terrain commands (heading-based)
        @configclass
        class StairRangesCfg:
            lin_vel_x: tuple[float, float] = (0.5, 1.5)
            lin_vel_y: tuple[float, float] = (-0.3, 0.3)
            heading: tuple[float, float] = (-3.14, 3.14)
            ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

        stair_ranges: StairRangesCfg = StairRangesCfg()

        # Parkour terrain commands (goal-guided)
        @configclass
        class ParkourRangesCfg:
            lin_vel_x: tuple[float, float] = (0.5, 1.5)
            ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

        parkour_ranges: ParkourRangesCfg = ParkourRangesCfg()

    commands: CommandsCfg = CommandsCfg()


@configclass
class HumanoidEnvCfg(ParkourEnvCfg):
    """Configuration for humanoid environment."""

    class_type: type = HumanoidEnv

    # ========== Humanoid Asset Configuration ==========
    @configclass
    class HumanoidAssetCfg(ParkourEnvCfg.AssetCfg):
        """Humanoid-specific asset configuration."""

        foot_name: str = ".*foot"
        """Regex pattern to match foot bodies."""

        knee_name: str = ".*knee"
        """Regex pattern to match knee bodies."""

    asset: HumanoidAssetCfg = HumanoidAssetCfg()

    # ========== Contact Configuration ==========
    contact_force_threshold: float = 2.0
    """Force threshold for contact detection (N)."""

    # ========== Gait Configuration ==========
    @configclass
    class GaitCfg:
        """Gait phase configuration for bipedal locomotion."""

        cycle_time: float = 0.64
        """Full gait cycle time (seconds)."""

        air_ratio: float = 0.5
        """Fraction of cycle in swing phase [0, 1]."""

        delta_t: float = 0.05
        """Transition period for smooth stance/swing switching."""

        phase_offset_l: float = 0.0
        """Phase offset for left leg [0, 1]."""

        phase_offset_r: float = 0.5
        """Phase offset for right leg [0, 1] (typically 0.5 for alternating gait)."""

    gait: GaitCfg = GaitCfg()

    # ========== Terrain Configuration (Humanoid-specific) ==========
    @configclass
    class HumanoidTerrainCfg(ParkourEnvCfg.TerrainCfg):
        """Humanoid terrain configuration."""

        foothold_pts: tuple[tuple[float, float, int], tuple[float, float, int], float] = (
            (-0.05, 0.05, 3),  # x: -5cm to +5cm, 3 points
            (-0.05, 0.05, 3),  # y: -5cm to +5cm, 3 points
            -0.02,  # z shift: -2cm below foot center
        )
        """Foothold detection grid: ((x_min, x_max, x_res), (y_min, y_max, y_res), z_shift)."""

        foothold_contact_thresh: float = 0.01
        """Height threshold for foothold contact detection (m)."""

    terrain: HumanoidTerrainCfg = HumanoidTerrainCfg()

    # ========== Reward Configuration ==========
    @configclass
    class HumanoidRewardsCfg(ParkourEnvCfg.RewardsCfg):
        """Humanoid-specific rewards."""

        # Gait rewards
        feet_contact_number = ManagerTermCfg(
            func=HumanoidEnv._reward_feet_contact_number,
            weight=1.0
        )

        swing_phase = ManagerTermCfg(
            func=HumanoidEnv._reward_swing_phase,
            weight=0.5
        )

        support_phase = ManagerTermCfg(
            func=HumanoidEnv._reward_support_phase,
            weight=0.5
        )

        # Feet behavior
        feet_clearance = ManagerTermCfg(
            func=HumanoidEnv._reward_feet_clearance,
            weight=0.3
        )

        feet_air_time = ManagerTermCfg(
            func=HumanoidEnv._reward_feet_air_time,
            weight=0.2
        )

        feet_slip = ManagerTermCfg(
            func=HumanoidEnv._reward_feet_slip,
            weight=-0.1
        )

        feet_stumble = ManagerTermCfg(
            func=HumanoidEnv._reward_feet_stumble,
            weight=-0.5
        )

        feet_rotation = ManagerTermCfg(
            func=HumanoidEnv._reward_feet_rotation,
            weight=0.3
        )

        # Feet positioning
        feet_distance = ManagerTermCfg(
            func=HumanoidEnv._reward_feet_distance,
            weight=0.2
        )

        knee_distance = ManagerTermCfg(
            func=HumanoidEnv._reward_knee_distance,
            weight=0.1
        )

        # Terrain interaction
        feet_edge = ManagerTermCfg(
            func=HumanoidEnv._reward_feet_edge,
            weight=-0.5
        )

        foothold = ManagerTermCfg(
            func=HumanoidEnv._reward_foothold,
            weight=-0.3
        )

        # Reward tuning parameters
        feet_height_target: float = 0.05
        """Target feet clearance height during swing (m)."""

        min_feet_dist: float = 0.15
        """Minimum desired distance between feet (m)."""

        max_feet_dist: float = 0.40
        """Maximum desired distance between feet (m)."""

        tracking_sigma: float = 0.25
        """Sigma for exponential tracking rewards."""

        use_contact_averaging: bool = True
        """Whether to use exponential moving average for contact forces."""

        contact_ema_alpha: float = 0.9
        """Alpha for contact force EMA (higher = more smoothing)."""

    rewards: HumanoidRewardsCfg = HumanoidRewardsCfg()
