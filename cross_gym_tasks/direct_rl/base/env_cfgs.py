from __future__ import annotations

from dataclasses import MISSING

from cross_gym import DirectRLEnvCfg, ManagerTermCfg
from cross_gym.utils import configclass
from . import LocomotionEnv, ParkourEnv, HumanoidEnv


@configclass
class LocomotionEnvCfg(DirectRLEnvCfg):
    """Configuration for locomotion environment."""

    @configclass
    class ControlCfg:
        """PD controller configuration."""
        action_scale: float = MISSING
        """Scale factor for actions to joint positions."""

        clip_actions: float = 100.0
        """Clip actions to this range."""

        stiffness: dict = MISSING
        """DOF stiffness (kp) for PD controller. {dof_pattern: kp}"""

        damping: dict = MISSING
        """DOF damping (kd) for PD controller. {dof_pattern: kd}"""

    @configclass
    class DomainRandCfg:
        """Domain randomization configuration for robust training."""

        # Root state randomization
        randomize_start_pos_xy: bool = False
        """Randomize initial xy position."""
        randomize_start_pos_xy_range: tuple[float, float] = (-0.5, 0.5)
        """Range for xy position randomization (min, max) in meters."""

        randomize_start_pos_z: bool = False
        """Randomize initial z height."""
        randomize_start_pos_z_range: tuple[float, float] = (0.0, 0.1)
        """Range for z height randomization (min, max) in meters."""

        randomize_start_yaw: bool = False
        """Randomize initial yaw orientation."""
        randomize_start_yaw_range: tuple[float, float] = (-3.14, 3.14)
        """Range for yaw randomization (min, max) in radians."""

        randomize_start_pitch: bool = False
        """Randomize initial pitch orientation."""
        randomize_start_pitch_range: tuple[float, float] = (-0.2, 0.2)
        """Range for pitch randomization (min, max) in radians."""

        randomize_start_lin_vel_xy: bool = False
        """Randomize initial linear velocity (xy components only)."""
        randomize_start_lin_vel_xy_range: tuple[float, float] = (-0.5, 0.5)
        """Range for xy linear velocity randomization (min, max) in m/s."""

        # Joint state randomization
        randomize_start_dof_pos: bool = False
        """Randomize initial DOF positions."""
        randomize_start_dof_pos_range: tuple[float, float] = (-0.1, 0.1)
        """Range for DOF position randomization (min, max) in radians."""

        randomize_start_dof_vel: bool = False
        """Randomize initial DOF velocities."""
        randomize_start_dof_vel_range: tuple[float, float] = (-0.5, 0.5)
        """Range for DOF velocity randomization (min, max) in rad/s."""

        # Torque computation randomization
        randomize_motor_offset: bool = False
        """Randomize motor position offset (simulates calibration error)."""
        motor_offset_range: tuple[float, float] = (-0.02, 0.02)
        """Motor offset range (min, max) in radians."""

        randomize_gains: bool = False
        """Randomize PD gains (simulates model uncertainty)."""
        kp_multiplier_range: tuple[float, float] = (0.8, 1.2)
        """Kp gain multiplier range (min, max)."""
        kd_multiplier_range: tuple[float, float] = (0.5, 1.5)
        """Kd gain multiplier range (min, max)."""

        randomize_torque: bool = False
        """Randomize output torque (simulates actuator variance)."""
        torque_multiplier_range: tuple[float, float] = (0.9, 1.1)
        """Torque multiplier range (min, max)."""

        randomize_friction: bool = False
        """Randomize joint friction (Coulomb + viscous)."""
        friction_coulomb_range: tuple[float, float] = (0.0, 0.5)
        """Coulomb friction range (min, max) - constant friction opposing motion."""
        friction_viscous_range: tuple[float, float] = (0.0, 0.1)
        """Viscous friction range (min, max) - velocity-proportional damping."""

    class_type: type = LocomotionEnv

    num_actions: int = MISSING
    """Number of actions (usually equal to num_dof)."""

    control: ControlCfg = ControlCfg()

    domain_rand: DomainRandCfg = DomainRandCfg()

    rewards: dict = MISSING


@configclass
class ParkourEnvCfg(LocomotionEnvCfg):
    """Configuration for parkour environment."""

    @configclass
    class CommandsCfg:
        @configclass
        class FlatRangesCfg:
            lin_vel_x: tuple[float, float] = (-1.0, 1.5)
            lin_vel_y: tuple[float, float] = (-0.5, 0.5)
            ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

        @configclass
        class StairRangesCfg:
            lin_vel_x: tuple[float, float] = (0.5, 1.5)
            lin_vel_y: tuple[float, float] = (-0.3, 0.3)
            heading: tuple[float, float] = (-3.14, 3.14)
            ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

        @configclass
        class ParkourRangesCfg:
            lin_vel_x: tuple[float, float] = (0.5, 1.5)
            ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

        """Command configuration for different terrain types."""
        lin_vel_clip: float = 0.1
        """Minimum linear velocity to be considered non-zero."""

        ang_vel_clip: float = 0.1
        """Minimum angular velocity to be considered non-zero."""

        flat_ranges: FlatRangesCfg = FlatRangesCfg()
        stair_ranges: StairRangesCfg = StairRangesCfg()
        parkour_ranges: ParkourRangesCfg = ParkourRangesCfg()

    @configclass
    class ParkourCfg:
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

    class_type: type = ParkourEnv

    commands: CommandsCfg = CommandsCfg()

    parkour: ParkourCfg = ParkourCfg()


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
