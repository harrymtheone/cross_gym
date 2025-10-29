from __future__ import annotations

from dataclasses import MISSING

from cross_gym import DirectRLEnvCfg
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

    @configclass
    class RewardsCfg:
        """Reward configuration."""

        only_positive_rewards: bool = False
        """Only positive rewards."""
        only_positive_rewards_until_epoch: int = 1000
        """Epoch until only positive rewards."""

        scales: dict[str, float] = MISSING
        """Reward scales. {reward_name: scale}"""

    class_type: type = LocomotionEnv

    num_actions: int = MISSING
    """Number of actions (usually equal to num_dof)."""

    control: ControlCfg = ControlCfg()

    domain_rand: DomainRandCfg = DomainRandCfg()

    rewards: RewardsCfg = RewardsCfg()


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

    @configclass
    class HumanoidAssetCfg:
        """Humanoid-specific asset configuration."""

        foot_name: str = ".*foot"
        """Regex pattern to match foot bodies."""

        knee_name: str = ".*knee"
        """Regex pattern to match knee bodies."""

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

    class_type: type = HumanoidEnv

    asset: HumanoidAssetCfg = HumanoidAssetCfg()

    contact_force_threshold: float = 2.0
    """Force threshold for contact detection (N)."""

    gait: GaitCfg = None

    terrain: HumanoidTerrainCfg = HumanoidTerrainCfg()
