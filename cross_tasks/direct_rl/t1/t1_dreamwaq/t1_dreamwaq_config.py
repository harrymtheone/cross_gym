"""Configuration for T1 DreamWAQ task."""

from __future__ import annotations

from dataclasses import MISSING

from cross_gym.utils import configclass
from cross_gym_tasks import TaskCfg
from cross_gym_tasks.direct_rl.base import HumanoidEnvCfg, ParkourEnvCfg
from cross_gym.sim.isaacgym import IsaacGymCfg
from cross_gym.scene import InteractiveSceneCfg
from cross_gym.assets import ArticulationCfg
from cross_gym.terrains import TerrainGeneratorCfg
from . import T1DreamWaqEnv


# ============================================================================
# Environment Configuration
# ============================================================================

@configclass
class T1DreamWaqEnvCfg(HumanoidEnvCfg):
    """Environment configuration for T1 DreamWAQ."""

    class_type: type = T1DreamWaqEnv

    # ========== Simulation ==========
    sim: IsaacGymCfg = IsaacGymCfg(
        dt=0.005,
        substeps=1,
        device='cuda:0',
        headless=False
    )

    # ========== Scene ==========
    # NOTE: Scene robot and terrain configs should be provided when instantiating the task
    # For now, we provide a minimal placeholder structure
    scene: InteractiveSceneCfg = MISSING  # TODO: Provide scene with robot + terrain

    # ========== Control ==========
    decimation: int = 4  # Simulation steps per environment step

    # ========== Actions ==========
    num_actions: int = 13  # T1 has 13 actuated DOFs

    # ========== Episode ==========
    episode_length_s: float = 30.0  # 30 seconds per episode

    # ========== Gait ==========
    @configclass
    class GaitCfg(HumanoidEnvCfg.GaitCfg):
        cycle_time: float = 0.7
        sw_switch: bool = True  # Reset phase when stationary
        phase_offset_l: float = 0.0
        phase_offset_r: float = 0.5
        air_ratio: float = 0.5
        delta_t: float = 0.02

    gait: GaitCfg = GaitCfg()

    # ========== Commands ==========
    @configclass
    class CommandsCfg(ParkourEnvCfg.CommandsCfg):
        lin_vel_clip: float = 0.2
        ang_vel_clip: float = 0.2

        # Flat terrain commands
        @configclass
        class FlatRangesCfg:
            lin_vel_x: tuple[float, float] = (-0.8, 1.2)
            lin_vel_y: tuple[float, float] = (-0.8, 0.8)
            ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

        flat_ranges: FlatRangesCfg = FlatRangesCfg()

        # Stair terrain commands
        @configclass
        class StairRangesCfg:
            lin_vel_x: tuple[float, float] = (-0.5, 0.8)
            lin_vel_y: tuple[float, float] = (-0.5, 0.5)
            heading: tuple[float, float] = (-1.5, 1.5)
            ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

        stair_ranges: StairRangesCfg = StairRangesCfg()

        # Parkour terrain commands
        @configclass
        class ParkourRangesCfg:
            lin_vel_x: tuple[float, float] = (0.3, 0.8)
            ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

        parkour_ranges: ParkourRangesCfg = ParkourRangesCfg()

    commands: CommandsCfg = CommandsCfg()

    # ========== Control (PD Gains for T1) ==========
    @configclass
    class T1ControlCfg(HumanoidEnvCfg.ControlCfg):
        action_scale: float = 0.2
        clip_actions: float = 100.0

        # T1-specific PD gains
        stiffness: dict = {
            'Head': 30,
            'Shoulder_Pitch': 300, 'Shoulder_Roll': 200, 'Elbow_Pitch': 200, 'Elbow_Yaw': 100,
            'Waist': 100,
            'Hip_Pitch': 55, 'Hip_Roll': 55, 'Hip_Yaw': 30,
            'Knee_Pitch': 100, 'Ankle_Pitch': 30, 'Ankle_Roll': 30,
        }

        damping: dict = {
            'Head': 1,
            'Shoulder_Pitch': 3, 'Shoulder_Roll': 3, 'Elbow_Pitch': 3, 'Elbow_Yaw': 3,
            'Waist': 3,
            'Hip_Pitch': 3, 'Hip_Roll': 3, 'Hip_Yaw': 4,
            'Knee_Pitch': 5, 'Ankle_Pitch': 0.3, 'Ankle_Roll': 0.3,
        }

    control: T1ControlCfg = T1ControlCfg()

    # ========== Domain Randomization ==========
    @configclass
    class T1DomainRandCfg(HumanoidEnvCfg.DomainRandCfg):
        # Reset state randomization
        randomize_start_pos_xy: bool = True
        randomize_start_pos_z: bool = False
        randomize_start_yaw: bool = True
        randomize_start_pitch: bool = True
        randomize_start_lin_vel_xy: bool = True

        randomize_start_dof_pos: bool = False
        randomize_start_dof_vel: bool = False

        # Torque randomization
        randomize_motor_offset: bool = True
        randomize_gains: bool = True
        randomize_torque: bool = True
        randomize_friction: bool = True

    domain_rand: T1DomainRandCfg = T1DomainRandCfg()

    # ========== Rewards ==========
    @configclass
    class T1RewardsCfg(HumanoidEnvCfg.RewardsCfg):
        # Reward targets
        base_height_target: float = 0.64
        feet_height_target: float = 0.04
        tracking_sigma: float = 5.0

        # Contact settings
        contact_ema_alpha: float = 0.99
        use_contact_averaging: bool = True
        min_feet_dist: float = 0.18
        max_feet_dist: float = 0.50
        max_contact_force: float = 300.0

        # Curriculum
        only_positive_rewards: bool = False
        only_positive_rewards_until_epoch: int = 100

        # Reward scales
        scales: dict = {
            # Gait
            'joint_pos': 2.0,
            'feet_contact_number': 1.2,
            'feet_clearance': 1.0,
            'feet_distance': 0.2,
            'knee_distance': 0.2,
            'feet_rotation': 0.5,

            # Contact
            'feet_slip': -0.3,
            'feet_contact_forces': -0.001,

            # Velocity tracking
            'tracking_lin_vel': 2.5,
            'tracking_ang_vel': 1.5,
            'vel_mismatch_exp': 0.5,

            # Base pose
            'default_joint_pos': 1.5,
            'orientation': 1.0,
            'base_height': 0.2,
            'base_acc': 0.2,

            # Energy
            'action_smoothness': -3e-4,
            'dof_vel_smoothness': -1e-3,
            'torques': -1e-5,
            'dof_vel': -5e-4,
            'dof_acc': -1e-7,
            'collision': -1.0,
        }

    rewards: T1RewardsCfg = T1RewardsCfg()

    # ========== Asset (T1 Robot) ==========
    @configclass
    class T1AssetCfg(HumanoidEnvCfg.AssetCfg):
        foot_name: str = ".*Ankle.*"  # Regex for T1 foot bodies
        knee_name: str = ".*Knee.*"  # Regex for T1 knee bodies

    asset: T1AssetCfg = T1AssetCfg()


# ============================================================================
# Task Configuration (Env + Algorithm + Runner)
# ============================================================================

@configclass
class T1DreamWaqCfg(TaskCfg):
    """Complete task configuration for T1 DreamWAQ training."""

    # Environment
    env: T1DreamWaqEnvCfg = T1DreamWaqEnvCfg()

    # Algorithm (PPO) - TODO: Fill with actual PPO config when cross_rl is ready
    algorithm = MISSING  # from cross_rl.algorithms.ppo import PPOCfg

    # Runner - TODO: Fill with actual runner config when cross_rl is ready
    runner = MISSING  # from cross_rl.runners import OnPolicyRunnerCfg


__all__ = ["T1DreamWaqEnvCfg", "T1DreamWaqCfg"]
