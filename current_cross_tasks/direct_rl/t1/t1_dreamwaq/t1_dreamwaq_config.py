"""Configuration for T1 DreamWAQ task."""

from __future__ import annotations

import os

from cross_assets import ASSETS_ROOT_DIR
from cross_gym import terrains
from cross_gym.assets import ArticulationCfg
from cross_gym.scene import InteractiveSceneCfg
from cross_gym.sim.isaacgym import IsaacGymCfg
from cross_gym.utils import configclass
from cross_rl.algorithms.ppo import PPOCfg
from cross_rl.runners import OnPolicyRunnerCfg
from cross_tasks import TaskCfg
from cross_tasks.direct_rl.base import HumanoidEnvCfg, ParkourEnvCfg
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
    @configclass
    class T1SceneCfg(InteractiveSceneCfg):
        num_envs: int = 4096

        # T1 Robot
        robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            file=os.path.join(ASSETS_ROOT_DIR, "robots/T1/T1_legs.urdf"),

            # Initial state
            init_state=ArticulationCfg.InitStateCfg(
                pos=(0.0, 0.0, 0.64),
                rot=(1.0, 0.0, 0.0, 0.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),

                # Joint positions (default standing pose)
                joint_pos={
                    'AAHead_yaw': 0.,
                    # 'Head_pitch': 0.5236,
                    'Head_pitch': 0.785,

                    'Left_Shoulder_Pitch': 0.,
                    'Left_Shoulder_Roll': -1.3,
                    'Left_Elbow_Pitch': 0.,
                    'Left_Elbow_Yaw': -1.,
                    'Right_Shoulder_Pitch': 0.,
                    'Right_Shoulder_Roll': 1.3,
                    'Right_Elbow_Pitch': 0.,
                    'Right_Elbow_Yaw': 1.,
                    'Waist': 0.,

                    'Left_Hip_Pitch': -0.2,
                    'Left_Hip_Roll': 0.,
                    'Left_Hip_Yaw': 0.,
                    'Left_Knee_Pitch': 0.4,
                    'Left_Ankle_Pitch': -0.25,
                    'Left_Ankle_Roll': 0.,
                    'Right_Hip_Pitch': -0.2,
                    'Right_Hip_Roll': 0.,
                    'Right_Hip_Yaw': 0.,
                    'Right_Knee_Pitch': 0.4,
                    'Right_Ankle_Pitch': -0.25,
                    'Right_Ankle_Roll': 0.,
                },
            ),
        )

        # Terrain
        terrain = terrains.TerrainGeneratorCfg(
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.02,
            border_width=10.0,
            curriculum=True,
            sub_terrains={"flat": terrains.FlatCfg(proportion=1.0, size=(8.0, 8.0))},
        )

    scene: T1SceneCfg = T1SceneCfg()

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
    env = T1DreamWaqEnvCfg()

    # Algorithm (PPO)
    algorithm = PPOCfg()
    algorithm.num_steps_per_update = 24
    algorithm.actor_critic.actor_input_size = 50
    algorithm.actor_critic.action_size = 13
    algorithm.actor_critic.critic_obs_shape = (50, 62)
    algorithm.actor_critic.scan_shape = (32, 16)

    # Runner
    runner = OnPolicyRunnerCfg()
    runner.max_iterations = 20000
