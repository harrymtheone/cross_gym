"""Example locomotion environment using direct RL workflow.

This demonstrates how to create a locomotion task by inheriting from DirectRLEnv
and implementing compute_observations, compute_rewards, check_terminations.
"""
from __future__ import annotations

import re
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from cross_gym.assets import Articulation
from cross_gym.envs import DirectRLEnv, DirectRLEnvCfg
from cross_gym.managers import RewardManager, RewardManagerCfg, ManagerTermCfg
from cross_gym.terrains import TerrainGenerator
from cross_gym.utils import configclass
from cross_gym.utils import math as math_utils
from . import rewards

if TYPE_CHECKING:
    from . import LocomotionEnvCfg


class LocomotionEnv(DirectRLEnv):
    """Simple locomotion environment.
    
    Trains a quadruped/biped to walk forward using PD control.
    """

    def __init__(self, cfg: LocomotionEnvCfg):
        """Initialize locomotion environment.
        
        Args:
            cfg: Locomotion environment configuration
        """
        super().__init__(cfg)

        # Locomotion-specific buffers
        self._init_buffers()

        # Reward manager
        self.reward_manager = RewardManager(cfg.rewards, self)

    @property
    def robot(self) -> Articulation:
        return self.scene["robot"]

    @property
    def terrain(self) -> TerrainGenerator:
        return self.scene.terrain

    def _init_buffers(self):
        # Actions and torques
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.robot.num_dof, device=self.device)

        # PD controller for locomotion
        self._init_pd_controller()

        # Environment origins and classes
        self._init_env_origins()

    def _init_env_origins(self):
        """Initialize environment spawn origins.
        
        If terrain exists, randomly assigns each environment to a terrain patch.
        Otherwise, creates a grid layout with env_spacing.
        """
        # Randomly assign each environment to a terrain patch (row, col)
        num_rows = self.scene.terrain.num_rows
        num_cols = self.scene.terrain.num_cols

        # Random (row, col) for each environment
        env_rows = torch.randint(0, num_rows, (self.num_envs,), device=self.device)
        env_cols = torch.randint(0, num_cols, (self.num_envs,), device=self.device)

        # Get terrain origins and types
        terrain_origins = torch.from_numpy(self.scene.terrain.terrain_origins).to(self.device)  # [num_rows, num_cols, 3]

        # Index using random (row, col) assignments
        self.env_origins = terrain_origins[env_rows, env_cols]

    def _init_pd_controller(self):
        """Initialize PD controller for locomotion."""
        self.p_gains = torch.zeros(1, self.robot.num_dof, device=self.device)
        self.d_gains = torch.zeros(1, self.robot.num_dof, device=self.device)

        for i, dof_name in enumerate(self.robot.dof_names):
            # Find matching PD gains using regex
            for pattern, kp in self.cfg.control.stiffness.items():
                if re.search(pattern, dof_name):
                    self.p_gains[0, i] = kp
                    break

            for pattern, kd in self.cfg.control.damping.items():
                if re.search(pattern, dof_name):
                    self.d_gains[0, i] = kd
                    break

    def process_actions(self, actions: torch.Tensor):
        """Process actions using PD controller.
        
        Args:
            actions: Policy actions (num_envs, num_actions)
        """
        # Store actions
        self.actions[:] = torch.clip(
            actions,
            -self.cfg.control.clip_actions,
            self.cfg.control.clip_actions
        )

        # Compute torques using PD controller
        target_dof_pos = self.actions * self.cfg.control.action_scale + self.robot.data.default_root_pos
        self.torques[:] = self.p_gains * (target_dof_pos - self.robot.data.dof_pos)
        self.torques[:] -= self.d_gains * self.robot.data.dof_vel

        # Apply torques
        self.robot.set_dof_effort_target(self.torques)

    def step(self, actions: torch.Tensor):
        """Step with base state refresh."""
        # Call parent step
        result = super().step(actions)

        return result

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments with domain randomization."""
        # Call parent reset first (resets to defaults)
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # ========== Root State (from articulation defaults + terrain origins) ==========
        # Position: terrain origin + default offset
        root_pos = self.env_origins[env_ids] + self.robot.data.default_root_pos[env_ids]

        # Randomize xy position
        if self.cfg.domain_rand.randomize_start_pos_xy:
            min_val, max_val = self.cfg.domain_rand.randomize_start_pos_xy_range
            root_pos[:, :2] += math_utils.torch_rand_float(
                min_val, max_val, (num_resets, 2), device=self.device
            )

        # Randomize z height
        if self.cfg.domain_rand.randomize_start_pos_z:
            min_val, max_val = self.cfg.domain_rand.randomize_start_pos_z_range
            root_pos[:, 2] += torch.abs(math_utils.torch_rand_float(
                min_val, max_val, (num_resets, 1), device=self.device
            ).squeeze(1))

        # Orientation: default + random rotation
        root_quat = self.robot.data.default_root_quat[env_ids].clone()

        # Randomize pitch and yaw
        rand_euler = torch.zeros(num_resets, 3, device=self.device)

        if self.cfg.domain_rand.randomize_start_pitch:
            min_val, max_val = self.cfg.domain_rand.randomize_start_pitch_range
            rand_euler[:, 1] = math_utils.torch_rand_float(
                min_val, max_val, (num_resets, 1), device=self.device
            ).squeeze(1)

        if self.cfg.domain_rand.randomize_start_yaw:
            min_val, max_val = self.cfg.domain_rand.randomize_start_yaw_range
            rand_euler[:, 2] = math_utils.torch_rand_float(
                min_val, max_val, (num_resets, 1), device=self.device
            ).squeeze(1)

        # Apply random rotation
        if self.cfg.domain_rand.randomize_start_pitch or self.cfg.domain_rand.randomize_start_yaw:
            rand_quat = math_utils.quat_from_euler_xyz(rand_euler)
            root_quat = math_utils.quat_mul(rand_quat, root_quat)

        # Velocity: default + randomization
        root_lin_vel = self.robot.data.default_root_lin_vel[env_ids].clone()
        root_ang_vel = self.robot.data.default_root_ang_vel[env_ids].clone()

        if self.cfg.domain_rand.randomize_start_lin_vel_xy:
            min_val, max_val = self.cfg.domain_rand.randomize_start_lin_vel_xy_range
            root_lin_vel[:, :2] += math_utils.torch_rand_float(
                min_val, max_val, (num_resets, 2), device=self.device
            )

        # ========== Joint State ==========
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        if self.cfg.domain_rand.randomize_start_dof_pos:
            min_val, max_val = self.cfg.domain_rand.randomize_start_dof_pos_range
            joint_pos[:] += math_utils.torch_rand_float(
                min_val, max_val, (num_resets, self.robot.num_dof), device=self.device
            )

        if self.cfg.domain_rand.randomize_start_dof_vel:
            min_val, max_val = self.cfg.domain_rand.randomize_start_dof_vel_range
            joint_vel[:] += math_utils.torch_rand_float(
                min_val, max_val, (num_resets, self.robot.num_dof), device=self.device
            )

        # ========== Apply State to Simulation ==========
        self.robot.write_root_pos_to_sim(pos=root_pos, env_ids=env_ids)
        self.robot.write_root_quat_to_sim(quat=root_quat, env_ids=env_ids)
        self.robot.write_root_lin_vel_to_sim(lin_vel=root_lin_vel, env_ids=env_ids)
        self.robot.write_root_ang_vel_to_sim(ang_vel=root_ang_vel, env_ids=env_ids)
        self.robot.write_joint_pos_to_sim(joint_pos=joint_pos, env_ids=env_ids)
        self.robot.write_joint_vel_to_sim(joint_vel=joint_vel, env_ids=env_ids)

    # ===== Implement Abstract Methods =====

    def compute_rewards(self) -> torch.Tensor:
        """Compute rewards using reward manager.
        
        Returns:
            Reward tensor (num_envs,)
        """
        return self.reward_manager.compute(self.dt)

    def check_terminations(self):
        """Check termination conditions.
        
        Returns:
            Boolean tensor (num_envs,)
        """
        self.reset_terminated[:] = False
        self.reset_truncated[:] = False

        # Timeout
        self.reset_truncated[:] |= self.episode_length_buf >= self.max_episode_length

        # Collision
        # self.terminated_buf[:] |= self.robot.data.net_contact_forces[:, ]

        # Height cutoff
        self.reset_terminated[:] |= self.robot.data.root_pos_w[:, 2] < -10.


# ============================================================================
# Configuration
# ============================================================================

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

    # Initial state
    @configclass
    class LocomotionInitStateCfg(DirectRLEnvCfg.InitStateCfg):
        pos: tuple = (0.0, 0.0, 0.6)
        rot: tuple = (1.0, 0.0, 0.0, 0.0)

        # Default joint angles (adjust for your robot)
        default_joint_angles: dict = {
            ".*": 0.0,
        }

    init_state: LocomotionInitStateCfg = LocomotionInitStateCfg()

    # Rewards
    @configclass
    class LocomotionRewardsCfg(RewardManagerCfg):
        """Reward configuration for locomotion."""

        alive = ManagerTermCfg(func=rewards.alive, weight=1.0)

        forward_vel = ManagerTermCfg(
            func=rewards.lin_vel_x_tracking,
            weight=2.0,
            params={"target_vel": 1.0}
        )

        lateral_vel = ManagerTermCfg(func=rewards.lin_vel_y_penalty, weight=0.5)
        vertical_vel = ManagerTermCfg(func=rewards.lin_vel_z_penalty, weight=0.5)
        ang_vel = ManagerTermCfg(func=rewards.ang_vel_penalty, weight=0.1)
        energy = ManagerTermCfg(func=rewards.energy_penalty, weight=0.01)
        upright = ManagerTermCfg(func=rewards.upright_reward, weight=0.5)

    rewards: LocomotionRewardsCfg = LocomotionRewardsCfg()
