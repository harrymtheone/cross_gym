"""Example locomotion environment using direct RL workflow.

This demonstrates how to create a locomotion task by inheriting from DirectRLEnv
and implementing compute_observations, compute_rewards, check_terminations.
"""
from __future__ import annotations

import math
import re
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from cross_gym.assets import Articulation
from cross_gym.envs import DirectRLEnv, DirectRLEnvCfg
from cross_gym.utils import configclass
from cross_gym.utils import math as math_utils
from cross_gym.terrains import TerrainGenerator

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
        target_dof_pos = self.actions * self.cfg.control.action_scale + self.default_dof_pos
        self.torques[:] = self.p_gains * (target_dof_pos - self.robot.data.dof_pos)
        self.torques[:] -= self.d_gains * self.robot.data.dof_vel

        # Apply torques
        self.robot.set_joint_effort_target(self.torques)

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
        self.robot.set_root_state(
            pos=root_pos,
            quat=root_quat,
            lin_vel=root_lin_vel,
            ang_vel=root_ang_vel,
            env_ids=env_ids
        )
        
        self.robot.set_joint_state(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            env_ids=env_ids
        )

    # ===== Implement Abstract Methods =====

    def compute_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations for locomotion.
        
        Returns:
            Dictionary with 'policy' observations
        """
        obs = torch.cat([
            self.base_lin_vel,  # 3
            self.base_ang_vel,  # 3
            self.projected_gravity,  # 3
            self.robot.data.dof_pos,  # num_dof
            self.robot.data.dof_vel,  # num_dof
            self.actions,  # num_actions (last action)
        ], dim=-1)

        return {"policy": obs}

    def compute_rewards(self) -> torch.Tensor:
        """Compute rewards for forward walking.
        
        Returns:
            Reward tensor (num_envs,)
        """
        # Alive reward
        reward = torch.ones(self.num_envs, device=self.device)

        # Forward velocity tracking
        forward_vel = self.base_lin_vel[:, 0]
        target_vel = self.target_velocity[0]
        reward += 2.0 * torch.exp(-torch.abs(forward_vel - target_vel))

        # Penalize lateral and vertical velocity
        reward -= 0.5 * torch.abs(self.base_lin_vel[:, 1])
        reward -= 0.5 * torch.abs(self.base_lin_vel[:, 2])

        # Penalize angular velocity
        reward -= 0.1 * torch.sum(torch.abs(self.base_ang_vel), dim=-1)

        # Energy penalty
        reward -= 0.01 * torch.sum(self.torques ** 2, dim=-1)

        # Upright reward
        reward += 0.5 * self.projected_gravity[:, 2]

        return reward

    def check_terminations(self) -> torch.Tensor:
        """Check termination conditions.
        
        Returns:
            Boolean tensor (num_envs,)
        """
        # Timeout
        timeout = self.episode_length_buf >= self.max_episode_length

        # Base too low or too high
        base_height = self.robot.data.root_pos_w[:, 2]
        height_termination = (base_height < 0.3) | (base_height > 2.0)

        return timeout | height_termination


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
