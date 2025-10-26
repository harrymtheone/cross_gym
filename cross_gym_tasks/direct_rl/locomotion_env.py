"""Example locomotion environment using direct RL workflow.

This demonstrates how to create a locomotion task by inheriting from DirectRLEnv
and implementing compute_observations, compute_rewards, check_terminations.
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from cross_gym.assets import Articulation
from cross_gym.envs import DirectRLEnv, DirectRLEnvCfg
from cross_gym.utils import configclass, math as math_utils

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

        # PD controller for locomotion
        self._init_pd_controller()

        # Target velocity
        self.target_velocity = torch.tensor([1.0, 0.0, 0.0], device=self.device)

    @property
    def robot(self) -> Articulation:
        return self.scene["robot"]

    def _init_buffers(self):
        """Initialize locomotion-specific buffers."""
        # Base state in base frame
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)

        # Actions and torques
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.robot.num_dof, device=self.device)

    def _init_pd_controller(self):
        """Initialize PD controller for locomotion."""
        # Default joint positions
        self.default_dof_pos = torch.zeros(1, self.robot.num_dof, device=self.device)
        self.p_gains = torch.zeros(1, self.robot.num_dof, device=self.device)
        self.d_gains = torch.zeros(1, self.robot.num_dof, device=self.device)

        for i, dof_name in enumerate(self.robot.dof_names):
            # Find matching joint angle
            for pattern, angle in self.cfg.init_state.default_joint_angles.items():
                if pattern in dof_name:
                    self.default_dof_pos[0, i] = angle
                    break

            # Find matching PD gains
            for pattern, kp in self.cfg.control.stiffness.items():
                if pattern in dof_name:
                    self.p_gains[0, i] = kp
                    break

            for pattern, kd in self.cfg.control.damping.items():
                if pattern in dof_name:
                    self.d_gains[0, i] = kd
                    break

    def _refresh_base_state(self):
        """Refresh base state in base frame.
        
        Uses cached properties from ArticulationData to avoid redundant computation.
        """
        # Use cached properties (computed once per timestep, reused multiple times)
        self.base_lin_vel = self.robot.data.root_lin_vel_b
        self.base_ang_vel = self.robot.data.root_ang_vel_b
        self.projected_gravity = self.robot.data.projected_gravity_b

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
        self.torques[:] = self.p_gains * (target_dof_pos - self.robot.data.joint_pos)
        self.torques[:] -= self.d_gains * self.robot.data.joint_vel

        # Apply torques
        self.robot.set_joint_effort_target(self.torques)

    def step(self, actions: torch.Tensor):
        """Step with base state refresh."""
        # Call parent step
        result = super().step(actions)

        # Refresh base state for next iteration
        self._refresh_base_state()

        return result

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset with base state refresh."""
        super()._reset_idx(env_ids)
        self._refresh_base_state()

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
            self.robot.data.joint_pos,  # num_dof
            self.robot.data.joint_vel,  # num_dof
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
