"""Example locomotion environment using direct RL workflow.

This demonstrates how to create a locomotion task by inheriting from DirectRLEnv
and implementing compute_observations, compute_rewards, check_terminations.
"""
from __future__ import annotations

import re
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Sequence

import torch

from cross_gym.assets import Articulation
from cross_gym.envs import VecEnvStepReturn, DirectRLEnv
from cross_gym.terrains import TerrainGenerator
from cross_gym.utils import math as math_utils

if TYPE_CHECKING:
    from . import LocomotionEnvCfg


class LocomotionEnv(DirectRLEnv, ABC):
    """Simple locomotion environment."""

    cfg: LocomotionEnvCfg

    def __init__(self, cfg: LocomotionEnvCfg):
        """Initialize locomotion environment.
        
        Args:
            cfg: Locomotion environment configuration
        """
        super().__init__(cfg)

        # Locomotion-specific buffers
        self._init_buffers()

        # Prepare reward functions
        self._prepare_rewards()

    @property
    def num_envs(self):
        return self.cfg.scene.num_envs

    @property
    def device(self):
        return self.cfg.sim.device

    @property
    def robot(self) -> Articulation:
        return self.scene["robot"]

    @property
    def terrain(self) -> TerrainGenerator:
        return self.scene.terrain

    def _init_buffers(self):

        # Commands
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)  # [x, y, yaw, heading]

        # Actions and torques
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.target_dof_pos = torch.zeros(self.num_envs, self.robot.num_dof, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.robot.num_dof, device=self.device)

        # Environment origins and classes
        self._init_env_origins()

        # PD controller for locomotion
        self._init_pd_controller()

        # Domain randomization buffers
        self._init_domain_rand()

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
        terrain_class = torch.from_numpy(self.scene.terrain.terrain_type).to(self.device)  # [num_rows, num_cols]

        # Index using random (row, col) assignments
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_origins[:] = terrain_origins[env_rows, env_cols]

        self.env_class = torch.zeros(self.num_envs, device=self.device)
        self.env_class[:] = terrain_class[env_rows, env_cols]

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

    def _init_domain_rand(self):
        """Initialize domain randomization buffers."""
        if self.cfg.domain_rand.randomize_motor_offset:
            self.motor_offsets = torch.zeros(self.num_envs, self.robot.num_dof, device=self.device)

        if self.cfg.domain_rand.randomize_gains:
            self.p_gain_multiplier = torch.ones(self.num_envs, self.robot.num_dof, device=self.device)
            self.d_gain_multiplier = torch.ones(self.num_envs, self.robot.num_dof, device=self.device)

        if self.cfg.domain_rand.randomize_torque:
            self.torque_multiplier = torch.ones(self.num_envs, self.robot.num_dof, device=self.device)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coulomb = torch.zeros(self.num_envs, self.robot.num_dof, device=self.device)
            self.friction_viscous = torch.zeros(self.num_envs, self.robot.num_dof, device=self.device)

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment.

        Args:
            actions: Actions (num_envs, action_dim)

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        # Process actions
        self._process_action(actions)

        # Step simulation with decimation
        for _ in range(self.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.physics_dt)

        self._post_physics_step()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_truncated, self.extras

    def _process_action(self, actions: torch.Tensor):
        """Process policy actions into target joint positions.

        Args:
            actions: Policy actions (num_envs, num_actions)
        """
        # Clip and store actions
        self.actions[:] = torch.clip(
            actions, -self.cfg.control.clip_actions, self.cfg.control.clip_actions
        )

        # Compute target joint positions
        self.target_dof_pos[:] = self.robot.data.default_joint_pos + self.cfg.control.action_scale * self.actions

        # Apply motor offset randomization (simulates calibration error)
        if self.cfg.domain_rand.randomize_motor_offset:
            self.target_dof_pos[:] += self.motor_offsets

    def _apply_action(self):
        """Compute PD torques and apply to robot.
        
        Called every physics step (inside decimation loop).
        Computes torques from target positions using PD control,
        applies friction models, and sends to robot.
        """
        # Compute PD torques
        if self.cfg.domain_rand.randomize_gains:
            # Randomized gains
            self.torques[:] = self.p_gain_multiplier * self.p_gains * (self.target_dof_pos - self.robot.data.dof_pos)
            self.torques[:] -= self.d_gain_multiplier * self.d_gains * self.robot.data.dof_vel
        else:
            # Standard PD control
            self.torques[:] = self.p_gains * (self.target_dof_pos - self.robot.data.dof_pos)
            self.torques[:] -= self.d_gains * self.robot.data.dof_vel

        # Apply friction model (Coulomb + viscous)
        if self.cfg.domain_rand.randomize_friction:
            # Viscous friction: proportional to velocity
            self.torques[:] -= self.robot.data.dof_vel * self.friction_viscous

            # Coulomb friction: constant, opposes motion
            self.torques[:] -= torch.sign(self.robot.data.dof_vel) * self.friction_coulomb

        # Apply torque randomization (simulates actuator variance)
        if self.cfg.domain_rand.randomize_torque:
            self.torques.mul_(self.torque_multiplier)

        # Apply torques to robot
        self.robot.set_dof_effort_target(self.torques)

    def _post_physics_step(self):
        """Post-physics step callback."""
        # Update counters
        self.common_step_counter += 1  # Global step counter (for curriculum)
        self.episode_length_buf.add_(1)  # Per-environment episode length

        # Compute rewards
        self.compute_rewards()

        # Check terminations
        self.check_terminations()
        self.reset_buf[:] = self.reset_terminated | self.reset_truncated

        # Reset envs
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)

        # Compute observations
        self.obs_buf = self.compute_observations()

    def check_terminations(self):
        """Check termination conditions.
        
        Returns:
            Boolean tensor (num_envs,)
        """
        self.reset_terminated[:] = False
        self.reset_truncated[:] = False

        # Timeout
        self.reset_truncated[:] |= self.episode_length_buf >= self.max_episode_length

        # Collision  TODO
        # self.terminated_buf[:] |= self.robot.data.net_contact_forces[:, ]

        # Height cutoff
        self.reset_terminated[:] |= self.robot.data.root_pos_w[:, 2] < -10.

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments with domain randomization."""
        # Call parent reset first (resets to defaults)
        super()._reset_idx(env_ids)

        self._reset_root_state(env_ids)
        self._reset_joint_state(env_ids)
        self._resample_commands(env_ids)

        # ========== Randomize Torque Computation Parameters ==========
        self._randomize_dof_parameters(env_ids)

    def _reset_root_state(self, env_ids: Sequence[int]):
        num_resets = len(env_ids)

        # ========== Root State (from articulation defaults + terrain origins) ==========
        # Position: terrain origin + default offset
        root_pos = self.env_origins[env_ids] + self.robot.data.default_root_pos[env_ids]

        # Randomize xy position
        if self.cfg.domain_rand.randomize_start_pos_xy:
            min_val, max_val = self.cfg.domain_rand.randomize_start_pos_xy_range
            root_pos[:, :2] += math_utils.torch_rand_float_2d(
                min_val, max_val, (num_resets, 2), device=self.device
            )

        # Randomize z height
        if self.cfg.domain_rand.randomize_start_pos_z:
            min_val, max_val = self.cfg.domain_rand.randomize_start_pos_z_range
            root_pos[:, 2] += torch.abs(math_utils.torch_rand_float_1d(
                min_val, max_val, num_resets, device=self.device
            ))

        # Orientation: default + random rotation
        root_quat = self.robot.data.default_root_quat[env_ids].clone()

        # Randomize pitch and yaw
        rand_euler = torch.zeros(num_resets, 3, device=self.device)

        if self.cfg.domain_rand.randomize_start_pitch:
            min_val, max_val = self.cfg.domain_rand.randomize_start_pitch_range
            rand_euler[:, 1] = math_utils.torch_rand_float_1d(
                min_val, max_val, num_resets, device=self.device
            )

        if self.cfg.domain_rand.randomize_start_yaw:
            min_val, max_val = self.cfg.domain_rand.randomize_start_yaw_range
            rand_euler[:, 2] = math_utils.torch_rand_float_1d(
                min_val, max_val, num_resets, device=self.device
            )

        # Apply random rotation
        if self.cfg.domain_rand.randomize_start_pitch or self.cfg.domain_rand.randomize_start_yaw:
            rand_quat = math_utils.quat_from_euler_xyz(rand_euler)
            root_quat = math_utils.quat_mul(rand_quat, root_quat)

        # Velocity: default + randomization
        root_lin_vel = self.robot.data.default_root_lin_vel[env_ids].clone()
        root_ang_vel = self.robot.data.default_root_ang_vel[env_ids].clone()

        if self.cfg.domain_rand.randomize_start_lin_vel_xy:
            min_val, max_val = self.cfg.domain_rand.randomize_start_lin_vel_xy_range
            root_lin_vel[:, :2] += math_utils.torch_rand_float_2d(
                min_val, max_val, (num_resets, 2), device=self.device
            )

        # ========== Apply State to Simulation ==========
        self.robot.write_root_pos_to_sim(pos=root_pos, env_ids=env_ids)
        self.robot.write_root_quat_to_sim(quat=root_quat, env_ids=env_ids)
        self.robot.write_root_lin_vel_to_sim(lin_vel=root_lin_vel, env_ids=env_ids)
        self.robot.write_root_ang_vel_to_sim(ang_vel=root_ang_vel, env_ids=env_ids)

    def _reset_joint_state(self, env_ids: Sequence[int]):
        num_resets = len(env_ids)

        # ========== Joint State ==========
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        if self.cfg.domain_rand.randomize_start_dof_pos:
            min_val, max_val = self.cfg.domain_rand.randomize_start_dof_pos_range
            joint_pos[:] += math_utils.torch_rand_float_2d(
                min_val, max_val, (num_resets, self.robot.num_dof), device=self.device
            )

        if self.cfg.domain_rand.randomize_start_dof_vel:
            min_val, max_val = self.cfg.domain_rand.randomize_start_dof_vel_range
            joint_vel[:] += math_utils.torch_rand_float_2d(
                min_val, max_val, (num_resets, self.robot.num_dof), device=self.device
            )

        # ========== Apply State to Simulation ==========
        self.robot.write_joint_pos_to_sim(joint_pos=joint_pos, env_ids=env_ids)
        self.robot.write_joint_vel_to_sim(joint_vel=joint_vel, env_ids=env_ids)

    def _randomize_dof_parameters(self, env_ids: Sequence[int]):
        """Randomize DOF parameters for robustness.
        
        Args:
            env_ids: Environment IDs to randomize
        """
        num_resets = len(env_ids)

        # Randomize motor offsets (calibration error)
        if self.cfg.domain_rand.randomize_motor_offset:
            min_val, max_val = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids] = math_utils.torch_rand_float_2d(
                min_val, max_val, (num_resets, self.robot.num_dof), device=self.device
            )

        # Randomize PD gain multipliers (model uncertainty)
        if self.cfg.domain_rand.randomize_gains:
            min_val, max_val = self.cfg.domain_rand.kp_multiplier_range
            self.p_gain_multiplier[env_ids] = math_utils.torch_rand_float_2d(
                min_val, max_val, (num_resets, self.robot.num_dof), device=self.device
            )

            min_val, max_val = self.cfg.domain_rand.kd_multiplier_range
            self.d_gain_multiplier[env_ids] = math_utils.torch_rand_float_2d(
                min_val, max_val, (num_resets, self.robot.num_dof), device=self.device
            )

        # Randomize torque multiplier (actuator variance)
        if self.cfg.domain_rand.randomize_torque:
            min_val, max_val = self.cfg.domain_rand.torque_multiplier_range
            self.torque_multiplier[env_ids] = math_utils.torch_rand_float_2d(
                min_val, max_val, (num_resets, self.robot.num_dof), device=self.device
            )

        # Randomize friction (joint resistance)
        if self.cfg.domain_rand.randomize_friction:
            min_val, max_val = self.cfg.domain_rand.friction_coulomb_range
            self.friction_coulomb[env_ids] = math_utils.torch_rand_float_2d(
                min_val, max_val, (num_resets, self.robot.num_dof), device=self.device
            )

            min_val, max_val = self.cfg.domain_rand.friction_viscous_range
            self.friction_viscous[env_ids] = math_utils.torch_rand_float_2d(
                min_val, max_val, (num_resets, self.robot.num_dof), device=self.device
            )

    # ===== Implement Abstract Methods =====
    @abstractmethod
    def _resample_commands(self, env_ids: Sequence[int]):
        pass

    # ---------------------------------------------- Reward ----------------------------------------------

    @staticmethod
    def linear_interpolation(
            start: float,
            end: float,
            span: float,
            start_it: int,
            cur_it: int
    ) -> float:
        """Linear interpolation between start and end values.
        
        Args:
            start: Starting value
            end: Ending value
            span: Total span of interpolation
            start_it: Starting iteration
            cur_it: Current iteration
            
        Returns:
            Interpolated value
        """
        cur_value = start + (end - start) * (cur_it - start_it) / span
        cur_value = max(cur_value, min(start, end))
        cur_value = min(cur_value, max(start, end))
        return cur_value

    def update_reward_curriculum(self, epoch: int):
        """Update reward curriculum.
        
        Args:
            epoch: Current epoch
        """
        if self.cfg.rewards.only_positive_rewards:
            self.only_positive_rewards = epoch < self.cfg.rewards.only_positive_rewards_until_epoch

        for rew_name, prop in self.reward_scales_variable.items():
            self.reward_scales[rew_name] = self.linear_interpolation(*prop, epoch)

    def _prepare_rewards(self):
        """Prepare reward functions and scales."""
        self.only_positive_rewards = self.cfg.rewards.only_positive_rewards

        self._reward_names: list[str] = []
        self._reward_functions: list[callable] = []
        self.reward_scales: dict[str, float] = {}
        self.reward_scales_variable: dict[str, tuple] = {}  # For variable reward scales

        for name, scale in self.cfg.rewards.scales.items():
            self._reward_names.append(name)
            self._reward_functions.append(getattr(self, '_reward_' + name))

            # Handle variable scales (tuples for curriculum)
            if isinstance(scale, tuple):
                self.reward_scales[name] = scale[0]  # Start value
                self.reward_scales_variable[name] = scale
            elif isinstance(scale, (float, int)):
                self.reward_scales[name] = float(scale)
            else:
                raise ValueError(f"Invalid reward scale type: {type(scale)}")

        # Episode sums for logging
        self.episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in self.reward_scales.keys()
        }

    def compute_rewards(self):
        """Compute total reward from individual reward terms."""
        self.reward_buf[:] = 0.
        self.extras['reward_cur_step'] = {}

        # Sum weighted rewards
        for i, name in enumerate(self._reward_names):
            rew = self._reward_functions[i]() * self.reward_scales[name] * self.dt
            self.reward_buf[:] += rew
            self.episode_sums[name] += rew
            self.extras['reward_cur_step'][name] = rew

        self.extras['reward_raw'] = self.reward_buf.clone()

        # Clip to positive if curriculum enabled
        if self.only_positive_rewards:
            self.reward_buf[:] = torch.clip(self.reward_buf, min=0.)
