"""Parkour environment with terrain curriculum and goal-based navigation.

This extends LocomotionEnv with:
- Terrain curriculum (progressive difficulty)
- Goal-based navigation for parkour terrains
- Multi-modal commands (flat, stair, parkour)
- Height scanning for observations
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch

from cross_gym.terrains import TerrainCommandType
from cross_gym.utils import math as math_utils
from cross_gym_tasks.direct_rl import LocomotionEnv

if TYPE_CHECKING:
    from . import ParkourEnvCfg


class ParkourEnv(LocomotionEnv, ABC):
    """Parkour environment with curriculum learning and goal navigation."""

    cfg: ParkourEnvCfg

    def __init__(self, cfg: ParkourEnvCfg):
        """Initialize parkour environment.
        
        Args:
            cfg: Parkour environment configuration
        """
        super().__init__(cfg)

        self.curriculum = self.cfg.parkour.terrain_curriculum

    def _init_buffers(self):
        """Initialize buffers for parkour."""
        # Parent buffers (actions, torques, PD gains, env_origins)
        super()._init_buffers()

        # Parkour-specific buffers
        self._init_parkour_buffers()

    def _init_parkour_buffers(self):
        """Initialize parkour-specific buffers."""
        # Terrain curriculum
        self.env_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_terrain_level = self.scene.terrain.num_rows
        self.env_cmd_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Generate goals (per-environment randomization) - returns numpy arrays
        terrain_goals_np, terrain_goal_num_np = self.scene.terrain.generate_goals(self.num_envs)

        # Convert to torch tensors
        self.terrain_goals = torch.from_numpy(terrain_goals_np).to(self.device).float()
        self.terrain_goal_num = torch.from_numpy(terrain_goal_num_np).to(self.device).long()

        # Extract max goals dimension
        max_goals = self.terrain_goals.shape[3]

        # Per-environment goal tracking
        self.env_goals = torch.zeros(self.num_envs, max_goals, 3, device=self.device)
        self.env_goal_num = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Current goal tracking
        self.cur_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cur_goals = torch.zeros(self.num_envs, 3, device=self.device)
        self.reached_goal_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reach_goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.reach_goal_cutoff = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Navigation helpers
        self.target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)

        # Curriculum tracking
        self.num_trials = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Command-specific
        self.command_x_parkour = torch.zeros(self.num_envs, device=self.device)

    def _init_env_origins(self):
        """Initialize environment origins with curriculum support."""

        # With terrain and curriculum
        num_rows = self.scene.terrain.num_rows
        num_cols = self.scene.terrain.num_cols

        # Initialize terrain levels based on curriculum
        max_init_level = self.cfg.parkour.max_init_terrain_level
        if max_init_level >= num_rows:
            print(f"Warning: max_init_level ({max_init_level}) >= num_rows ({num_rows}), clipping to {num_rows - 1}")
            max_init_level = num_rows - 1

        if not self.curriculum:
            max_init_level = num_rows - 1

        # Random terrain levels (rows) for curriculum
        self.env_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), dtype=torch.long, device=self.device)

        # Distribute environments evenly across columns
        self.env_cols = torch.div(
            self._ALL_ENVS,
            (self.num_envs / num_cols),
            rounding_mode='floor'
        ).to(torch.long)

        # Get terrain data
        self.terrain_origins_tensor = torch.from_numpy(self.scene.terrain.terrain_origins).to(self.device)
        self.terrain_class_tensor = torch.from_numpy(self.scene.terrain.terrain_type).to(self.device)
        self.terrain_cmd_type_tensor = torch.from_numpy(self.scene.terrain.terrain_command_type).to(self.device)

        # Assign origins, classes, and command types based on (level, col)
        self.env_origins[:] = self.terrain_origins_tensor[self.env_levels, self.env_cols]
        self.env_class[:] = self.terrain_class_tensor[self.env_levels, self.env_cols]
        self.env_cmd_type[:] = self.terrain_cmd_type_tensor[self.env_levels, self.env_cols]

        # Assign goals - terrain_goals shape: (num_rows, num_cols, num_envs, max_goals, 3)
        self.env_goals[:] = self.terrain_goals[self.env_levels, self.env_cols, self._ALL_ENVS]
        self.env_goal_num[:] = self.terrain_goal_num[self.env_levels, self.env_cols]

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments with curriculum update."""
        # Update curriculum before reset
        if self.curriculum:
            self._update_terrain_curriculum(env_ids)

        # Parent reset (state + domain rand)
        super()._reset_idx(env_ids)

        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0

    def _resample_commands(self, env_ids: torch.Tensor):
        """Resample commands based on terrain command type.
        
        Three modes:
        - Omni (0): Omnidirectional (vx, vy, yaw_rate) - for flat terrains
        - Heading (1): Heading-based (vx, vy, heading) - for stair terrains
        - Goal (2): Goal-guided (vx, auto yaw_rate) - for parkour terrains
        """
        self.commands[env_ids] = 0

        # Sample based on command type
        # Omni: flat terrain commands
        env_cmd_omni = torch.eq(self.env_cmd_type[env_ids], TerrainCommandType.Omni.value)
        if torch.any(env_cmd_omni):
            self._resample_flat_commands(env_ids[env_cmd_omni])

        # Heading: stair terrain commands
        env_cmd_heading = torch.eq(self.env_cmd_type[env_ids], TerrainCommandType.Heading.value)
        if torch.any(env_cmd_heading):
            self._resample_stair_commands(env_ids[env_cmd_heading])

        # Goal: parkour terrain commands
        env_cmd_goal = torch.eq(self.env_cmd_type[env_ids], TerrainCommandType.Goal.value)
        if torch.any(env_cmd_goal):
            self._resample_parkour_commands(env_ids[env_cmd_goal])

    def _resample_flat_commands(self, env_ids: torch.Tensor):
        """Sample omnidirectional commands for flat terrain."""
        num_resample_envs = len(env_ids)
        cmd_cfg = self.cfg.commands

        # Sample linear velocities
        self.commands[env_ids, 0] = self._sample_command(
            cmd_cfg.flat_ranges.lin_vel_x, cmd_cfg.lin_vel_clip, num_resample_envs
        )
        self.commands[env_ids, 1] = self._sample_command(
            cmd_cfg.flat_ranges.lin_vel_y, cmd_cfg.lin_vel_clip, num_resample_envs
        )

        # Sample yaw rate
        self.commands[env_ids, 2] = self._sample_command(
            cmd_cfg.flat_ranges.ang_vel_yaw, cmd_cfg.ang_vel_clip, num_resample_envs
        )

    def _resample_stair_commands(self, env_ids: torch.Tensor):
        """Sample heading-based commands for stair terrain."""
        num_resample_envs = len(env_ids)
        cmd_cfg = self.cfg.commands

        # Sample linear velocities
        self.commands[env_ids, 0] = self._sample_command(
            cmd_cfg.stair_ranges.lin_vel_x, cmd_cfg.lin_vel_clip, num_resample_envs
        )
        self.commands[env_ids, 1] = self._sample_command(
            cmd_cfg.stair_ranges.lin_vel_y, cmd_cfg.lin_vel_clip, num_resample_envs
        )

        # Sample heading (stored in index 3, converted to yaw_rate in update_command)
        self.commands[env_ids, 3] = self._sample_command(
            cmd_cfg.stair_ranges.heading, cmd_cfg.ang_vel_clip, num_resample_envs
        )

    def _resample_parkour_commands(self, env_ids: torch.Tensor):
        """Sample goal-guided commands for parkour terrain."""
        num_resample_envs = len(env_ids)
        cmd_cfg = self.cfg.commands

        # Sample forward velocity only (yaw controlled by goal direction)
        self.commands[env_ids, 0] = self._sample_command(
            cmd_cfg.parkour_ranges.lin_vel_x, cmd_cfg.lin_vel_clip, num_resample_envs
        )

        # Store for goal-based modulation
        self.command_x_parkour[env_ids] = self.commands[env_ids, 0]

    def _sample_command(
            self,
            cmd_range: tuple[float, float],
            min_clip: float,
            num_samples: int
    ) -> torch.Tensor:
        """Sample command with clipping logic.
        
        If range has same sign, samples with minimum magnitude = clip.
        If range spans zero, samples normally but avoids dead zone around zero.
        
        Args:
            cmd_range: (min, max) range
            min_clip: Minimum magnitude (dead zone)
            num_samples: Number of samples
            
        Returns:
            Sampled commands
        """
        min_val, max_val = cmd_range

        if abs(min_val) < min_clip:
            raise ValueError(f"Abs of range_min ({min_val}) should be >= clip ({min_clip})")
        if abs(max_val) < min_clip:
            raise ValueError(f"Abs of range_max ({max_val}) should be >= clip ({min_clip})")

        if min_val * max_val > 0:  # Same sign
            return math_utils.torch_rand_float(min_val, max_val, (num_samples, 1), device=self.device).squeeze(1)

        else:  # Different sign (spans zero)
            cmd = math_utils.torch_rand_float(min_val, max_val, (num_samples, 1), device=self.device).squeeze(1)

            # Scale to avoid dead zone [-clip, clip]
            ratio = torch.where(
                cmd < 0,
                (min_val + min_clip) / min_val if min_val != 0 else 1.0,
                (max_val - min_clip) / max_val if max_val != 0 else 1.0
            )
            return cmd * ratio + torch.sign(cmd) * min_clip

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        """Update terrain difficulty based on performance (game-inspired curriculum).
        
        Args:
            env_ids: Environment IDs being reset
        """
        # Compute distance traveled from origin
        dis_to_origin = torch.norm(
            self.robot.data.root_pos_w[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        threshold = torch.norm(self.commands[env_ids, :2], dim=1) * self.episode_length_buf[env_ids] * self.dt

        move_up = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        move_down = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        # Curriculum logic for flat terrain (Omni command type)  TODO: we should obtain the actual terrain size of the terrain
        env_is_omni = torch.eq(self.env_cmd_type[env_ids], TerrainCommandType.Omni.value)
        if torch.any(env_is_omni):
            move_up[env_is_omni] = dis_to_origin[env_is_omni] > self.cfg.parkour.terrain_size[0] / 4  # TODO: using cfg.terrain_size is a temporary expression!
            move_down[env_is_omni] = dis_to_origin[env_is_omni] < self.cfg.parkour.terrain_size[0] / 8

        # Curriculum logic for stair terrain (Heading command type)  TODO: we should obtain the actual terrain size of the terrain
        env_is_heading = torch.eq(self.env_cmd_type[env_ids], TerrainCommandType.Heading.value)
        if torch.any(env_is_heading):
            move_up[env_is_heading] = dis_to_origin[env_is_heading] > self.cfg.parkour.terrain_size[0] / 2
            move_down[env_is_heading] = dis_to_origin[env_is_heading] < 0.4 * threshold[env_is_heading]

        # Curriculum logic for parkour terrain (Goal command type)
        env_is_goal = torch.eq(self.env_cmd_type[env_ids], TerrainCommandType.Goal.value)
        if torch.any(env_is_goal):
            # Move up if reached all goals
            move_up[env_is_goal] = (self.cur_goal_idx[env_ids][env_is_goal] >= self.env_goal_num[env_ids][env_is_goal])
            # Move down if didn't reach half the goals
            move_down[env_is_goal] = (self.cur_goal_idx[env_ids][env_is_goal] < self.env_goal_num[env_ids][env_is_goal] // 2)

        # Update levels
        self.env_levels[env_ids] += move_up.to(torch.long) - move_down.to(torch.long)

        # Track trials for downgrade after multiple failures
        level_changed = move_up ^ move_down
        self.num_trials[env_ids[level_changed]] = 0
        self.num_trials[env_ids[~level_changed]] += 1

        # Downgrade after 5 consecutive failures
        downgrade_mask = self.num_trials[env_ids] >= 5
        if torch.any(downgrade_mask):
            # Random level below current
            self.env_levels[env_ids[downgrade_mask]] = (
                    torch.rand(downgrade_mask.sum(), device=self.device) * self.env_levels[env_ids[downgrade_mask]]
            ).long()
            self.num_trials[env_ids[downgrade_mask]] = 0

        # Clamp levels and wrap around at max
        self.env_levels[env_ids] = torch.where(
            self.env_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.env_levels[env_ids], self.max_terrain_level),
            torch.clamp(self.env_levels[env_ids], min=0)
        )

        # Update env origins and goals based on new levels
        env_levels, env_cols = self.env_levels[env_ids], self.env_cols[env_ids]
        self.env_origins[env_ids] = self.terrain_origins_tensor[env_levels, env_cols]
        self.env_class[env_ids] = self.terrain_class_tensor[env_levels, env_cols]
        self.env_cmd_type[env_ids] = self.terrain_cmd_type_tensor[env_levels, env_cols]

        # Update goals - need to index with env dimension
        self.env_goals[:] = self.terrain_goals[self.env_levels, self.env_cols, self._ALL_ENVS]
        self.env_goal_num[:] = self.terrain_goal_num[self.env_levels, self.env_cols]

    def _update_goal_tracking(self):
        """Update goal tracking for parkour terrains."""

        # Update current goal for each environment
        self.cur_goals[:] = self.env_goals[self._ALL_ENVS, self.cur_goal_idx]

        # Check if reached current goal (only for Goal command type / parkour terrains)
        dist = torch.norm(self.robot.data.root_pos_w[:, :2] - self.cur_goals[:, :2], dim=1)
        env_is_goal = torch.eq(self.env_cmd_type, TerrainCommandType.Goal.value)
        self.reached_goal_env[:] = (dist < self.cfg.parkour.next_goal_threshold) & env_is_goal

        # Update goal timer
        self.reach_goal_timer[self.reached_goal_env] += 1
        self.reach_goal_timer[~self.reached_goal_env] = 0

        # Move to next goal after delay
        next_goal_flag = self.reach_goal_timer > self.cfg.parkour.reach_goal_delay / self.dt
        self.cur_goal_idx[next_goal_flag] += 1

    def _update_parkour_commands(self):
        """Update commands for heading/goal-based navigation."""
        cmd_cfg = self.cfg.commands

        # Update target position and yaw for parkour
        self.target_pos_rel[:] = self.cur_goals[:, :2] - self.robot.data.root_pos_w[:, :2]
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw[:] = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        # Compute yaw error for parkour (every 5 steps)
        if self.common_step_counter > 5 and self.common_step_counter % 5 == 0:
            base_yaw = self.robot.data.root_euler_w[:, 2]
            self.delta_yaw[:] = math_utils.wrap_to_pi(self.target_yaw - base_yaw) * (self.command_x_parkour > cmd_cfg.lin_vel_clip)

        # Update yaw command for stair terrain (Heading command type)
        env_is_heading = torch.eq(self.env_cmd_type, TerrainCommandType.Heading.value)
        if env_is_heading.any():
            base_yaw = self.robot.data.root_euler_w[env_is_heading, 2]
            heading_error = math_utils.wrap_to_pi(self.commands[env_is_heading, 3] - base_yaw)
            self.commands[env_is_heading, 2] = torch.clamp(
                heading_error,
                min=cmd_cfg.stair_ranges.ang_vel_yaw[0],
                max=cmd_cfg.stair_ranges.ang_vel_yaw[1]
            )

        # Update yaw command for parkour terrain (Goal command type)
        env_is_goal = torch.eq(self.env_cmd_type, TerrainCommandType.Goal.value)
        if env_is_goal.any():
            delta_yaw_error = self.delta_yaw[env_is_goal]
            self.commands[env_is_goal, 2] = torch.clamp(
                delta_yaw_error,
                min=cmd_cfg.parkour_ranges.ang_vel_yaw[0],
                max=cmd_cfg.parkour_ranges.ang_vel_yaw[1]
            )

            # Modulate forward velocity based on yaw alignment
            cmd_ratio = torch.clamp(1 - torch.abs(delta_yaw_error / torch.pi), min=0)
            self.commands[env_is_goal, 0] = cmd_ratio * self.command_x_parkour[env_is_goal]

    def step(self, actions: torch.Tensor):
        """Step with parkour-specific updates."""
        # Update goal tracking
        self._update_goal_tracking()

        # Parent step
        result = super().step(actions)

        # Update parkour-specific commands
        self._update_parkour_commands()

        return result

    def check_terminations(self):
        """Check terminations including goal completion."""
        # Parent terminations (timeout, height cutoff)
        super().check_terminations()

        # Parkour: terminate if all goals reached (Goal command type only)
        env_is_goal = torch.eq(self.env_cmd_type, TerrainCommandType.Goal.value)
        self.reach_goal_cutoff[:] = (self.cur_goal_idx >= self.env_goal_num) & env_is_goal
        self.reset_truncated[:] |= self.reach_goal_cutoff
