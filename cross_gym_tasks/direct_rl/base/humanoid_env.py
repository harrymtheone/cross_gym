"""Humanoid environment with bipedal gait control.

This extends ParkourEnv with:
- Bipedal gait phase tracking
- Feet contact and air time management
- Foothold quality detection
- Humanoid-specific reward functions
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Sequence

import torch

from cross_gym.utils import math as math_utils

if TYPE_CHECKING:
    from . import HumanoidEnvCfg, ParkourEnv


class HumanoidEnv(ParkourEnv, ABC):
    """Humanoid locomotion environment with bipedal gait control.
    
    Adds humanoid-specific features on top of parkour:
    - Bipedal gait phase tracking (stance/swing coordination)
    - Feet and knee tracking
    - Contact filtering and air time measurement
    - Foothold quality detection
    - Specialized humanoid rewards (feet clearance, slip, etc.)
    """

    cfg: HumanoidEnvCfg

    def __init__(self, cfg: HumanoidEnvCfg):
        """Initialize humanoid environment.
        
        Args:
            cfg: Humanoid environment configuration
        """
        super().__init__(cfg)

    def _setup_scene(self):
        """Set up humanoid-specific scene elements."""
        super()._setup_scene()

        # Find feet and knee bodies
        self._resolve_humanoid_bodies()

    def _resolve_humanoid_bodies(self):
        """Resolve feet and knee body indices from robot.
        
        Uses regex patterns from config to find body indices.
        """
        # Get all body names
        body_names = self.robot.body_names

        # Find feet indices
        feet_names = self._find_bodies_matching(self.cfg.asset.foot_name)
        self.feet_indices = torch.tensor(
            [body_names.index(name) for name in feet_names],
            dtype=torch.long,
            device=self.device
        )

        # Find knee indices  
        knee_names = self._find_bodies_matching(self.cfg.asset.knee_name)
        self.knee_indices = torch.tensor(
            [body_names.index(name) for name in knee_names],
            dtype=torch.long,
            device=self.device
        )

        print(f"[HumanoidEnv] Found {len(self.feet_indices)} feet: {feet_names}")
        print(f"[HumanoidEnv] Found {len(self.knee_indices)} knees: {knee_names}")

    def _find_bodies_matching(self, pattern: str) -> list[str]:
        """Find body names matching a regex pattern.
        
        Args:
            pattern: Regex pattern to match
            
        Returns:
            List of matching body names
        """
        import re
        body_names = self.robot.body_names
        regex = re.compile(pattern)
        return [name for name in body_names if regex.match(name)]

    def _init_buffers(self):
        """Initialize buffers for humanoid."""
        super()._init_buffers()

        # Humanoid-specific buffers
        self._init_humanoid_buffers()

    def _init_humanoid_buffers(self):
        """Initialize humanoid-specific buffers."""
        num_feet = len(self.feet_indices)

        # Feet contact tracking
        self.feet_air_time = torch.zeros(self.num_envs, num_feet, device=self.device)
        self.feet_air_time_avg = torch.zeros(self.num_envs, num_feet, device=self.device) + 0.1
        self.contact_filt = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device)
        self.contact_forces_avg = torch.zeros(self.num_envs, num_feet, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device)

        # Gait phase tracking
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_length_buf = torch.zeros(self.num_envs, device=self.device)
        self.gait_start = torch.zeros(self.num_envs, device=self.device)

        # Feet state tracking
        self.last_feet_vel_xy = torch.zeros(self.num_envs, num_feet, 2, device=self.device)
        self.feet_height = torch.zeros(self.num_envs, num_feet, device=self.device)
        self.feet_euler_xyz = torch.zeros(self.num_envs, num_feet, 3, device=self.device)

        # Foothold detection (if terrain available)
        if hasattr(self.cfg, 'terrain') and self.cfg.terrain is not None:
            self.feet_at_edge = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device)

            # Generate foothold scan points
            self.foothold_pts_local = self._get_foothold_points()
            n_points = self.foothold_pts_local.size(1)
            self.foothold_pts_pos = torch.zeros(self.num_envs, num_feet, n_points, 3, device=self.device)
            self.foothold_pts_contact = torch.zeros(self.num_envs, num_feet, n_points, dtype=torch.bool, device=self.device)

    def _get_foothold_points(self) -> torch.Tensor:
        """Generate foothold detection points in local frame.
        
        Creates a small grid of points under each foot for contact detection.
        
        Returns:
            Foothold points in local frame. Shape: (num_envs, num_points, 3)
        """
        x_prop, y_prop, z_shift = self.cfg.terrain.foothold_pts

        x_range = torch.linspace(x_prop[0], x_prop[1], x_prop[2], device=self.device)
        y_range = torch.linspace(y_prop[0], y_prop[1], y_prop[2], device=self.device)
        grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='xy')

        foothold_pts = torch.stack([
            grid_x.flatten(),
            grid_y.flatten(),
            torch.full_like(grid_x.flatten(), z_shift)
        ], dim=-1)

        return foothold_pts.unsqueeze(0).repeat(self.num_envs, 1, 1)

    def reset_idx(self, env_ids: Sequence[int]):
        """Reset specified environments.
        
        Args:
            env_ids: Environment indices to reset
        """
        super().reset_idx(env_ids)

        # Reset gait phase
        self.phase_length_buf[env_ids] = 0.
        self.gait_start[env_ids] = 0.5 * torch.randint(
            0, 2, (len(env_ids),), device=self.device
        )

        # Reset feet velocities
        self.last_feet_vel_xy[env_ids] = 0.

    def pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step.
        
        Args:
            actions: Action tensor
        """
        super().pre_physics_step(actions)

        # Update gait phase
        self.phase_length_buf += self.dt
        self._update_phase()

        # Update contact averaging
        if self.cfg.rewards.use_contact_averaging:
            alpha = self.cfg.rewards.contact_ema_alpha
            contact_forces = torch.norm(
                self.robot.data.body_force_w[:, self.feet_indices],
                dim=-1
            )
            self.contact_forces_avg[self.contact_filt] = (
                    alpha * self.contact_forces_avg[self.contact_filt] +
                    (1 - alpha) * contact_forces[self.contact_filt]
            )

            # Update feet air time average
            first_contact = (self.feet_air_time > 0.) & self.contact_filt
            self.feet_air_time_avg[first_contact] = (
                    alpha * self.feet_air_time_avg[first_contact] +
                    (1 - alpha) * self.feet_air_time[first_contact]
            )

    def post_physics_step(self):
        """Update state after physics step."""
        super().post_physics_step()

        # Update feet contact state
        self._update_contact_state()

        # Update feet measurements
        self._update_feet_state()

        # Update foothold detection (if terrain available)
        if hasattr(self, 'foothold_pts_local'):
            self._update_foothold_state()

    def _update_contact_state(self):
        """Update contact filtering and air time."""
        # Detect contact (force threshold)
        contact = torch.norm(
            self.robot.data.body_force_w[:, self.feet_indices],
            dim=-1
        ) > self.cfg.contact_force_threshold

        # Filter: contact OR was contact last step (to avoid flicker)
        self.contact_filt[:] = contact | self.last_contacts

        # Update air time
        self.feet_air_time += self.dt
        self.feet_air_time[self.contact_filt] = 0.

        # Store for next step
        self.last_contacts[:] = contact

    def _update_feet_state(self):
        """Update feet height and orientation measurements."""
        # Get feet positions
        feet_pos = self.robot.data.body_pos_w[:, self.feet_indices]

        # Query terrain height under feet
        if hasattr(self.scene, 'terrain') and self.scene.terrain is not None:
            terrain_heights = self._query_terrain_heights(feet_pos[:, :, :2])

            # Compute feet height above terrain
            self.feet_height[:] = feet_pos[:, :, 2] - terrain_heights
        else:
            # No terrain - use absolute height
            self.feet_height[:] = feet_pos[:, :, 2]

        # Get feet orientations
        feet_quat = self.robot.data.body_quat_w[:, self.feet_indices]
        self.feet_euler_xyz[:] = math_utils.quat_to_euler_xyz(feet_quat)

        # Store velocity for next step
        self.last_feet_vel_xy[:] = self.robot.data.body_lin_vel_w[:, self.feet_indices, :2]

    def _update_foothold_state(self):
        """Update foothold detection points and contact status."""
        num_feet = len(self.feet_indices)

        # Transform foothold points to world frame
        # Shape: (num_envs, num_feet, num_points, 3)
        feet_pos = self.robot.data.body_pos_w[:, self.feet_indices]
        feet_quat = self.robot.data.body_quat_w[:, self.feet_indices]

        # For each foot, transform its foothold points
        for foot_idx in range(num_feet):
            # Get this foot's pose
            pos = feet_pos[:, foot_idx]  # (num_envs, 3)
            quat = feet_quat[:, foot_idx]  # (num_envs, 4)

            # Transform foothold points
            # Flatten for batch operation
            points_flat = self.foothold_pts_local.reshape(-1, 3)
            quat_flat = quat.repeat_interleave(self.foothold_pts_local.size(1), dim=0)

            rotated_flat = math_utils.quat_rotate(points_flat, quat_flat)
            rotated = rotated_flat.reshape(self.num_envs, self.foothold_pts_local.size(1), 3)

            # Add foot position
            self.foothold_pts_pos[:, foot_idx] = rotated + pos.unsqueeze(1)

        # Query terrain heights at foothold points
        foothold_pos_flat = self.foothold_pts_pos.reshape(self.num_envs, -1, 3)
        terrain_heights = self._query_terrain_heights(foothold_pos_flat[:, :, :2])
        terrain_heights = terrain_heights.reshape(self.num_envs, num_feet, -1)

        # Check which points are in contact (height difference < threshold)
        foothold_heights = self.foothold_pts_pos[:, :, :, 2]
        height_diff = torch.abs(foothold_heights - terrain_heights)
        self.foothold_pts_contact[:] = height_diff < self.cfg.terrain.foothold_contact_thresh

        # Check if feet are at edge (if edge map available)
        if hasattr(self.scene.terrain, 'edge_map'):
            edge_map = torch.from_numpy(self.scene.terrain.edge_map).to(self.device)

            # Convert feet positions to terrain grid indices
            feet_pos_xy = feet_pos[:, :, :2]
            grid_indices = (feet_pos_xy / self.scene.terrain.cfg.horizontal_scale).long()
            grid_indices[:, :, 0] = torch.clamp(grid_indices[:, :, 0], 0, edge_map.shape[0] - 1)
            grid_indices[:, :, 1] = torch.clamp(grid_indices[:, :, 1], 0, edge_map.shape[1] - 1)

            # Query edge map
            feet_at_edge = edge_map[grid_indices[:, :, 0], grid_indices[:, :, 1]]
            self.feet_at_edge[:] = self.contact_filt & feet_at_edge

    def _query_terrain_heights(self, positions_xy: torch.Tensor) -> torch.Tensor:
        """Query terrain heights at XY positions using bilinear interpolation.
        
        Args:
            positions_xy: XY positions to query. Shape: (num_envs, num_points, 2)
            
        Returns:
            Heights at positions. Shape: (num_envs, num_points)
        """
        heightmap = torch.from_numpy(self.scene.terrain.height_map).to(self.device)
        horizontal_scale = self.scene.terrain.cfg.horizontal_scale

        # Convert to grid indices
        grid_pos = positions_xy / horizontal_scale
        px = grid_pos[:, :, 0].flatten()
        py = grid_pos[:, :, 1].flatten()

        # Clamp to valid range
        px = torch.clamp(px, 0, heightmap.shape[0] - 1.001)
        py = torch.clamp(py, 0, heightmap.shape[1] - 1.001)

        # Bilinear interpolation
        px_floor = torch.floor(px).long()
        py_floor = torch.floor(py).long()
        px_ceil = torch.clamp(px_floor + 1, 0, heightmap.shape[0] - 1)
        py_ceil = torch.clamp(py_floor + 1, 0, heightmap.shape[1] - 1)

        h00 = heightmap[px_floor, py_floor]
        h01 = heightmap[px_floor, py_ceil]
        h10 = heightmap[px_ceil, py_floor]
        h11 = heightmap[px_ceil, py_ceil]

        wx = (px - px_floor.float()).clamp(0, 1)
        wy = (py - py_floor.float()).clamp(0, 1)

        heights = (
                h00 * (1 - wx) * (1 - wy) +
                h01 * (1 - wx) * wy +
                h10 * wx * (1 - wy) +
                h11 * wx * wy
        )

        return heights.reshape(positions_xy.shape[0], positions_xy.shape[1])

    def _update_phase(self):
        """Update gait phase [0, 1] based on cycle time.
        
        Phase determines which leg should be in stance/swing.
        Phase wraps from 0 to 1 continuously during walking.
        """
        cycle_time = self.cfg.gait.cycle_time

        # Compute phase (wraps 0-1)
        self.phase[:] = ((self.phase_length_buf / cycle_time) + self.gait_start) % 1.0

    def _get_clock_input(self, wrap_sin: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Get phase clock inputs for left and right legs.
        
        Args:
            wrap_sin: If True, wrap with sin(2π*phase). If False, return raw phase.
            
        Returns:
            (clock_left, clock_right) - phase signals for each leg
        """
        clock_l = self.phase + self.cfg.gait.phase_offset_l
        clock_r = self.phase + self.cfg.gait.phase_offset_r

        if wrap_sin:
            return torch.sin(2 * torch.pi * clock_l), torch.sin(2 * torch.pi * clock_r)
        else:
            return clock_l % 1.0, clock_r % 1.0

    def _get_stance_mask(self) -> torch.Tensor:
        """Get binary mask indicating which feet should be in stance phase.
        
        Returns:
            Boolean mask. Shape: (num_envs, num_feet)
        """
        air_ratio = self.cfg.gait.air_ratio
        delta_t = self.cfg.gait.delta_t
        phase = torch.stack(self._get_clock_input(wrap_sin=False), dim=1)

        # Stance when phase is in [air_ratio + delta_t, 1 - delta_t]
        stance_mask = (phase >= air_ratio + delta_t) & (phase < (1. - delta_t))

        return stance_mask

    def _get_swing_mask(self) -> torch.Tensor:
        """Get binary mask indicating which feet should be in swing phase.
        
        Returns:
            Boolean mask. Shape: (num_envs, num_feet)
        """
        air_ratio = self.cfg.gait.air_ratio
        delta_t = self.cfg.gait.delta_t
        phase = torch.stack(self._get_clock_input(wrap_sin=False), dim=1)

        # Swing when phase is in [delta_t, air_ratio - delta_t]
        swing_mask = (phase >= delta_t) & (phase < (air_ratio - delta_t))

        return swing_mask

    def _get_soft_stance_mask(self) -> torch.Tensor:
        """Get soft (continuous) stance mask for smooth transitions.
        
        Returns continuous values [0, 1] instead of binary:
        - 0: Full swing
        - 1: Full stance
        - (0, 1): Transition
        
        Returns:
            Continuous stance values. Shape: (num_envs, num_feet)
        """
        phase = torch.stack(self._get_clock_input(wrap_sin=False), dim=1)
        air_ratio = 1. - self.cfg.gait.air_ratio
        delta_t = self.cfg.gait.delta_t

        # Five phases: trans1, swing, trans2, stance, trans3
        trans_flag1 = phase < delta_t
        swing_flag = (phase >= delta_t) & (phase < (air_ratio - delta_t))
        trans_flag2 = (phase >= (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
        stance_flag = (phase >= (air_ratio + delta_t)) & (phase < (1 - delta_t))
        trans_flag3 = phase >= (1 - delta_t)

        soft_stance_mask = (
                (0.5 - phase / (2 * delta_t)) * trans_flag1 +
                0.0 * swing_flag +
                (phase - air_ratio + delta_t) / (2 * delta_t) * trans_flag2 +
                1.0 * stance_flag +
                (1 + delta_t - phase) / (2 * delta_t) * trans_flag3
        )

        return soft_stance_mask

    # ========== Reward Functions ==========

    def _reward_feet_contact_number(self) -> torch.Tensor:
        """Reward proper gait phase (feet in air during swing, on ground during stance)."""
        swing = self._get_swing_mask()
        stance = self._get_stance_mask()
        contact = self.contact_filt

        rew = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)

        # Swing phase: penalize contact, reward no contact
        rew[swing] = torch.where(contact[swing], -0.3, 1.0)

        # Stance phase: reward contact, penalize no contact
        rew[stance] = torch.where(contact[stance], 1.0, -0.3)

        return torch.mean(rew, dim=1)

    def _reward_swing_phase(self) -> torch.Tensor:
        """Encourage no ground contact during swing phase."""
        contact_frc = torch.norm(
            self.robot.data.body_force_w[:, self.feet_indices],
            dim=2
        )
        rew = (1. - self._get_soft_stance_mask()) * torch.exp(-200 * torch.square(contact_frc))
        return torch.mean(rew, dim=1)

    def _reward_support_phase(self) -> torch.Tensor:
        """Encourage stationary feet during stance phase."""
        feet_speed = torch.norm(
            self.robot.data.body_lin_vel_w[:, self.feet_indices],
            dim=2
        )
        rew = self._get_soft_stance_mask() * torch.exp(-100 * torch.square(feet_speed))
        return torch.mean(rew, dim=1)

    def _reward_feet_clearance(self) -> torch.Tensor:
        """Encourage lifting feet to target height during swing."""
        target_height = self.cfg.rewards.feet_height_target

        # Reward based on how close to target height
        rew = (self.feet_height / target_height).clamp(min=-1, max=1)

        # Only apply during stance (feet should be on ground during stance)
        rew[self._get_stance_mask()] = 0.

        return rew.sum(dim=1)

    def _reward_feet_air_time(self) -> torch.Tensor:
        """Reward longer strides by encouraging feet air time."""
        first_contact = (self.feet_air_time > 0.) & self.contact_filt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        return air_time.sum(dim=1)

    def _reward_feet_slip(self) -> torch.Tensor:
        """Penalize foot slipping when in contact with ground."""
        feet_lin_vel = torch.norm(
            self.robot.data.body_lin_vel_w[:, self.feet_indices, :2],
            dim=2
        )
        feet_ang_vel = torch.abs(
            self.robot.data.body_ang_vel_w[:, self.feet_indices, 2]
        )

        # Only penalize slip when in contact
        rew = self.contact_filt * (feet_lin_vel + feet_ang_vel)
        return rew.sum(dim=1)

    def _reward_feet_distance(self) -> torch.Tensor:
        """Encourage proper feet distance (not too close, not too far)."""
        if len(self.feet_indices) != 2:
            return torch.zeros(self.num_envs, device=self.device)

        foot_pos = self.robot.data.body_pos_w[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0] - foot_pos[:, 1], dim=1)

        min_dist = self.cfg.rewards.min_feet_dist
        max_dist = self.cfg.rewards.max_feet_dist

        # Penalize too close
        d_min = torch.clamp(foot_dist - min_dist, -0.5, 0.)
        # Penalize too far
        d_max = torch.clamp(foot_dist - max_dist, 0, 0.5)

        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self) -> torch.Tensor:
        """Encourage proper knee distance."""
        if len(self.knee_indices) != 2:
            return torch.zeros(self.num_envs, device=self.device)

        knee_pos = self.robot.data.body_pos_w[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0] - knee_pos[:, 1], dim=1)

        min_dist = self.cfg.rewards.min_feet_dist
        max_dist = self.cfg.rewards.max_feet_dist / 2

        d_min = torch.clamp(knee_dist - min_dist, -0.5, 0.)
        d_max = torch.clamp(knee_dist - max_dist, 0, 0.5)

        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_feet_stumble(self) -> torch.Tensor:
        """Penalize feet hitting vertical surfaces (stumbling)."""
        horizontal_forces = torch.norm(
            self.robot.data.body_force_w[:, self.feet_indices, :2],
            dim=2
        )
        vertical_forces = torch.abs(
            self.robot.data.body_force_w[:, self.feet_indices, 2]
        )

        # Stumble = horizontal force > 5x vertical force
        stumble = horizontal_forces > 5 * vertical_forces

        return stumble.any(dim=1).float()

    def _reward_feet_rotation(self) -> torch.Tensor:
        """Encourage feet to stay flat (minimize roll/pitch)."""
        # Penalize non-zero roll and pitch
        rew = -torch.sum(self.feet_euler_xyz[..., :2].square(), dim=2)
        return torch.exp(torch.sum(rew, dim=1) * self.cfg.rewards.tracking_sigma)

    def _reward_feet_edge(self) -> torch.Tensor:
        """Penalize feet stepping on terrain edges."""
        if not hasattr(self, 'feet_at_edge'):
            return torch.zeros(self.num_envs, device=self.device)

        return self.feet_at_edge.float().sum(dim=-1)

    def _reward_foothold(self) -> torch.Tensor:
        """Penalize poor foothold quality (few contact points under foot)."""
        if not hasattr(self, 'foothold_pts_contact'):
            return torch.zeros(self.num_envs, device=self.device)

        # Percentage of foothold points in contact
        valid_foothold_perc = self.foothold_pts_contact.sum(dim=2) / self.foothold_pts_contact.size(2)

        # Penalize low contact percentage when foot is supposed to be in contact
        rew = (1 - valid_foothold_perc) * self.contact_filt

        return rew.sum(dim=1)
