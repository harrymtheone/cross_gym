"""Data container for articulation state."""

from dataclasses import dataclass

import torch


@dataclass
class ArticulationData:
    """Data container for articulation state.
    
    This class stores all the state information for an articulation,
    organized in a way that's easy to access from the environment.
    """

    # Root state (base link)
    root_pos_w: torch.Tensor = None  # (num_envs, 3) - position in world frame
    root_quat_w: torch.Tensor = None  # (num_envs, 4) - orientation as quaternion (w, x, y, z)
    root_vel_w: torch.Tensor = None  # (num_envs, 3) - linear velocity in world frame
    root_ang_vel_w: torch.Tensor = None  # (num_envs, 3) - angular velocity in world frame

    # Joint state
    joint_pos: torch.Tensor = None  # (num_envs, num_dof) - joint positions
    joint_vel: torch.Tensor = None  # (num_envs, num_dof) - joint velocities
    joint_acc: torch.Tensor = None  # (num_envs, num_dof) - joint accelerations (if available)

    # Body state (all rigid bodies/links)
    body_pos_w: torch.Tensor = None  # (num_envs, num_bodies, 3) - positions in world frame
    body_quat_w: torch.Tensor = None  # (num_envs, num_bodies, 4) - orientations as quaternions (w, x, y, z)
    body_vel_w: torch.Tensor = None  # (num_envs, num_bodies, 3) - linear velocities
    body_ang_vel_w: torch.Tensor = None  # (num_envs, num_bodies, 3) - angular velocities

    # Contact forces
    net_contact_forces: torch.Tensor = None  # (num_envs, num_bodies, 3) - net contact forces

    # Applied joint torques (for logging/analysis)
    applied_torques: torch.Tensor = None  # (num_envs, num_dof) - commanded torques

    def __post_init__(self):
        """Initialize empty tensors if None."""
        # This will be called by the Articulation class with proper shapes
        pass
