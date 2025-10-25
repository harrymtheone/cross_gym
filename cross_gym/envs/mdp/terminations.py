"""Common termination functions.

These are functions that can be used as termination terms in TerminationManager.
Each function takes the environment as input and returns a boolean tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


# ============================================================================
# Time-based Terminations
# ============================================================================

def time_out(env: ManagerBasedEnv) -> torch.Tensor:
    """Episode timeout termination.
    
    Args:
        env: Environment
        
    Returns:
        Boolean tensor (num_envs,) indicating timeout
    """
    return env.episode_length_buf >= env.max_episode_length


# ============================================================================
# Height-based Terminations
# ============================================================================

def base_height_termination(
        env: ManagerBasedEnv,
        asset_name: str = "robot",
        min_height: float = 0.3,
) -> torch.Tensor:
    """Terminate if base falls below minimum height.
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        min_height: Minimum allowed base height
        
    Returns:
        Boolean tensor (num_envs,)
    """
    asset = env.scene[asset_name]
    base_height = asset.data.root_pos_w[:, 2]
    return base_height < min_height


def base_height_range_termination(
        env: ManagerBasedEnv,
        asset_name: str = "robot",
        min_height: float = 0.2,
        max_height: float = 2.0,
) -> torch.Tensor:
    """Terminate if base goes outside height range.
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        min_height: Minimum allowed height
        max_height: Maximum allowed height
        
    Returns:
        Boolean tensor (num_envs,)
    """
    asset = env.scene[asset_name]
    base_height = asset.data.root_pos_w[:, 2]
    return (base_height < min_height) | (base_height > max_height)


# ============================================================================
# Orientation-based Terminations
# ============================================================================

def base_tilt_termination(
        env: ManagerBasedEnv,
        asset_name: str = "robot",
        max_tilt_angle: float = 1.57,  # ~90 degrees in radians
) -> torch.Tensor:
    """Terminate if base tilts beyond threshold.
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        max_tilt_angle: Maximum tilt angle in radians
        
    Returns:
        Boolean tensor (num_envs,)
    """
    asset = env.scene[asset_name]
    quat = asset.data.root_quat_w  # (w, x, y, z)

    # Get z-component of rotated up vector
    from cross_gym.utils.math import quat_rotate
    up_vec = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    rotated_up = quat_rotate(quat, up_vec)

    # If z-component < cos(max_tilt_angle), we've tilted too much
    return rotated_up[:, 2] < torch.cos(torch.tensor(max_tilt_angle, device=env.device))


# ============================================================================
# Contact-based Terminations
# ============================================================================

def base_contact_termination(
        env: ManagerBasedEnv,
        asset_name: str = "robot",
        threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if base link has contact with ground.
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        threshold: Contact force threshold
        
    Returns:
        Boolean tensor (num_envs,)
    """
    asset = env.scene[asset_name]
    contact_forces = asset.data.net_contact_forces

    # Base is usually the first body (index 0)
    base_contact_force = torch.norm(contact_forces[:, 0, :], dim=-1)

    return base_contact_force > threshold


def illegal_contact_termination(
        env: ManagerBasedEnv,
        asset_name: str = "robot",
        body_names: list = ["base"],
        threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if specified bodies have contact.
    
    Useful for terminating when "illegal" body parts touch the ground
    (e.g., robot's torso or knees).
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        body_names: List of body names that shouldn't have contact
        threshold: Contact force threshold
        
    Returns:
        Boolean tensor (num_envs,)
    """
    asset = env.scene[asset_name]
    contact_forces = asset.data.net_contact_forces

    # Get indices of illegal bodies
    body_ids = [i for i, name in enumerate(asset.body_names) if name in body_names]

    # Check if any illegal body has contact
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for body_id in body_ids:
        contact_force = torch.norm(contact_forces[:, body_id, :], dim=-1)
        terminated |= (contact_force > threshold)

    return terminated


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Time
    "time_out",
    # Height
    "base_height_termination",
    "base_height_range_termination",
    # Orientation
    "base_tilt_termination",
    # Contact
    "base_contact_termination",
    "illegal_contact_termination",
]
