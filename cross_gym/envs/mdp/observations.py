"""Common observation functions.

These are functions that can be used as observation terms in ObservationManager.
Each function takes the environment as input and returns a tensor of observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


# ============================================================================
# Base Observations
# ============================================================================

def base_pos(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Base position in world frame.
    
    Args:
        env: Environment
        asset_name: Name of the articulation asset
        
    Returns:
        Base position (num_envs, 3)
    """
    asset = env.scene[asset_name]
    return asset.data.root_pos_w


def base_quat(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Base orientation as quaternion (w, x, y, z).
    
    Args:
        env: Environment
        asset_name: Name of the articulation asset
        
    Returns:
        Base quaternion (num_envs, 4) in (w, x, y, z) format
    """
    asset = env.scene[asset_name]
    return asset.data.root_quat_w


def base_lin_vel(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Base linear velocity in world frame.
    
    Args:
        env: Environment
        asset_name: Name of the articulation asset
        
    Returns:
        Linear velocity (num_envs, 3)
    """
    asset = env.scene[asset_name]
    return asset.data.root_vel_w


def base_ang_vel(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Base angular velocity in world frame.
    
    Args:
        env: Environment
        asset_name: Name of the articulation asset
        
    Returns:
        Angular velocity (num_envs, 3)
    """
    asset = env.scene[asset_name]
    return asset.data.root_ang_vel_w


# ============================================================================
# Joint Observations
# ============================================================================

def joint_pos(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Joint positions.
    
    Args:
        env: Environment
        asset_name: Name of the articulation asset
        
    Returns:
        Joint positions (num_envs, num_dof)
    """
    asset = env.scene[asset_name]
    return asset.data.joint_pos


def joint_vel(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Joint velocities.
    
    Args:
        env: Environment
        asset_name: Name of the articulation asset
        
    Returns:
        Joint velocities (num_envs, num_dof)
    """
    asset = env.scene[asset_name]
    return asset.data.joint_vel


def joint_pos_normalized(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Normalized joint positions (to [-1, 1] based on joint limits).
    
    Args:
        env: Environment
        asset_name: Name of the articulation asset
        
    Returns:
        Normalized joint positions (num_envs, num_dof)
    """
    asset = env.scene[asset_name]
    pos = asset.data.joint_pos

    if asset.dof_pos_limits is not None:
        # Normalize to [-1, 1]
        lower = asset.dof_pos_limits[:, 0]
        upper = asset.dof_pos_limits[:, 1]
        mid = (upper + lower) / 2.0
        range_half = (upper - lower) / 2.0
        return (pos - mid) / range_half
    else:
        return pos


# ============================================================================
# Body/Link Observations
# ============================================================================

def body_pos(env: ManagerBasedEnv, asset_name: str = "robot", body_names: list = None) -> torch.Tensor:
    """Body positions in world frame.
    
    Args:
        env: Environment
        asset_name: Name of the articulation asset
        body_names: List of body names. If None, return all bodies.
        
    Returns:
        Body positions (num_envs, num_bodies, 3) or (num_envs, len(body_names), 3)
    """
    asset = env.scene[asset_name]

    if body_names is None:
        return asset.data.body_pos_w
    else:
        # Get specific bodies
        body_ids = [i for i, name in enumerate(asset.body_names) if name in body_names]
        return asset.data.body_pos_w[:, body_ids, :]


# ============================================================================
# Episode/Time Observations
# ============================================================================

def episode_progress(env: ManagerBasedEnv) -> torch.Tensor:
    """Episode progress as fraction [0, 1].
    
    Args:
        env: Environment
        
    Returns:
        Progress (num_envs, 1)
    """
    progress = env.episode_length_buf.float() / env.max_episode_length
    return progress.unsqueeze(-1)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base
    "base_pos",
    "base_quat",
    "base_lin_vel",
    "base_ang_vel",
    # Joint
    "joint_pos",
    "joint_vel",
    "joint_pos_normalized",
    # Body
    "body_pos",
    # Episode
    "episode_progress",
]
