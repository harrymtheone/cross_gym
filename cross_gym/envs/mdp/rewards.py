"""Common reward functions.

These are functions that can be used as reward terms in RewardManager.
Each function takes the environment as input and returns a reward tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


# ============================================================================
# Survival/Alive Rewards
# ============================================================================

def alive_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Constant reward for staying alive.
    
    Args:
        env: Environment
        
    Returns:
        Reward (num_envs,)
    """
    return torch.ones(env.num_envs, device=env.device)


# ============================================================================
# Tracking Rewards
# ============================================================================

def lin_vel_tracking_reward(
        env: ManagerBasedEnv,
        asset_name: str = "robot",
        target_x: float = 1.0,
        target_y: float = 0.0,
) -> torch.Tensor:
    """Reward for tracking desired linear velocity.
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        target_x: Target forward velocity
        target_y: Target lateral velocity
        
    Returns:
        Reward (num_envs,)
    """
    asset = env.scene[asset_name]
    lin_vel = asset.data.root_vel_w

    # Error from target
    error_x = lin_vel[:, 0] - target_x
    error_y = lin_vel[:, 1] - target_y

    # Reward is negative squared error
    return -torch.sqrt(error_x ** 2 + error_y ** 2)


def ang_vel_tracking_reward(
        env: ManagerBasedEnv,
        asset_name: str = "robot",
        target_yaw_rate: float = 0.0,
) -> torch.Tensor:
    """Reward for tracking desired angular velocity (yaw rate).
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        target_yaw_rate: Target yaw rate (rotation around z-axis)
        
    Returns:
        Reward (num_envs,)
    """
    asset = env.scene[asset_name]
    ang_vel = asset.data.root_ang_vel_w

    # Error from target (z-component is yaw rate)
    error = ang_vel[:, 2] - target_yaw_rate

    # Reward is negative squared error
    return -(error ** 2)


# ============================================================================
# Energy/Torque Penalties
# ============================================================================

def energy_penalty(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Penalty for using energy (sum of squared torques).
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        
    Returns:
        Penalty (num_envs,)
    """
    asset = env.scene[asset_name]
    torques = asset.data.applied_torques
    return -torch.sum(torques ** 2, dim=-1)


def torque_penalty(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Penalty for high torques (L1 norm).
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        
    Returns:
        Penalty (num_envs,)
    """
    asset = env.scene[asset_name]
    torques = asset.data.applied_torques
    return -torch.sum(torch.abs(torques), dim=-1)


# ============================================================================
# Motion Quality Rewards
# ============================================================================

def upright_reward(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Reward for keeping base upright (z-axis pointing up).
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        
    Returns:
        Reward (num_envs,)
    """
    asset = env.scene[asset_name]
    quat = asset.data.root_quat_w  # (w, x, y, z)

    # Extract z-component of the up vector after rotation
    # For identity quaternion (1,0,0,0), up is (0,0,1)
    # After rotation by quat, up becomes quat_rotate(quat, (0,0,1))
    from cross_gym.utils.math import quat_rotate
    up_vec = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    rotated_up = quat_rotate(quat, up_vec)

    # Reward is z-component (how much up vector points in +z direction)
    return rotated_up[:, 2]


def height_reward(
        env: ManagerBasedEnv,
        asset_name: str = "robot",
        target_height: float = 0.5,
) -> torch.Tensor:
    """Reward for maintaining target height.
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        target_height: Desired base height
        
    Returns:
        Reward (num_envs,)
    """
    asset = env.scene[asset_name]
    height = asset.data.root_pos_w[:, 2]

    # Negative squared error from target
    return -(height - target_height) ** 2


# ============================================================================
# Smoothness/Regularization Rewards
# ============================================================================

def action_smoothness_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for smooth actions (penalize large changes).
    
    Args:
        env: Environment
        
    Returns:
        Reward (num_envs,)
    """
    # This requires storing previous actions
    # For now, return zeros as placeholder
    # TODO: Implement action history in ActionManager
    return torch.zeros(env.num_envs, device=env.device)


def joint_acc_penalty(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Penalty for high joint accelerations (smoothness).
    
    Args:
        env: Environment
        asset_name: Name of the articulation
        
    Returns:
        Penalty (num_envs,)
    """
    asset = env.scene[asset_name]

    if asset.data.joint_acc is not None:
        return -torch.sum(asset.data.joint_acc ** 2, dim=-1)
    else:
        # If no acceleration data, return zeros
        return torch.zeros(env.num_envs, device=env.device)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Survival
    "alive_reward",
    # Tracking
    "lin_vel_tracking_reward",
    "ang_vel_tracking_reward",
    # Energy
    "energy_penalty",
    "torque_penalty",
    # Motion quality
    "upright_reward",
    "height_reward",
    # Smoothness
    "action_smoothness_reward",
    "joint_acc_penalty",
]
