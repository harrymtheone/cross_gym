"""Reward functions for locomotion tasks.

All reward functions follow the signature:
    func(env, **kwargs) -> torch.Tensor
    
where env is the environment instance and kwargs are optional parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .locomotion_env import LocomotionEnv


# ============================================================================
# Survival/Alive Rewards
# ============================================================================

def alive(env: LocomotionEnv) -> torch.Tensor:
    """Constant reward for staying alive.
    
    Args:
        env: Locomotion environment
        
    Returns:
        Reward (num_envs,)
    """
    return torch.ones(env.num_envs, device=env.device)


# ============================================================================
# Velocity Tracking Rewards
# ============================================================================

def lin_vel_x_tracking(env: LocomotionEnv, target_vel: float = 1.0) -> torch.Tensor:
    """Reward for tracking forward velocity.
    
    Args:
        env: Locomotion environment
        target_vel: Target forward velocity (m/s)
        
    Returns:
        Reward (num_envs,)
    """
    forward_vel = env.base_lin_vel[:, 0]
    return torch.exp(-torch.abs(forward_vel - target_vel))


def lin_vel_y_penalty(env: LocomotionEnv) -> torch.Tensor:
    """Penalty for lateral velocity.
    
    Args:
        env: Locomotion environment
        
    Returns:
        Penalty (num_envs,) - negative values
    """
    return -torch.abs(env.base_lin_vel[:, 1])


def lin_vel_z_penalty(env: LocomotionEnv) -> torch.Tensor:
    """Penalty for vertical velocity.
    
    Args:
        env: Locomotion environment
        
    Returns:
        Penalty (num_envs,) - negative values
    """
    return -torch.abs(env.base_lin_vel[:, 2])


def ang_vel_penalty(env: LocomotionEnv) -> torch.Tensor:
    """Penalty for angular velocity (all axes).
    
    Args:
        env: Locomotion environment
        
    Returns:
        Penalty (num_envs,) - negative values
    """
    return -torch.sum(torch.abs(env.base_ang_vel), dim=-1)


def ang_vel_z_penalty(env: LocomotionEnv) -> torch.Tensor:
    """Penalty for yaw rate (rotation around z).
    
    Args:
        env: Locomotion environment
        
    Returns:
        Penalty (num_envs,) - negative values
    """
    return -torch.abs(env.base_ang_vel[:, 2])


# ============================================================================
# Energy & Torque Penalties
# ============================================================================

def energy_penalty(env: LocomotionEnv) -> torch.Tensor:
    """Penalty for energy usage (torque squared).
    
    Args:
        env: Locomotion environment
        
    Returns:
        Penalty (num_envs,) - negative values
    """
    return -torch.sum(env.torques ** 2, dim=-1)


def torque_rate_penalty(env: LocomotionEnv) -> torch.Tensor:
    """Penalty for torque rate of change.
    
    Encourages smooth control.
    
    Args:
        env: Locomotion environment
        
    Returns:
        Penalty (num_envs,) - negative values
    """
    if not hasattr(env, 'last_torques'):
        env.last_torques = torch.zeros_like(env.torques)
    
    torque_diff = env.torques - env.last_torques
    env.last_torques = env.torques.clone()
    
    return -torch.sum(torque_diff ** 2, dim=-1)


# ============================================================================
# Stability & Posture Rewards
# ============================================================================

def upright_reward(env: LocomotionEnv) -> torch.Tensor:
    """Reward for staying upright.
    
    Uses gravity vector projected to base frame. When upright, z-component is -1.
    
    Args:
        env: Locomotion environment
        
    Returns:
        Reward (num_envs,)
    """
    return env.projected_gravity[:, 2]


def base_height_reward(env: LocomotionEnv, target_height: float = 0.6) -> torch.Tensor:
    """Reward for maintaining target base height.
    
    Args:
        env: Locomotion environment
        target_height: Target height above ground (meters)
        
    Returns:
        Reward (num_envs,)
    """
    base_height = env.robot.data.root_pos_w[:, 2]
    return -torch.abs(base_height - target_height)


# ============================================================================
# Joint Limits & Constraints
# ============================================================================

def dof_pos_limits_penalty(env: LocomotionEnv, margin: float = 0.1) -> torch.Tensor:
    """Penalty for joints near their position limits.
    
    Args:
        env: Locomotion environment
        margin: Safety margin from limits (radians)
        
    Returns:
        Penalty (num_envs,) - negative values
    """
    # Get soft DOF limits from robot
    # Assumes robot has soft_dof_pos_limits attribute
    if not hasattr(env.robot, 'soft_dof_pos_limits'):
        return torch.zeros(env.num_envs, device=env.device)
    
    dof_pos = env.robot.data.dof_pos
    lower_limits = env.robot.soft_dof_pos_limits[:, 0] + margin
    upper_limits = env.robot.soft_dof_pos_limits[:, 1] - margin
    
    # Penalty when outside safe zone
    lower_violation = torch.clamp(lower_limits - dof_pos, min=0.0)
    upper_violation = torch.clamp(dof_pos - upper_limits, min=0.0)
    
    return -torch.sum(lower_violation + upper_violation, dim=-1)


def dof_vel_limits_penalty(env: LocomotionEnv) -> torch.Tensor:
    """Penalty for joints exceeding velocity limits.
    
    Args:
        env: Locomotion environment
        
    Returns:
        Penalty (num_envs,) - negative values
    """
    if not hasattr(env.robot, 'soft_dof_vel_limits'):
        return torch.zeros(env.num_envs, device=env.device)
    
    dof_vel = env.robot.data.dof_vel
    vel_limits = env.robot.soft_dof_vel_limits
    
    # Penalty when exceeding limits
    violation = torch.clamp(torch.abs(dof_vel) - vel_limits, min=0.0)
    
    return -torch.sum(violation, dim=-1)


# ============================================================================
# Action Smoothness
# ============================================================================

def action_rate_penalty(env: LocomotionEnv) -> torch.Tensor:
    """Penalty for action rate of change.
    
    Encourages smooth policy outputs.
    
    Args:
        env: Locomotion environment
        
    Returns:
        Penalty (num_envs,) - negative values
    """
    if not hasattr(env, 'last_actions'):
        env.last_actions = torch.zeros_like(env.actions)
    
    action_diff = env.actions - env.last_actions
    env.last_actions = env.actions.clone()
    
    return -torch.sum(action_diff ** 2, dim=-1)

