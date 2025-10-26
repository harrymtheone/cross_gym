"""Base class for actuator models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class ActuatorCommand:
    """Container for actuator commands.
    
    Holds the desired positions, velocities, and efforts.
    Only relevant fields need to be filled.
    """

    joint_positions: torch.Tensor | None = None
    """Target joint positions (num_envs, num_joints)."""

    joint_velocities: torch.Tensor | None = None
    """Target joint velocities (num_envs, num_joints)."""

    joint_efforts: torch.Tensor | None = None
    """Target joint efforts/torques (num_envs, num_joints)."""


class ActuatorBase(ABC):
    """Base class for actuator models.
    
    Actuators convert policy actions to joint torques, modeling:
    - PD controllers
    - Motor dynamics
    - Delays/lags
    - Saturation limits
    """

    def __init__(
            self,
            num_envs: int,
            num_joints: int,
            stiffness: torch.Tensor,
            damping: torch.Tensor,
            effort_limit: torch.Tensor,
            device: torch.device,
    ):
        """Initialize actuator.
        
        Args:
            num_envs: Number of environments
            num_joints: Number of joints controlled by this actuator
            stiffness: PD stiffness/kp (num_envs, num_joints)
            damping: PD damping/kd (num_envs, num_joints)
            effort_limit: Max torque (num_envs, num_joints)
            device: Torch device
        """
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = device

        # PD gains
        self.stiffness = stiffness
        self.damping = damping

        # Limits
        self.effort_limit = effort_limit

        # Torque buffers
        self.computed_torque = torch.zeros(num_envs, num_joints, device=device)
        self.applied_torque = torch.zeros(num_envs, num_joints, device=device)

    @abstractmethod
    def compute(
            self,
            command: ActuatorCommand,
            joint_pos: torch.Tensor,
            joint_vel: torch.Tensor,
    ) -> ActuatorCommand:
        """Compute torques from commands and current state.
        
        Args:
            command: Desired joint commands
            joint_pos: Current joint positions
            joint_vel: Current joint velocities
            
        Returns:
            Modified command with computed joint_efforts
        """
        pass

    @abstractmethod
    def reset(self, env_ids: torch.Tensor):
        """Reset actuator state for specified environments.
        
        Args:
            env_ids: Environment IDs to reset
        """
        pass

    def _clip_torque(self, torque: torch.Tensor) -> torch.Tensor:
        """Clip torque to effort limits.
        
        Args:
            torque: Unclipped torque
            
        Returns:
            Clipped torque
        """
        return torch.clip(torque, -self.effort_limit, self.effort_limit)
