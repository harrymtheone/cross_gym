"""Base class for actuator models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class ActuatorCommand:
    """Actuator command with position, velocity, and effort targets."""
    joint_positions: torch.Tensor | None = None
    joint_velocities: torch.Tensor | None = None
    joint_efforts: torch.Tensor | None = None


class ActuatorBase(ABC):
    """Base class for actuator models.
    
    Actuators convert user commands (pos/vel targets) into torques based on the actuator model.
    Parameters are resolved by merging URDF values with actuator config values.
    """

    def __init__(
            self,
            cfg,
            joint_names: list[str],
            joint_ids: torch.Tensor | slice,
            num_envs: int,
            device: torch.device,
            stiffness: torch.Tensor | float = 0.0,
            damping: torch.Tensor | float = 0.0,
            armature: torch.Tensor | float = 0.0,
            friction: torch.Tensor | float = 0.0,
            effort_limit: torch.Tensor | float = torch.inf,
            velocity_limit: torch.Tensor | float = torch.inf,
    ):
        """Initialize actuator with config and URDF parameters.
        
        Args:
            cfg: Actuator configuration
            joint_names: Names of joints in this actuator group
            joint_ids: Indices of joints (tensor or slice(None))
            num_envs: Number of environments
            device: Torch device
            stiffness: URDF stiffness values
            damping: URDF damping values
            armature: URDF armature values
            friction: URDF friction values
            effort_limit: URDF effort limit values
            velocity_limit: URDF velocity limit values
        """
        self.cfg = cfg
        self.joint_names = joint_names
        self.dof_indices = joint_ids
        self.num_envs = num_envs
        self.num_joints = len(joint_names)
        self.device = device

        # Parse parameters: merge URDF (default) with config values
        # Priority: cfg value > URDF value
        self.stiffness = self._parse_dof_parameter(cfg.stiffness, stiffness)
        self.damping = self._parse_dof_parameter(cfg.damping, damping)
        self.armature = self._parse_dof_parameter(cfg.armature, armature)
        self.friction = self._parse_dof_parameter(cfg.friction, friction)
        self.effort_limit = self._parse_dof_parameter(cfg.effort_limit, effort_limit)
        self.velocity_limit = self._parse_dof_parameter(cfg.velocity_limit, velocity_limit)

        # Initialize buffers for computed and applied torques
        self.computed_torque = torch.zeros(num_envs, self.num_joints, device=device)
        self.applied_torque = torch.zeros(num_envs, self.num_joints, device=device)

    def _parse_dof_parameter(
            self,
            cfg_value: float | dict[str, float] | None,
            default_value: torch.Tensor | float,
    ) -> torch.Tensor:
        """Parse DOF parameter from config or use default.
        
        Resolution priority:
        1. If cfg_value is specified, use it
        2. Otherwise, use default_value from URDF
        
        Args:
            cfg_value: Value from actuator config (can be float, dict, or None)
            default_value: Default value from URDF (float or tensor)
            
        Returns:
            Parsed parameter as tensor with shape (num_envs, num_joints)
        """
        param = torch.zeros(self.num_envs, self.num_joints, device=self.device)

        if cfg_value is None:
            # Use default from URDF
            if isinstance(default_value, (int, float)):
                param[:] = float(default_value)
            elif isinstance(default_value, torch.Tensor):
                if default_value.shape == (self.num_envs, self.num_joints):
                    param[:] = default_value
                else:
                    raise ValueError(
                        f"Default tensor shape mismatch: {default_value.shape} vs "
                        f"expected {(self.num_envs, self.num_joints)}"
                    )
            else:
                raise TypeError(f"Default value must be float or tensor, got {type(default_value)}")

        else:
            if isinstance(cfg_value, (int, float)):
                # Single value for all joints
                param[:] = float(cfg_value)
            elif isinstance(cfg_value, dict):
                # Dictionary with joint names as keys
                for joint_idx, joint_name in enumerate(self.joint_names):
                    if joint_name in cfg_value:
                        param[:, joint_idx] = float(cfg_value[joint_name])
                    else:
                        # Joint not in config dict, use default
                        if isinstance(default_value, torch.Tensor):
                            param[:, joint_idx] = default_value[:, joint_idx]
                        else:
                            param[:, joint_idx] = float(default_value)
            else:
                raise TypeError(f"Config value must be float or dict, got {type(cfg_value)}")

        return param

    @abstractmethod
    def compute(
            self,
            command: ActuatorCommand,
            joint_pos: torch.Tensor,
            joint_vel: torch.Tensor,
    ) -> ActuatorCommand:
        """Compute actuator output from commands.
        
        Args:
            command: Target positions, velocities, efforts
            joint_pos: Current joint positions (num_envs, num_joints)
            joint_vel: Current joint velocities (num_envs, num_joints)
            
        Returns:
            Modified command with computed efforts
        """
        pass

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset actuator state for given environments."""
        pass
