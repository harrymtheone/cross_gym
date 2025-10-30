"""Abstract base class for articulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch


class ArticulationConfigBase(ABC):
    """Base class for articulation configuration."""
    
    prim_path: str  # Path/pattern to articulation in scene
    file: str | None  # Path to URDF/asset file


class ArticulationBase(ABC):
    """Abstract base class for articulated robot.
    
    This defines a common interface for robot control across different simulators.
    Each backend implements these methods using their own API.
    """
    
    @abstractmethod
    def get_joint_positions(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get joint positions.
        
        Args:
            env_ids: Environment indices (None = all envs)
            
        Returns:
            Joint positions [num_envs, num_dof] or [len(env_ids), num_dof]
        """
        pass
    
    @abstractmethod
    def set_joint_position_targets(
        self,
        targets: torch.Tensor,
        env_ids: torch.Tensor | None = None
    ):
        """Set joint position targets for PD control.
        
        Args:
            targets: Target positions [num_envs, num_dof] or [len(env_ids), num_dof]
            env_ids: Environment indices (None = all envs)
        """
        pass
    
    @abstractmethod
    def get_joint_velocities(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get joint velocities.
        
        Args:
            env_ids: Environment indices (None = all envs)
            
        Returns:
            Joint velocities [num_envs, num_dof] or [len(env_ids), num_dof]
        """
        pass
    
    @abstractmethod
    def set_joint_velocity_targets(
        self,
        targets: torch.Tensor,
        env_ids: torch.Tensor | None = None
    ):
        """Set joint velocity targets for PD control.
        
        Args:
            targets: Target velocities [num_envs, num_dof] or [len(env_ids), num_dof]
            env_ids: Environment indices (None = all envs)
        """
        pass
    
    @abstractmethod
    def get_root_state(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get root body state (position, orientation, linear velocity, angular velocity).
        
        Args:
            env_ids: Environment indices (None = all envs)
            
        Returns:
            Root state [num_envs, 13] or [len(env_ids), 13]
            Format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z,
                     vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
        """
        pass
    
    @abstractmethod
    def set_root_state(
        self,
        state: torch.Tensor,
        env_ids: torch.Tensor | None = None
    ):
        """Set root body state.
        
        Args:
            state: Root state [num_envs, 13] or [len(env_ids), 13]
            env_ids: Environment indices (None = all envs)
        """
        pass
    
    @abstractmethod
    def apply_forces(
        self,
        forces: torch.Tensor,
        env_ids: torch.Tensor | None = None
    ):
        """Apply forces/torques to DOFs.
        
        Args:
            forces: Forces [num_envs, num_dof] or [len(env_ids), num_dof]
            env_ids: Environment indices (None = all envs)
        """
        pass
    
    @property
    @abstractmethod
    def num_dof(self) -> int:
        """Number of degrees of freedom."""
        pass
    
    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        pass
    
    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of parallel environments."""
        pass

