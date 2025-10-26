"""Abstract base class for articulation views."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class ArticulationView(ABC):
    """Abstract base class for articulation views.
    
    Each simulator implements this interface to provide access to
    articulated body (robot) state and control.
    
    All methods use (w, x, y, z) quaternion format.
    """
    
    # Properties set by implementation
    num_dof: int
    """Number of degrees of freedom."""
    
    num_bodies: int
    """Number of rigid bodies/links."""
    
    _dof_names: list[str]
    """List of joint/DOF names."""
    
    _body_names: list[str]
    """List of body/link names."""
    
    @abstractmethod
    def initialize_tensors(self):
        """Initialize state tensors after simulator preparation."""
        pass
    
    @abstractmethod
    def update(self, dt: float):
        """Update state by reading from simulator.
        
        Args:
            dt: Time step
        """
        pass
    
    # ========== Root State ==========
    
    @abstractmethod
    def get_root_positions(self) -> torch.Tensor:
        """Get root link positions.
        
        Returns:
            Positions (num_envs, 3) in world frame
        """
        pass
    
    @abstractmethod
    def get_root_orientations(self) -> torch.Tensor:
        """Get root link orientations.
        
        Returns:
            Quaternions (num_envs, 4) in (w, x, y, z) format
        """
        pass
    
    @abstractmethod
    def get_root_velocities(self) -> torch.Tensor:
        """Get root link linear velocities.
        
        Returns:
            Velocities (num_envs, 3) in world frame
        """
        pass
    
    @abstractmethod
    def get_root_angular_velocities(self) -> torch.Tensor:
        """Get root link angular velocities.
        
        Returns:
            Angular velocities (num_envs, 3) in world frame
        """
        pass
    
    @abstractmethod
    def set_root_state(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None
    ):
        """Set root state for specified environments.
        
        Args:
            root_pos: Positions (num_resets, 3)
            root_quat: Orientations (num_resets, 4) in (w, x, y, z) format
            root_lin_vel: Linear velocities (num_resets, 3)
            root_ang_vel: Angular velocities (num_resets, 3)
            env_ids: Environment IDs to set (None = all)
        """
        pass
    
    # ========== Joint State ==========
    
    @abstractmethod
    def get_joint_positions(self) -> torch.Tensor:
        """Get joint positions.
        
        Returns:
            Joint positions (num_envs, num_dof)
        """
        pass
    
    @abstractmethod
    def get_joint_velocities(self) -> torch.Tensor:
        """Get joint velocities.
        
        Returns:
            Joint velocities (num_envs, num_dof)
        """
        pass
    
    @abstractmethod
    def set_joint_state(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None
    ):
        """Set joint state for specified environments.
        
        Args:
            joint_pos: Joint positions (num_resets, num_dof)
            joint_vel: Joint velocities (num_resets, num_dof)
            env_ids: Environment IDs to set (None = all)
        """
        pass
    
    # ========== Body State ==========
    
    @abstractmethod
    def get_body_positions(self) -> torch.Tensor:
        """Get all body positions.
        
        Returns:
            Positions (num_envs, num_bodies, 3) in world frame
        """
        pass
    
    @abstractmethod
    def get_body_orientations(self) -> torch.Tensor:
        """Get all body orientations.
        
        Returns:
            Quaternions (num_envs, num_bodies, 4) in (w, x, y, z) format
        """
        pass
    
    @abstractmethod
    def get_body_velocities(self) -> torch.Tensor:
        """Get all body linear velocities.
        
        Returns:
            Velocities (num_envs, num_bodies, 3) in world frame
        """
        pass
    
    @abstractmethod
    def get_body_angular_velocities(self) -> torch.Tensor:
        """Get all body angular velocities.
        
        Returns:
            Angular velocities (num_envs, num_bodies, 3) in world frame
        """
        pass
    
    # ========== Contact Forces ==========
    
    @abstractmethod
    def get_net_contact_forces(self) -> torch.Tensor:
        """Get net contact forces on all bodies.
        
        Returns:
            Contact forces (num_envs, num_bodies, 3)
        """
        pass
    
    # ========== Torque Control ==========
    
    @abstractmethod
    def set_joint_torques(self, torques: torch.Tensor):
        """Set joint torques.
        
        Args:
            torques: Joint torques (num_envs, num_dof)
        """
        pass


__all__ = ["ArticulationView"]

