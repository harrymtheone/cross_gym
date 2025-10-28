"""Data container for articulation state."""

from __future__ import annotations

import torch

from cross_gym.sim import ArticulationView
from cross_gym.utils import math as math_utils
from cross_gym.utils.buffers import TimestampedBuffer


class ArticulationData:
    """Container for articulation (robot) state data.
    
    Handles all data reading from backend view with lazy evaluation
    and timestamped caching. Based on IsaacLab's ArticulationData.
    """

    # ========== Names ==========
    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    dof_names: list[str] = None
    """DOF names in the order parsed by the simulation view."""

    # ========== Dimensions ==========
    num_dof: int = None
    """Number of degrees of freedom."""

    num_bodies: int = None
    """Number of rigid bodies/links."""

    # ========== Default Spawn State ==========
    default_root_pos: torch.Tensor = None
    """Default root position for reset. Shape: (num_envs, 3)."""
    
    default_root_quat: torch.Tensor = None
    """Default root orientation for reset. Shape: (num_envs, 4)."""
    
    default_root_lin_vel: torch.Tensor = None
    """Default root linear velocity for reset. Shape: (num_envs, 3)."""
    
    default_root_ang_vel: torch.Tensor = None
    """Default root angular velocity for reset. Shape: (num_envs, 3)."""

    # ========== Default Joint State ==========
    default_joint_pos: torch.Tensor = None
    """Default joint positions from configuration. Shape: (num_envs, num_dof)."""

    default_joint_vel: torch.Tensor = None
    """Default joint velocities from configuration. Shape: (num_envs, num_dof)."""

    # ========== DOF Commands ==========
    dof_pos_target: torch.Tensor = None
    """DOF position targets commanded by user. Shape (num_envs, num_dof).
    
    For explicit actuators: Used to compute torques.
    For implicit actuators: Sent directly to simulation.
    """

    dof_vel_target: torch.Tensor = None
    """DOF velocity targets commanded by user. Shape (num_envs, num_dof)."""

    dof_effort_target: torch.Tensor = None
    """DOF effort targets commanded by user. Shape (num_envs, num_dof)."""

    # ========== Computed Torques (Set by Actuators) ==========
    computed_torque: torch.Tensor = None
    """DOF torques computed by actuators (before clipping). Shape (num_envs, num_dof).
    
    This is the raw torque output before any clipping is applied.
    """

    applied_torque: torch.Tensor = None
    """DOF torques applied to simulation (after clipping). Shape (num_envs, num_dof).
    
    These torques are set into the simulation after clipping based on effort limits.
    """

    def __init__(self, backend: ArticulationView, device: torch.device):
        """Initialize articulation data.
        
        Args:
            backend: Articulation view (simulator backend)
            device: Torch device
        """
        self.device = device
        self._backend: ArticulationView = backend
        self._sim_timestamp = 0.0

        # Copy names and dimensions from backend
        self.num_dof = backend.num_dof
        self.num_bodies = backend.num_bodies
        self.dof_names = backend.dof_names
        self.body_names = backend.body_names

        # Initialize gravity constant
        self.GRAVITY_VEC_W = torch.tensor(
            [0.0, 0.0, -1.0],
            device=device
        ).repeat(backend.num_envs, 1)

        # Initialize timestamped buffers
        # World frame (reads from backend)
        self._root_pos_w = TimestampedBuffer()
        self._root_quat_w = TimestampedBuffer()
        self._root_euler_w = TimestampedBuffer()
        self._root_vel_w = TimestampedBuffer()
        self._root_ang_vel_w = TimestampedBuffer()
        self._dof_pos = TimestampedBuffer()
        self._dof_vel = TimestampedBuffer()
        self._body_pos_w = TimestampedBuffer()
        self._body_quat_w = TimestampedBuffer()
        self._body_vel_w = TimestampedBuffer()
        self._body_ang_vel_w = TimestampedBuffer()
        self._net_contact_forces = TimestampedBuffer()
        # Base frame (transforms from world frame)
        self._root_lin_vel_b = TimestampedBuffer()
        self._root_ang_vel_b = TimestampedBuffer()
        self._projected_gravity_b = TimestampedBuffer()

    def update(self, dt: float):
        """Update simulation timestamp.
        
        Args:
            dt: Time step
        """
        self._backend.update(dt)
        self._sim_timestamp += dt

    # ========== Root State (World Frame) ==========

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in world frame. Shape (num_envs, 3)."""
        if self._root_pos_w.timestamp < self._sim_timestamp:
            self._root_pos_w.data = self._backend.get_root_pos_w()
            self._root_pos_w.timestamp = self._sim_timestamp
        return self._root_pos_w.data

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w,x,y,z) in world frame. Shape (num_envs, 4)."""
        if self._root_quat_w.timestamp < self._sim_timestamp:
            self._root_quat_w.data = self._backend.get_root_quat_w()
            self._root_quat_w.timestamp = self._sim_timestamp
        return self._root_quat_w.data

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root linear velocity in world frame. Shape (num_envs, 3)."""
        if self._root_vel_w.timestamp < self._sim_timestamp:
            self._root_vel_w.data = self._backend.get_root_lin_vel_w()
            self._root_vel_w.timestamp = self._sim_timestamp
        return self._root_vel_w.data

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in world frame. Shape (num_envs, 3)."""
        if self._root_ang_vel_w.timestamp < self._sim_timestamp:
            self._root_ang_vel_w.data = self._backend.get_root_ang_vel_w()
            self._root_ang_vel_w.timestamp = self._sim_timestamp
        return self._root_ang_vel_w.data

    @property
    def root_euler_w(self) -> torch.Tensor:
        """Root orientation as Euler angles (roll, pitch, yaw) in world frame. Shape (num_envs, 3)."""
        if self._root_euler_w.timestamp < self._sim_timestamp:
            self._root_euler_w.data = math_utils.quat_to_euler_xyz(self.root_quat_w)
            self._root_euler_w.timestamp = self._sim_timestamp
        return self._root_euler_w.data

    # ========== Root State (Base Frame) ==========

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape (num_envs, 3)."""
        if self._root_lin_vel_b.timestamp < self._sim_timestamp:
            quat_inv = math_utils.quat_conjugate(self.root_quat_w)
            self._root_lin_vel_b.data = math_utils.quat_rotate(quat_inv, self.root_vel_w)
            self._root_lin_vel_b.timestamp = self._sim_timestamp
        return self._root_lin_vel_b.data

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base frame. Shape (num_envs, 3)."""
        if self._root_ang_vel_b.timestamp < self._sim_timestamp:
            quat_inv = math_utils.quat_conjugate(self.root_quat_w)
            self._root_ang_vel_b.data = math_utils.quat_rotate(quat_inv, self.root_ang_vel_w)
            self._root_ang_vel_b.timestamp = self._sim_timestamp
        return self._root_ang_vel_b.data

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Gravity vector in base frame. Shape (num_envs, 3)."""
        if self._projected_gravity_b.timestamp < self._sim_timestamp:
            quat_inv = math_utils.quat_conjugate(self.root_quat_w)
            self._projected_gravity_b.data = math_utils.quat_rotate(quat_inv, self.GRAVITY_VEC_W)
            self._projected_gravity_b.timestamp = self._sim_timestamp
        return self._projected_gravity_b.data

    # ========== DOF State ==========

    @property
    def dof_pos(self) -> torch.Tensor:
        """DOF positions. Shape (num_envs, num_dof)."""
        if self._dof_pos.timestamp < self._sim_timestamp:
            self._dof_pos.data = self._backend.get_joint_pos()
            self._dof_pos.timestamp = self._sim_timestamp
        return self._dof_pos.data

    @property
    def dof_vel(self) -> torch.Tensor:
        """DOF velocities. Shape (num_envs, num_dof)."""
        if self._dof_vel.timestamp < self._sim_timestamp:
            self._dof_vel.data = self._backend.get_joint_vel()
            self._dof_vel.timestamp = self._sim_timestamp
        return self._dof_vel.data

    @property
    def dof_acc(self) -> torch.Tensor:
        """DOF accelerations (if available). Shape (num_envs, num_dof).
        
        Note: Not all simulators provide acceleration directly.
        """
        # TODO: Implement finite differencing for acceleration
        if hasattr(self._backend, 'get_dof_accelerations'):
            return self._backend.get_dof_accelerations()
        else:
            # Return zeros if not available
            return torch.zeros_like(self.dof_vel)

    # ========== Body State (World Frame) ==========

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Body positions in world frame. Shape (num_envs, num_bodies, 3)."""
        if self._body_pos_w.timestamp < self._sim_timestamp:
            self._body_pos_w.data = self._backend.get_body_pos_w()
            self._body_pos_w.timestamp = self._sim_timestamp
        return self._body_pos_w.data

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Body orientations (w,x,y,z) in world frame. Shape (num_envs, num_bodies, 4)."""
        if self._body_quat_w.timestamp < self._sim_timestamp:
            self._body_quat_w.data = self._backend.get_body_quat_w()
            self._body_quat_w.timestamp = self._sim_timestamp
        return self._body_quat_w.data

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Body linear velocities in world frame. Shape (num_envs, num_bodies, 3)."""
        if self._body_vel_w.timestamp < self._sim_timestamp:
            self._body_vel_w.data = self._backend.get_body_lin_vel_w()
            self._body_vel_w.timestamp = self._sim_timestamp
        return self._body_vel_w.data

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Body angular velocities in world frame. Shape (num_envs, num_bodies, 3)."""
        if self._body_ang_vel_w.timestamp < self._sim_timestamp:
            self._body_ang_vel_w.data = self._backend.get_body_ang_vel_w()
            self._body_ang_vel_w.timestamp = self._sim_timestamp
        return self._body_ang_vel_w.data

    # ========== Contact Forces ==========

    @property
    def net_contact_forces(self) -> torch.Tensor:
        """Net contact forces on bodies. Shape (num_envs, num_bodies, 3)."""
        if self._net_contact_forces.timestamp < self._sim_timestamp:
            self._net_contact_forces.data = self._backend.get_net_contact_forces()
            self._net_contact_forces.timestamp = self._sim_timestamp
        return self._net_contact_forces.data

    # ========== Applied Torques ==========

    @property
    def applied_torques(self) -> torch.Tensor:
        """Applied joint torques. Shape (num_envs, num_dof).
        
        Note: This is a writable property. This is used to set the joint torques.
        """
        if not hasattr(self, '_applied_torques'):
            self._applied_torques = torch.zeros(
                self._backend.num_envs,
                self._backend.num_dof,
                device=self.device
            )
        return self._applied_torques

    @applied_torques.setter
    def applied_torques(self, value: torch.Tensor):
        """Set applied torques."""
        if not hasattr(self, '_applied_torques'):
            self._applied_torques = value
        else:
            self._applied_torques[:] = value
