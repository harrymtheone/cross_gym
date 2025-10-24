"""Articulation asset implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from cross_gym.assets.asset_base import AssetBase
from .articulation_data import ArticulationData

if TYPE_CHECKING:
    from . import ArticulationCfg


class Articulation(AssetBase):
    """Articulation asset (robot with joints).
    
    This class wraps an articulated body (robot) in the simulation, providing:
    - Easy access to robot state (joint positions, velocities, etc.)
    - Methods to set joint commands (torques, positions, velocities)
    - Automatic state updates from simulation
    """

    def __init__(self, cfg: ArticulationCfg):
        """Initialize articulation.
        
        Args:
            cfg: Configuration for the articulation
        """
        super().__init__(cfg)
        self.cfg: ArticulationCfg = cfg

        # Create backend view (simulator-specific)
        # This will be None until the scene creates environments
        self._backend = None

        # Articulation properties
        self.num_dof = 0
        self.num_bodies = 0
        self.dof_names: List[str] = []
        self.body_names: List[str] = []

        # Data container
        self.data = ArticulationData()

        # Joint limits
        self.dof_pos_limits: torch.Tensor | None = None  # (num_dof, 2) - [lower, upper]
        self.dof_vel_limits: torch.Tensor | None = None  # (num_dof,)
        self.dof_effort_limits: torch.Tensor | None = None  # (num_dof,)

    @property
    def num_instances(self) -> int:
        """Number of articulation instances.
        
        Returns:
            Number of instances (num_envs)
        """
        return self.num_envs

    def initialize(self, env_ids: torch.Tensor, num_envs: int):
        """Initialize articulation after environments are created.
        
        This is called by the scene after all environments are set up.
        
        Args:
            env_ids: Environment IDs
            num_envs: Total number of environments
        """
        self.num_envs = num_envs

        # Create simulator-specific backend view
        self._backend = self.sim.create_articulation_view(self.cfg.prim_path, num_envs)

        # Get properties from backend (if it's IsaacGym)
        if hasattr(self._backend, 'num_dof'):
            self.num_dof = self._backend.num_dof
            self.num_bodies = self._backend.num_bodies
            self.dof_names = self._backend._dof_names
            self.body_names = self._backend._body_names

        # Initialize data tensors
        self._initialize_data_tensors()

        # Initialize backend tensors (for IsaacGym)
        if hasattr(self._backend, 'initialize_tensors'):
            self._backend.initialize_tensors()

    def _initialize_data_tensors(self):
        """Initialize data container tensors with proper shapes."""
        device = self.device

        # Root state
        self.data.root_pos_w = torch.zeros(self.num_envs, 3, device=device)
        self.data.root_quat_w = torch.zeros(self.num_envs, 4, device=device)
        self.data.root_quat_w[:, 3] = 1.0  # Initialize to identity quaternion
        self.data.root_vel_w = torch.zeros(self.num_envs, 3, device=device)
        self.data.root_ang_vel_w = torch.zeros(self.num_envs, 3, device=device)

        # Joint state
        self.data.joint_pos = torch.zeros(self.num_envs, self.num_dof, device=device)
        self.data.joint_vel = torch.zeros(self.num_envs, self.num_dof, device=device)
        self.data.joint_acc = torch.zeros(self.num_envs, self.num_dof, device=device)

        # Body state
        self.data.body_pos_w = torch.zeros(self.num_envs, self.num_bodies, 3, device=device)
        self.data.body_quat_w = torch.zeros(self.num_envs, self.num_bodies, 4, device=device)
        self.data.body_quat_w[:, :, 3] = 1.0
        self.data.body_vel_w = torch.zeros(self.num_envs, self.num_bodies, 3, device=device)
        self.data.body_ang_vel_w = torch.zeros(self.num_envs, self.num_bodies, 3, device=device)

        # Contact forces
        self.data.net_contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=device)

        # Applied torques
        self.data.applied_torques = torch.zeros(self.num_envs, self.num_dof, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset articulation state for specified environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # Reset to initial state from config
        num_resets = len(env_ids)

        # Root state
        root_pos = torch.tensor(self.cfg.init_state.pos, device=self.device).repeat(num_resets, 1)
        root_quat = torch.tensor(self.cfg.init_state.rot, device=self.device).repeat(num_resets, 1)
        root_lin_vel = torch.tensor(self.cfg.init_state.lin_vel, device=self.device).repeat(num_resets, 1)
        root_ang_vel = torch.tensor(self.cfg.init_state.ang_vel, device=self.device).repeat(num_resets, 1)

        # Set in backend
        if hasattr(self._backend, 'set_root_state'):
            self._backend.set_root_state(root_pos, root_quat, root_lin_vel, root_ang_vel, env_ids)

        # Reset joint state to zeros (or can be configured)
        joint_pos = torch.zeros(num_resets, self.num_dof, device=self.device)
        joint_vel = torch.zeros(num_resets, self.num_dof, device=self.device)

        if hasattr(self._backend, 'set_joint_state'):
            self._backend.set_joint_state(joint_pos, joint_vel, env_ids)

    def update(self, dt: float):
        """Update articulation state from simulation.
        
        Args:
            dt: Time step in seconds
        """
        # Update backend (reads from simulator)
        if self._backend is not None:
            self._backend.update(dt)

            # Copy data from backend to data container
            if hasattr(self._backend, 'get_root_positions'):
                self.data.root_pos_w = self._backend.get_root_positions()
                self.data.root_quat_w = self._backend.get_root_orientations()
                self.data.root_vel_w = self._backend.get_root_velocities()
                self.data.root_ang_vel_w = self._backend.get_root_angular_velocities()

                self.data.joint_pos = self._backend.get_joint_positions()
                self.data.joint_vel = self._backend.get_joint_velocities()

                self.data.body_pos_w = self._backend.get_body_positions()
                self.data.body_quat_w = self._backend.get_body_orientations()
                self.data.body_vel_w = self._backend.get_body_velocities()
                self.data.body_ang_vel_w = self._backend.get_body_angular_velocities()

                self.data.net_contact_forces = self._backend.get_net_contact_forces()

    def write_data_to_sim(self):
        """Write articulation data to simulation.
        
        This writes buffered joint torques to the simulator.
        """
        if self._backend is not None and hasattr(self._backend, 'set_joint_torques'):
            self._backend.set_joint_torques(self.data.applied_torques)

    # ========== Convenience Methods ==========

    def set_joint_position_target(self, targets: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Set joint position targets (for position control mode).
        
        Args:
            targets: Target positions (num_envs, num_dof) or (num_resets, num_dof)
            env_ids: Environment IDs. If None, apply to all.
        """
        # This would be used with PD controllers / position mode
        # For now, just store (actual PD control would be in actuator module)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Placeholder - will be implemented with actuator module
        pass

    def set_joint_velocity_target(self, targets: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Set joint velocity targets (for velocity control mode).
        
        Args:
            targets: Target velocities (num_envs, num_dof) or (num_resets, num_dof)
            env_ids: Environment IDs. If None, apply to all.
        """
        # Placeholder - will be implemented with actuator module
        pass

    def set_joint_effort_target(self, efforts: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Set joint efforts/torques directly.
        
        Args:
            efforts: Desired torques (num_envs, num_dof) or (num_resets, num_dof)
            env_ids: Environment IDs. If None, apply to all.
        """
        if env_ids is None:
            self.data.applied_torques[:] = efforts
        else:
            self.data.applied_torques[env_ids] = efforts
