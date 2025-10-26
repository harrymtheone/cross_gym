"""Articulation asset implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cross_gym.assets import AssetBase
from cross_gym.sim import ArticulationView
from . import ArticulationData

if TYPE_CHECKING:
    from . import ArticulationCfg


class Articulation(AssetBase):
    """Articulation asset (robot with joints).
    
    This class wraps an articulated body (robot) in the simulation, providing:
    - Easy access to robot state (joint positions, velocities, etc.)
    - Methods to set joint commands (torques, positions, velocities)
    - Automatic state updates from simulation
    """
    _backend: ArticulationView = None
    data: ArticulationData = None

    def __init__(self, cfg: ArticulationCfg):
        """Initialize articulation.
        
        Args:
            cfg: Configuration for the articulation
        """
        super().__init__(cfg)
        self.cfg: ArticulationCfg = cfg

    @property
    def num_instances(self) -> int:
        """Number of articulation instances.
        
        Returns:
            Number of instances (num_envs)
        """
        return self.num_envs

    @property
    def num_dof(self) -> int:
        """Number of degrees of freedom."""
        return self.data.num_dof

    @property
    def num_bodies(self) -> int:
        """Number of rigid bodies."""
        return self.data.num_bodies

    @property
    def dof_names(self) -> list[str]:
        """DOF names."""
        return self.data.dof_names

    @property
    def body_names(self) -> list[str]:
        """Body/link names."""
        return self.data.body_names

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

        # Initialize backend tensors
        self._backend.initialize_tensors()

        # Create data container (copies properties from backend)
        self.data = ArticulationData(self._backend, self.device)

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
        self._backend.set_root_state(root_pos, root_quat, root_lin_vel, root_ang_vel, env_ids)

        # Reset joint state to zeros (or can be configured)
        joint_pos = torch.zeros(num_resets, self.num_dof, device=self.device)
        joint_vel = torch.zeros(num_resets, self.num_dof, device=self.device)

        self._backend.set_joint_state(joint_pos, joint_vel, env_ids)

    def update(self, dt: float):
        """Update articulation state from simulation.
        
        Args:
            dt: Time step in seconds
        """
        # Data container handles everything via lazy properties
        self.data.update(dt)

    def write_data_to_sim(self):
        """Write articulation data to simulation.
        
        This writes buffered joint torques to the simulator.
        """
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
