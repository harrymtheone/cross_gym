"""IsaacGym articulation - direct API access."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaacgym import gymtorch

from cross_core.base import ArticulationBase

if TYPE_CHECKING:
    from . import GymArticulationCfg


class GymArticulation(ArticulationBase):
    """IsaacGym articulation using direct API access.
    
    No wrapper layers - directly uses gym and sim handles.
    Implements ArticulationBase interface using IsaacGym API.
    """

    def __init__(
            self,
            cfg: GymArticulationCfg,
            actor_handles: list,
            gym,
            sim,
            device: torch.device,
            num_envs: int
    ):
        """Initialize articulation with direct IsaacGym handles.
        
        Args:
            cfg: Articulation configuration
            actor_handles: List of gym actor handles (one per env)
            gym: Gym instance
            sim: Sim handle
            device: Torch device
            num_envs: Number of environments
        """
        super().__init__(cfg)

        # Direct IsaacGym API access
        self.actor_handles = actor_handles
        self.gym = gym
        self.sim = sim

        # Set properties (template class uses these)
        self._device = device
        self._num_envs = num_envs
        self._num_dof = self.gym.get_actor_dof_count(None, actor_handles[0])
        self._num_bodies = self.gym.get_actor_rigid_body_count(None, actor_handles[0])

        # Initialize state tensors (will be populated on first access)
        self._dof_state = None
        self._root_state = None

    @property
    def num_dof(self) -> int:
        """Number of degrees of freedom."""
        return self._num_dof

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        return self._num_bodies

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    def get_joint_positions(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get joint positions using direct IsaacGym API."""
        # Get DOF state tensor (lazy init)
        if self._dof_state is None:
            self._dof_state = self.gym.acquire_dof_state_tensor(self.sim)
            gymtorch.wrap_tensor(self._dof_state)

        # Extract positions for this articulation
        # Note: This is a simplified version - full implementation would need
        # proper indexing into the global state tensor
        positions = torch.zeros(self._num_envs, self._num_dof, device=self._device)

        for i, actor in enumerate(self.actor_handles):
            # Direct API call per actor
            # TODO: Use tensor API for efficiency
            pass

        if env_ids is not None:
            return positions[env_ids]
        return positions

    def set_joint_position_targets(
            self,
            targets: torch.Tensor,
            env_ids: torch.Tensor | None = None
    ):
        """Set joint position targets using direct IsaacGym API."""
        # Direct API call
        # Implementation depends on control mode
        pass

    def get_joint_velocities(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get joint velocities using direct IsaacGym API."""
        velocities = torch.zeros(self._num_envs, self._num_dof, device=self._device)
        # Direct API implementation
        if env_ids is not None:
            return velocities[env_ids]
        return velocities

    def set_joint_velocity_targets(
            self,
            targets: torch.Tensor,
            env_ids: torch.Tensor | None = None
    ):
        """Set joint velocity targets using direct IsaacGym API."""
        pass

    def get_root_state(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get root body state using direct IsaacGym API."""
        root_state = torch.zeros(self._num_envs, 13, device=self._device)
        # Direct API implementation
        if env_ids is not None:
            return root_state[env_ids]
        return root_state

    def set_root_state(
            self,
            state: torch.Tensor,
            env_ids: torch.Tensor | None = None
    ):
        """Set root body state using direct IsaacGym API."""
        pass

    def apply_forces(
            self,
            forces: torch.Tensor,
            env_ids: torch.Tensor | None = None
    ):
        """Apply forces using direct IsaacGym API."""
        pass
