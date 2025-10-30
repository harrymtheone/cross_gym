"""IsaacGym articulation - direct API access."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaacgym import gymtorch

from cross_core.articulations import ArticulationBase, ArticulationData

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

        # Common articulation data
        self._data = ArticulationData(
            device=device,
            num_envs=num_envs,
            num_dof=self.gym.get_actor_dof_count(None, actor_handles[0]),
            num_bodies=self.gym.get_actor_rigid_body_count(None, actor_handles[0])
        )

        # Initialize state tensors (will be populated on first access)
        self._dof_state_tensor = None
        self._root_state_tensor = None
        
        # Store actor indices for slicing global tensors
        self._actor_indices = torch.arange(num_envs, dtype=torch.int32, device=device)

    def _update_state_from_sim(self):
        """Update articulation data from simulation state.
        
        Acquires global state tensors from IsaacGym and slices them to populate
        this articulation's data fields.
        """
        # Acquire global DOF state tensor [num_total_dofs, 2] where [:, 0] = pos, [:, 1] = vel
        if self._dof_state_tensor is None:
            dof_state_raw = self.gym.acquire_dof_state_tensor(self.sim)
            self._dof_state_tensor = gymtorch.wrap_tensor(dof_state_raw)
        
        # Acquire global root state tensor [num_total_actors, 13]
        # Format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z,
        #          vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
        if self._root_state_tensor is None:
            root_state_raw = self.gym.acquire_actor_root_state_tensor(self.sim)
            self._root_state_tensor = gymtorch.wrap_tensor(root_state_raw)
        
        # Refresh tensors (in case simulation has stepped)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # Extract DOF states for this articulation
        # Note: In IsaacGym, DOFs are indexed per actor. For a single articulation type
        # across environments, DOFs are laid out sequentially: [env0_dofs, env1_dofs, ...]
        # Calculate DOF indices: each env has num_dof DOFs
        dof_start = 0  # Assuming this is the first/only articulation
        dof_end = self._data.num_envs * self._data.num_dof
        
        # Reshape to [num_envs, num_dof, 2] and extract position/velocity
        dof_states = self._dof_state_tensor[dof_start:dof_end].view(
            self._data.num_envs, self._data.num_dof, 2
        )
        
        # Populate DOF data
        self._data.dof_pos_target = dof_states[..., 0].clone()  # Current positions
        self._data.dof_vel_target = dof_states[..., 1].clone()  # Current velocities
        
        # Extract root states for this articulation
        # Assuming actors are indexed sequentially per environment
        root_states = self._root_state_tensor[self._actor_indices]
        
        # Populate root state data (full 13-element state)
        self._data.default_root_state = root_states.clone()

    @property
    def data(self) -> ArticulationData:
        """Get articulation data.
        
        Automatically updates from simulation before returning.
        
        Returns:
            ArticulationData: Container with all articulation state and properties.
        """
        self._update_state_from_sim()
        return self._data

    @property
    def num_dof(self) -> int:
        """Number of degrees of freedom."""
        return self._data.num_dof

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        return self._data.num_bodies

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._data.num_envs

    def get_joint_positions(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get joint positions using direct IsaacGym API."""
        # Get DOF state tensor (lazy init)
        if self._dof_state is None:
            self._dof_state = self.gym.acquire_dof_state_tensor(self.sim)
            gymtorch.wrap_tensor(self._dof_state)

        # Extract positions for this articulation
        # Note: This is a simplified version - full implementation would need
        # proper indexing into the global state tensor
        positions = torch.zeros(self._data.num_envs, self._data.num_dof, device=self._data.device)

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
        velocities = torch.zeros(self._data.num_envs, self._data.num_dof, device=self._data.device)
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
        root_state = torch.zeros(self._data.num_envs, 13, device=self._data.device)
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
