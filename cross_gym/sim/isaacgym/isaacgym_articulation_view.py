"""IsaacGym articulation view - wraps articulated bodies (robots)."""

import torch
from typing import Tuple
from isaacgym import gymapi, gymtorch


class IsaacGymArticulationView:
    """View for articulated bodies in IsaacGym.
    
    This class wraps IsaacGym's tensor API for articulations,
    providing a clean interface for reading and writing robot state.
    """
    
    def __init__(self, gym, sim, prim_path: str, num_envs: int, device: torch.device):
        """Initialize articulation view.
        
        Args:
            gym: IsaacGym gym instance
            sim: IsaacGym sim instance
            prim_path: Path pattern to the articulation (not used in IsaacGym directly)
            num_envs: Number of parallel environments
            device: Torch device
        """
        self.gym = gym
        self.sim = sim
        self.prim_path = prim_path
        self.num_envs = num_envs
        self.device = device
        
        # These will be set when articulation is added to envs
        self.num_dof = 0
        self.num_bodies = 0
        self._dof_names = []
        self._body_names = []
        
        # Tensors (will be initialized after prepare_sim)
        self._root_state = None
        self._dof_state = None
        self._rigid_body_states = None
        self._contact_forces = None
        
        # For torque control
        self._torques = None
    
    def initialize_tensors(self):
        """Initialize state tensors after sim.prepare_sim() is called."""
        # Refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        # Wrap tensors
        self._root_state = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        )
        self._dof_state = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        )
        self._rigid_body_states = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        ).view(self.num_envs, -1, 13)
        self._contact_forces = gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim)
        ).view(self.num_envs, -1, 3)
        
        # Initialize torque buffer
        self._torques = torch.zeros(
            self.num_envs, self.num_dof,
            dtype=torch.float32,
            device=self.device
        )
    
    def update(self, dt: float):
        """Update the articulation state by reading from simulator.
        
        Args:
            dt: Time step (not used in IsaacGym, but kept for interface compatibility)
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
    
    # ========== Root State (Base Link) ==========
    
    def get_root_positions(self) -> torch.Tensor:
        """Get root link positions in world frame.
        
        Returns:
            Tensor of shape (num_envs, 3)
        """
        return self._root_state[:, 0:3]
    
    def get_root_orientations(self) -> torch.Tensor:
        """Get root link orientations as quaternions (x, y, z, w).
        
        Returns:
            Tensor of shape (num_envs, 4)
        """
        return self._root_state[:, 3:7]
    
    def get_root_velocities(self) -> torch.Tensor:
        """Get root link linear velocities.
        
        Returns:
            Tensor of shape (num_envs, 3)
        """
        return self._root_state[:, 7:10]
    
    def get_root_angular_velocities(self) -> torch.Tensor:
        """Get root link angular velocities.
        
        Returns:
            Tensor of shape (num_envs, 3)
        """
        return self._root_state[:, 10:13]
    
    def set_root_state(self, root_pos: torch.Tensor, root_quat: torch.Tensor,
                       root_lin_vel: torch.Tensor, root_ang_vel: torch.Tensor,
                       env_ids: torch.Tensor | None = None):
        """Set root link state for specified environments.
        
        Args:
            root_pos: Root positions (num_envs, 3)
            root_quat: Root orientations as quaternions (num_envs, 4) 
            root_lin_vel: Root linear velocities (num_envs, 3)
            root_ang_vel: Root angular velocities (num_envs, 3)
            env_ids: Environment indices to set (None = all)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self._root_state[env_ids, 0:3] = root_pos
        self._root_state[env_ids, 3:7] = root_quat
        self._root_state[env_ids, 7:10] = root_lin_vel
        self._root_state[env_ids, 10:13] = root_ang_vel
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
    
    # ========== Joint State (DOFs) ==========
    
    def get_joint_positions(self) -> torch.Tensor:
        """Get joint positions.
        
        Returns:
            Tensor of shape (num_envs, num_dof)
        """
        return self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
    
    def get_joint_velocities(self) -> torch.Tensor:
        """Get joint velocities.
        
        Returns:
            Tensor of shape (num_envs, num_dof)
        """
        return self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
    
    def set_joint_state(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor,
                        env_ids: torch.Tensor | None = None):
        """Set joint state for specified environments.
        
        Args:
            joint_pos: Joint positions (num_envs, num_dof)
            joint_vel: Joint velocities (num_envs, num_dof)
            env_ids: Environment indices to set (None = all)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        dof_state_view = self._dof_state.view(self.num_envs, self.num_dof, 2)
        dof_state_view[env_ids, :, 0] = joint_pos
        dof_state_view[env_ids, :, 1] = joint_vel
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
    
    # ========== Body State (Links) ==========
    
    def get_body_positions(self) -> torch.Tensor:
        """Get all body positions.
        
        Returns:
            Tensor of shape (num_envs, num_bodies, 3)
        """
        return self._rigid_body_states[..., 0:3]
    
    def get_body_orientations(self) -> torch.Tensor:
        """Get all body orientations as quaternions.
        
        Returns:
            Tensor of shape (num_envs, num_bodies, 4)
        """
        return self._rigid_body_states[..., 3:7]
    
    def get_body_velocities(self) -> torch.Tensor:
        """Get all body linear velocities.
        
        Returns:
            Tensor of shape (num_envs, num_bodies, 3)
        """
        return self._rigid_body_states[..., 7:10]
    
    def get_body_angular_velocities(self) -> torch.Tensor:
        """Get all body angular velocities.
        
        Returns:
            Tensor of shape (num_envs, num_bodies, 3)
        """
        return self._rigid_body_states[..., 10:13]
    
    # ========== Contact Forces ==========
    
    def get_net_contact_forces(self) -> torch.Tensor:
        """Get net contact forces on all bodies.
        
        Returns:
            Tensor of shape (num_envs, num_bodies, 3)
        """
        return self._contact_forces
    
    # ========== Actuation ==========
    
    def set_joint_torques(self, torques: torch.Tensor):
        """Set joint torques for actuation.
        
        Args:
            torques: Desired torques (num_envs, num_dof)
        """
        self._torques[:] = torques
        self.gym.set_dof_actuation_force_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self._torques.view(-1))
        )

