"""Articulation asset implementation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from cross_gym.actuators import ActuatorCommand, ActuatorBase
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
    actuators: dict[str, ActuatorBase] = {}

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

        # Initialize simulation command buffers (sent to sim after actuator processing)
        self._dof_pos_target_sim = torch.zeros(num_envs, self.num_dof, device=self.device)
        self._dof_vel_target_sim = torch.zeros(num_envs, self.num_dof, device=self.device)
        self._dof_effort_target_sim = torch.zeros(num_envs, self.num_dof, device=self.device)

        # Process actuators (create actuator instances from config)
        self._process_actuators_cfg()

    def _process_actuators_cfg(self):
        """Process actuator configurations and create actuator instances."""
        if len(self.cfg.actuators) == 0:
            return

        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            # Find joints matching the patterns
            joint_names = []
            joint_ids = []

            for pattern in actuator_cfg.joint_names_expr:
                for i, dof_name in enumerate(self.dof_names):
                    if re.match(pattern, dof_name):
                        if i not in joint_ids:  # Avoid duplicates
                            joint_names.append(dof_name)
                            joint_ids.append(i)

            if len(joint_ids) == 0:
                print(f"[WARNING] Actuator '{actuator_name}' matched no joints")
                continue

            # Parse gains (can be single value or dict)
            stiffness = self._parse_dof_parameter(actuator_cfg.stiffness, joint_names)
            damping = self._parse_dof_parameter(actuator_cfg.damping, joint_names)

            # Parse effort limits
            if actuator_cfg.effort_limit is not None:
                # Use default (TODO: get from URDF)
                effort_limit = torch.full(
                    (self.num_envs, len(joint_ids)),
                    100.0,
                    device=self.device
                )
            else:
                effort_limit = self._parse_dof_parameter(actuator_cfg.effort_limit, joint_names)

            # Create actuator instance
            actuator = actuator_cfg.class_type(
                num_envs=self.num_envs,
                num_joints=len(joint_ids),
                stiffness=stiffness,
                damping=damping,
                effort_limit=effort_limit,
                device=self.device,
            )

            # Store DOF indices
            actuator.dof_indices = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
            actuator.dof_names = joint_names

            # Add to actuators dict
            self.actuators[actuator_name] = actuator

            print(f"[Articulation] Actuator '{actuator_name}': {len(joint_ids)} joints")

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
        
        Applies actuator models to compute torques from targets,
        then writes torques to simulator.
        """
        if len(self.actuators) > 0:
            # Apply actuator models
            self._apply_actuator_model()
        else:
            # No actuators, use user targets directly
            self._dof_effort_target_sim[:] = self.data.dof_effort_target

        self._backend.set_joint_torques(self._dof_effort_target_sim)

    def _apply_actuator_model(self):
        """Process DOF commands by forwarding them to actuators.
        
        The targets are first processed using actuator models. The actuator models
        compute the DOF-level simulation commands based on the user-specified targets.
        """
        # Process each actuator group
        for actuator in self.actuators.values():
            # Prepare input for actuator model from user targets
            control_action = ActuatorCommand(
                joint_positions=self.data.dof_pos_target[:, actuator.dof_indices],
                joint_velocities=self.data.dof_vel_target[:, actuator.dof_indices],
                joint_efforts=self.data.dof_effort_target[:, actuator.dof_indices],
            )

            # Compute DOF command from the actuator model
            control_action_output = actuator.compute(
                control_action,
                joint_pos=self.data.dof_pos[:, actuator.dof_indices],
                joint_vel=self.data.dof_vel[:, actuator.dof_indices],
            )

            # Update simulation targets (these are sent to simulation)
            if control_action_output.joint_positions is not None:
                self._dof_pos_target_sim[:, actuator.dof_indices] = control_action_output.joint_positions
            if control_action_output.joint_velocities is not None:
                self._dof_vel_target_sim[:, actuator.dof_indices] = control_action_output.joint_velocities
            if control_action_output.joint_efforts is not None:
                self._dof_effort_target_sim[:, actuator.dof_indices] = control_action_output.joint_efforts

            # Update state of the actuator model (for logging/inspection)
            self.data.computed_torque[:, actuator.dof_indices] = actuator.computed_torque
            self.data.applied_torque[:, actuator.dof_indices] = actuator.applied_torque

    def _parse_dof_parameter(
            self,
            param: float | dict[str, float],
            dof_names: list[str],
    ) -> torch.Tensor:
        """Parse DOF parameter that can be single value or dict.
        
        Args:
            param: Single value or dict {dof_pattern: value}
            dof_names: List of DOF names
            
        Returns:
            Parameter tensor (num_envs, num_dofs)
        """
        num_dofs = len(dof_names)
        param_tensor = torch.zeros(self.num_envs, num_dofs, device=self.device)

        if isinstance(param, (int, float)):
            # Single value for all DOFs
            param_tensor[:] = param

        elif isinstance(param, dict):
            # Different values per DOF pattern
            for i, dof_name in enumerate(dof_names):
                # Find matching pattern
                for pattern, value in param.items():
                    if re.match(pattern, dof_name):
                        param_tensor[:, i] = value
                        break
        else:
            raise ValueError(f"Invalid parameter type: {type(param)}")

        return param_tensor

    # ========== DOF Command Methods ==========

    def set_dof_position_target(self, targets: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Set DOF position targets.
        
        Stores targets which will be processed by actuators in write_data_to_sim().
        
        Args:
            targets: Target positions (num_envs, num_dof) or (num_resets, num_dof)
            env_ids: Environment IDs. If None, apply to all.
        """
        if env_ids is None:
            self.data.dof_pos_target[:] = targets
        else:
            self.data.dof_pos_target[env_ids] = targets

    def set_dof_velocity_target(self, targets: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Set DOF velocity targets.
        
        Stores targets which will be processed by actuators in write_data_to_sim().
        
        Args:
            targets: Target velocities (num_envs, num_dof) or (num_resets, num_dof)
            env_ids: Environment IDs. If None, apply to all.
        """
        if env_ids is None:
            self.data.dof_vel_target[:] = targets
        else:
            self.data.dof_vel_target[env_ids] = targets

    def set_dof_effort_target(self, efforts: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Set DOF effort/torque targets directly.
        
        Args:
            efforts: Desired torques (num_envs, num_dof) or (num_resets, num_dof)
            env_ids: Environment IDs. If None, apply to all.
        """
        if env_ids is None:
            self.data.dof_effort_target[:] = efforts
        else:
            self.data.dof_effort_target[env_ids] = efforts
