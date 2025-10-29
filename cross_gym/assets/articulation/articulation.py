"""Articulation asset implementation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence

import torch

from cross_gym.actuators import ActuatorCommand, ActuatorBase
from cross_gym.assets import AssetBase
from cross_gym.sim import ArticulationView, SimulationContext
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
        
        Queries simulator for the articulation view (physics representation).
        Scene must be built before calling this.
        
        Args:
            cfg: Configuration for the articulation
        """
        super().__init__(cfg)
        self.cfg: ArticulationCfg = cfg
        
        # Query simulator for view (scene must be built already)
        sim = SimulationContext.instance()
        self._backend = sim.get_articulation_view(cfg.prim_path)

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

    def find_bodies(self, name_patterns: str | list[str]) -> list[int]:
        """Find body indices matching regex patterns.
        
        Args:
            name_patterns: Single regex pattern or list of regex patterns
            
        Returns:
            List of body indices matching any of the patterns
            
        Example:
            >>> feet_indices = robot.find_bodies([".*_foot", ".*_ankle"])
            >>> knee_indices = robot.find_bodies(".*_knee.*")
        """
        if isinstance(name_patterns, str):
            name_patterns = [name_patterns]

        indices = []
        for pattern in name_patterns:
            for i, body_name in enumerate(self.body_names):
                if re.search(pattern, body_name) and i not in indices:
                    indices.append(i)

        return indices

    def find_joints(self, name_patterns: str | list[str]) -> list[int]:
        """Find joint indices matching regex patterns.
        
        Args:
            name_patterns: Single regex pattern or list of regex patterns
            
        Returns:
            List of joint indices matching any of the patterns
            
        Example:
            >>> leg_joints = robot.find_joints([".*_hip.*", ".*_knee.*"])
            >>> waist_joint = robot.find_joints("Waist")
        """
        if isinstance(name_patterns, str):
            name_patterns = [name_patterns]

        indices = []
        for pattern in name_patterns:
            for i, dof_name in enumerate(self.dof_names):
                if re.search(pattern, dof_name) and i not in indices:
                    indices.append(i)

        return indices

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

        # Parse configuration and initialize defaults
        self._parse_cfg()

        # Initialize simulation command buffers (sent to sim after actuator processing)
        self._dof_pos_target_sim = torch.zeros(num_envs, self.num_dof, device=self.device)
        self._dof_vel_target_sim = torch.zeros(num_envs, self.num_dof, device=self.device)
        self._dof_effort_target_sim = torch.zeros(num_envs, self.num_dof, device=self.device)

        # Process actuators (create actuator instances from config)
        self._process_actuators_cfg()

    def _parse_cfg(self):
        """Parse configuration and initialize default state.
        """
        # Initialize default root state from config
        self.data.default_root_pos = torch.tensor(
            self.cfg.init_state.pos, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.data.default_root_quat = torch.tensor(
            self.cfg.init_state.rot, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.data.default_root_lin_vel = torch.tensor(
            self.cfg.init_state.lin_vel, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.data.default_root_ang_vel = torch.tensor(
            self.cfg.init_state.ang_vel, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # Initialize default joint state to zeros
        self.data.default_joint_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.data.default_joint_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        # Apply default joint positions from pattern matching
        if self.cfg.init_state.joint_pos:
            for pattern, value in self.cfg.init_state.joint_pos.items():
                for i, dof_name in enumerate(self.dof_names):
                    if re.search(pattern, dof_name):
                        self.data.default_joint_pos[:, i] = value

        # Apply default joint velocities from pattern matching
        if self.cfg.init_state.joint_vel:
            for pattern, value in self.cfg.init_state.joint_vel.items():
                for i, dof_name in enumerate(self.dof_names):
                    if re.search(pattern, dof_name):
                        self.data.default_joint_vel[:, i] = value

    def _process_actuators_cfg(self):
        """Process actuator configurations and create actuator instances.
        
        Loads URDF parameters from backend and passes them to actuators
        for parameter resolution (config values override URDF values).
        """
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

            # Convert to tensor or slice for backend efficiency
            if len(joint_ids) == self.num_dof:
                joint_ids_tensor = slice(None)
            else:
                joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=self.device)

            # Load URDF parameters for these joints from backend
            # These are the "defaults" that will be passed to the actuator
            stiffness_urdf = torch.zeros(
                self.num_envs, len(joint_ids),
                device=self.device
            )  # TODO: Load from backend if available
            damping_urdf = torch.zeros(
                self.num_envs, len(joint_ids),
                device=self.device
            )  # TODO: Load from backend if available
            armature_urdf = torch.zeros(
                self.num_envs, len(joint_ids),
                device=self.device
            )  # TODO: Load from backend if available
            friction_urdf = torch.zeros(
                self.num_envs, len(joint_ids),
                device=self.device
            )  # TODO: Load from backend if available
            effort_limit_urdf = torch.full(
                (self.num_envs, len(joint_ids)), torch.inf,
                device=self.device
            )  # TODO: Load from backend if available
            velocity_limit_urdf = torch.full(
                (self.num_envs, len(joint_ids)), torch.inf,
                device=self.device
            )  # TODO: Load from backend if available

            # Create actuator instance with URDF parameters
            # Actuator will merge config values with URDF defaults
            actuator = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_ids=joint_ids_tensor,
                num_envs=self.num_envs,
                device=self.device,
                stiffness=stiffness_urdf,
                damping=damping_urdf,
                armature=armature_urdf,
                friction=friction_urdf,
                effort_limit=effort_limit_urdf,
                velocity_limit=velocity_limit_urdf,
            )

            # Add to actuators dict
            self.actuators[actuator_name] = actuator

            print(f"[Articulation] Actuator '{actuator_name}': {len(joint_ids)} joints")

    def reset(self, env_ids: Sequence[int] | None = None):
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = slice(None)

        # reset actuators
        for actuator in self.actuators.values():
            actuator.reset(env_ids)

        # TODO: reset external wrench
        # self._external_force_b[env_ids] = 0.0
        # self._external_torque_b[env_ids] = 0.0
        # self._external_wrench_positions_b[env_ids] = 0.0

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

    # ========== State Setters ==========

    def write_root_pos_to_sim(
            self,
            pos: torch.Tensor,
            env_ids: Sequence[int] | None = None
    ):
        """Write root position to simulation.
        
        Args:
            pos: Root positions (num_resets, 3).
            env_ids: Environment IDs. If None, apply to all.
        """
        if env_ids is None:
            env_ids = slice(None)

        # Update data buffer
        self.data.root_pos_w[env_ids] = pos

        # Write to simulation
        self._backend.set_root_state(root_pos=pos, env_ids=env_ids)

    def write_root_quat_to_sim(
            self,
            quat: torch.Tensor,
            env_ids: Sequence[int] | None = None
    ):
        """Write root orientation to simulation.
        
        Args:
            quat: Root orientations as quaternions (num_resets, 4) - (w, x, y, z).
            env_ids: Environment IDs. If None, apply to all.
        """
        if env_ids is None:
            env_ids = slice(None)

        # Update data buffer
        self.data.root_quat_w[env_ids] = quat

        # Write to simulation
        self._backend.set_root_state(root_quat=quat, env_ids=env_ids)

    def write_root_lin_vel_to_sim(
            self,
            lin_vel: torch.Tensor,
            env_ids: Sequence[int] | None = None
    ):
        """Write root linear velocity to simulation.
        
        Args:
            lin_vel: Root linear velocities (num_resets, 3).
            env_ids: Environment IDs. If None, apply to all.
        """
        if env_ids is None:
            env_ids = slice(None)

        # Update data buffer
        self.data.root_lin_vel_w[env_ids] = lin_vel

        # Write to simulation
        self._backend.set_root_state(root_lin_vel=lin_vel, env_ids=env_ids)

    def write_root_ang_vel_to_sim(
            self,
            ang_vel: torch.Tensor,
            env_ids: Sequence[int] | None = None
    ):
        """Write root angular velocity to simulation.
        
        Args:
            ang_vel: Root angular velocities (num_resets, 3).
            env_ids: Environment IDs. If None, apply to all.
        """
        if env_ids is None:
            env_ids = slice(None)

        # Update data buffer
        self.data.root_ang_vel_w[env_ids] = ang_vel

        # Write to simulation
        self._backend.set_root_state(root_ang_vel=ang_vel, env_ids=env_ids)

    def write_joint_pos_to_sim(
            self,
            joint_pos: torch.Tensor,
            env_ids: Sequence[int] | None = None
    ):
        """Write joint positions to simulation.
        
        Args:
            joint_pos: Joint positions (num_resets, num_dof).
            env_ids: Environment IDs. If None, apply to all.
        """
        if env_ids is None:
            env_ids = slice(None)

        # Update data buffer
        self.data.dof_pos[env_ids] = joint_pos

        # Write to simulation
        self._backend.set_joint_state(joint_pos=joint_pos, env_ids=env_ids)

    def write_joint_vel_to_sim(
            self,
            joint_vel: torch.Tensor,
            env_ids: Sequence[int] | None = None
    ):
        """Write joint velocities to simulation.
        
        Args:
            joint_vel: Joint velocities (num_resets, num_dof).
            env_ids: Environment IDs. If None, apply to all.
        """
        if env_ids is None:
            env_ids = slice(None)

        # Update data buffer
        self.data.dof_vel[env_ids] = joint_vel

        # Write to simulation
        self._backend.set_joint_state(joint_vel=joint_vel, env_ids=env_ids)

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
