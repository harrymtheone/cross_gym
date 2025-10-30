"""Articulation data container for common attributes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ArticulationData:
    """Data container for common articulation attributes.
    
    This holds simulator-agnostic data that all articulation implementations
    need, such as device, number of environments, DOFs, bodies, states, and properties.
    """
    
    # ========== Basic Properties ==========
    
    device: torch.device | None = None
    """Torch device for computation."""
    
    num_envs: int = 0
    """Number of parallel environments."""
    
    num_dof: int = 0
    """Number of degrees of freedom."""
    
    num_bodies: int = 0
    """Number of bodies in articulation."""
    
    # ========== Names ==========
    
    body_names: list[str] | None = None
    """Body names in the order parsed by the simulation view."""
    
    dof_names: list[str] | None = None
    """DOF names in the order parsed by the simulation view."""
    
    # ========== Default State ==========
    
    default_root_state: torch.Tensor | None = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in the local environment frame. 
    Shape is (num_instances, 13).
    
    The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
    velocities are of its center of mass frame.
    """
    
    default_dof_pos: torch.Tensor | None = None
    """Default DOF positions of all DOFs. Shape is (num_instances, num_dof)."""
    
    default_dof_vel: torch.Tensor | None = None
    """Default DOF velocities of all DOFs. Shape is (num_instances, num_dof)."""
    
    # ========== Default Physical Properties ==========
    
    default_mass: torch.Tensor | None = None
    """Default mass for all the bodies in the articulation. Shape is (num_instances, num_bodies)."""
    
    default_inertia: torch.Tensor | None = None
    """Default inertia for all the bodies in the articulation. Shape is (num_instances, num_bodies, 9).
    
    The inertia tensor should be given with respect to the center of mass, expressed in the articulation links' actor frame.
    The values are stored in the order :math:`[I_{xx}, I_{yx}, I_{zx}, I_{xy}, I_{yy}, I_{zy}, I_{xz}, I_{yz}, I_{zz}]`.
    """
    
    default_dof_stiffness: torch.Tensor | None = None
    """Default DOF stiffness of all DOFs. Shape is (num_instances, num_dof).
    
    This quantity is configured by the user or parsed from the asset. It should not be confused with
    :attr:`dof_stiffness`, which is the value set into the simulation.
    """
    
    default_dof_damping: torch.Tensor | None = None
    """Default DOF damping of all DOFs. Shape is (num_instances, num_dof).
    
    This quantity is configured by the user or parsed from the asset. It should not be confused with
    :attr:`dof_damping`, which is the value set into the simulation.
    """
    
    default_dof_armature: torch.Tensor | None = None
    """Default DOF armature of all DOFs. Shape is (num_instances, num_dof)."""
    
    default_dof_friction_coeff: torch.Tensor | None = None
    """Default DOF static friction coefficient of all DOFs. Shape is (num_instances, num_dof)."""
    
    default_dof_dynamic_friction_coeff: torch.Tensor | None = None
    """Default DOF dynamic friction coefficient of all DOFs. Shape is (num_instances, num_dof)."""
    
    default_dof_viscous_friction_coeff: torch.Tensor | None = None
    """Default DOF viscous friction coefficient of all DOFs. Shape is (num_instances, num_dof)."""
    
    default_dof_pos_limits: torch.Tensor | None = None
    """Default DOF position limits of all DOFs. Shape is (num_instances, num_dof, 2).
    
    The limits are in the order :math:`[lower, upper]`.
    """
    
    # ========== DOF Commands ==========
    
    dof_pos_target: torch.Tensor | None = None
    """DOF position targets commanded by the user. Shape is (num_instances, num_dof).
    
    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the DOF torques.
    """
    
    dof_vel_target: torch.Tensor | None = None
    """DOF velocity targets commanded by the user. Shape is (num_instances, num_dof).
    
    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the DOF torques.
    """
    
    dof_effort_target: torch.Tensor | None = None
    """DOF effort targets commanded by the user. Shape is (num_instances, num_dof).
    
    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the DOF torques.
    """
    
    # ========== DOF Commands - Explicit Actuators ==========
    
    computed_torque: torch.Tensor | None = None
    """DOF torques computed from the actuator model (before clipping). Shape is (num_instances, num_dof).
    
    This quantity is the raw torque output from the actuator mode, before any clipping is applied.
    """
    
    applied_torque: torch.Tensor | None = None
    """DOF torques applied from the actuator model (after clipping). Shape is (num_instances, num_dof).
    
    These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
    actuator model.
    """
    
    # ========== DOF Properties - Runtime ==========
    
    dof_stiffness: torch.Tensor | None = None
    """DOF stiffness provided to the simulation. Shape is (num_instances, num_dof).
    
    In the case of explicit actuators, the value for the corresponding DOFs is zero.
    """
    
    dof_damping: torch.Tensor | None = None
    """DOF damping provided to the simulation. Shape is (num_instances, num_dof).
    
    In the case of explicit actuators, the value for the corresponding DOFs is zero.
    """
    
    dof_armature: torch.Tensor | None = None
    """DOF armature provided to the simulation. Shape is (num_instances, num_dof)."""
    
    dof_friction_coeff: torch.Tensor | None = None
    """DOF static friction coefficient provided to the simulation. Shape is (num_instances, num_dof)."""
    
    dof_dynamic_friction_coeff: torch.Tensor | None = None
    """DOF dynamic friction coefficient provided to the simulation. Shape is (num_instances, num_dof)."""
    
    dof_viscous_friction_coeff: torch.Tensor | None = None
    """DOF viscous friction coefficient provided to the simulation. Shape is (num_instances, num_dof)."""
    
    dof_pos_limits: torch.Tensor | None = None
    """DOF position limits provided to the simulation. Shape is (num_instances, num_dof, 2).
    
    The limits are in the order :math:`[lower, upper]`.
    """
    
    dof_vel_limits: torch.Tensor | None = None
    """DOF maximum velocity provided to the simulation. Shape is (num_instances, num_dof)."""
    
    dof_effort_limits: torch.Tensor | None = None
    """DOF maximum effort provided to the simulation. Shape is (num_instances, num_dof)."""
    
    # ========== DOF Properties - Custom ==========
    
    soft_dof_pos_limits: torch.Tensor | None = None
    """Soft DOF positions limits for all DOFs. Shape is (num_instances, num_dof, 2).
    
    The limits are in the order :math:`[lower, upper]`. The soft DOF position limits are computed as
    a sub-region of the :attr:`dof_pos_limits` based on a soft limit factor.
    """
    
    soft_dof_vel_limits: torch.Tensor | None = None
    """Soft DOF velocity limits for all DOFs. Shape is (num_instances, num_dof).
    
    These are obtained from the actuator model. It may differ from :attr:`dof_vel_limits` if the actuator model
    has a variable velocity limit model.
    """
