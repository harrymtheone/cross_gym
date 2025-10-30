"""Base articulation configuration."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from cross_core.utils import configclass

if TYPE_CHECKING:
    from cross_core.actuators import ActuatorBaseCfg


@configclass
class ArticulationBaseCfg:
    """Base class for articulation configuration.
    
    This defines the common configuration parameters for articulations across
    different simulators. Simulator-specific implementations should inherit from this.
    """

    prim_path: str = MISSING
    """Path/pattern to articulation in scene."""

    @configclass
    class InitialStateCfg:
        """Initial state configuration for the articulation."""

        # Root position (from AssetBaseCfg.InitialStateCfg)
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) of the root in simulation world frame.
        Defaults to (1.0, 0.0, 0.0, 0.0).
        """

        # Root velocity
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

        # DOF state
        dof_pos: dict[str, float] = {".*": 0.0}
        """DOF positions of the DOFs. Defaults to 0.0 for all DOFs.
        
        Uses pattern matching where keys are regex patterns and values are the position values.
        Example: {"leg_.*": 0.5, "arm_.*": -0.3} sets all leg DOFs to 0.5 and arm DOFs to -0.3.
        """

        dof_vel: dict[str, float] = {".*": 0.0}
        """DOF velocities of the DOFs. Defaults to 0.0 for all DOFs.
        
        Uses pattern matching where keys are regex patterns and values are the velocity values.
        Example: {"leg_.*": 0.1} sets all leg DOF velocities to 0.1.
        """

    articulation_root_prim_path: str | None = None
    """Path to the articulation root prim under the :attr:`prim_path`. Defaults to None.
    
    This path should be relative to the :attr:`prim_path` of the asset. If the asset is loaded from a USD file,
    this path should be relative to the root of the USD stage. For instance, if the loaded USD file at :attr:`prim_path`
    contains two articulations, one at `/robot1` and another at `/robot2`, and you want to use `robot2`,
    then you should set this to `/robot2`.
    
    The path must start with a slash (`/`).
    """

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero DOF state."""

    soft_dof_pos_limit_factor: float = 1.0
    """Fraction specifying the range of DOF position limits (parsed from the asset) to use. Defaults to 1.0.
    
    The soft DOF position limits are scaled by this factor to specify a safety region within the simulated
    DOF position limits. This isn't used by the simulation, but is useful for learning agents to prevent the DOF
    positions from violating the limits, such as for termination conditions.
    
    The soft DOF position limits are accessible through the :attr:`ArticulationData.soft_dof_pos_limits` attribute.
    """

    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding DOF names.
    
    Dictionary mapping DOF names (or patterns) to actuator configurations.
    Each actuator defines how a DOF is controlled (e.g., PD control, impedance control).
    """
