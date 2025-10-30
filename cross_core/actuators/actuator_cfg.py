"""Configuration classes for actuators."""

from __future__ import annotations

from dataclasses import MISSING

from cross_core.utils import configclass
from . import IdealPDActuator


@configclass
class ActuatorBaseCfg:
    """Base configuration for actuators."""

    class_type: type = MISSING
    """Actuator class to instantiate."""

    joint_names_expr: list[str] = MISSING
    """Joint name patterns (regex) for this actuator group."""

    stiffness: float | dict[str, float] = 0.0
    """PD stiffness (kp). Can be single value or dict {joint_pattern: kp}."""

    damping: float | dict[str, float] = 0.0
    """PD damping (kd). Can be single value or dict {joint_pattern: kd}."""

    effort_limit: float | dict[str, float] | None = None
    """Max torque. If None, uses value from URDF."""

    velocity_limit: float | dict[str, float] | None = None
    """Max velocity. If None, uses value from URDF."""


@configclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for ideal PD actuator."""

    class_type: type = IdealPDActuator
