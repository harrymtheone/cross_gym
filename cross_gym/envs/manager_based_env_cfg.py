"""Configuration for manager-based environment."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Optional

from cross_gym.managers import ActionManagerCfg, ObservationManagerCfg, EventManagerCfg
from cross_gym.scene import InteractiveSceneCfg
from cross_gym.sim import SimulationContextCfg
from cross_gym.utils.configclass import configclass
from . import ManagerBasedEnv


@configclass
class ManagerBasedEnvCfg:
    """Configuration for manager-based environment.
    
    This is the base configuration for environments that use the manager system.
    """

    class_type: type[ManagerBasedEnv] = ManagerBasedEnv

    # Simulation
    sim: SimulationContextCfg = MISSING
    """Simulation configuration (use IsaacGymCfg, GenesisCfg, or IsaacSimCfg)."""

    # Scene
    scene: InteractiveSceneCfg = MISSING
    """Scene configuration."""

    # Managers
    actions: ActionManagerCfg = MISSING
    """Action manager configuration."""

    observations: ObservationManagerCfg = MISSING
    """Observation manager configuration."""

    events: Optional[EventManagerCfg] = None
    """Event manager configuration (optional)."""

    # Environment settings
    decimation: int = 1
    """Number of simulation steps per environment step.
    
    The environment step dt = decimation * sim.dt
    """

    is_finite_horizon: bool = False
    """Whether the cross_gym_rl task is treated as a finite or infinite horizon problem.
    
    This affects how terminal states are handled in RL algorithms.
    """

    # Random seed
    seed: Optional[int] = None
    """Random seed for the environment. Default is None."""
