"""Configuration for direct RL environment."""

from __future__ import annotations

from dataclasses import MISSING

from cross_gym.scene import InteractiveSceneCfg
from cross_gym.sim import SimulationContextCfg
from cross_gym.utils import configclass
from . import DirectRLEnv


@configclass
class DirectRLEnvCfg:
    """Configuration for direct RL environment (without managers).
    
    Simpler alternative to ManagerBasedRLEnvCfg for basic tasks.
    """

    class_type: type[DirectRLEnv] = MISSING

    # ========== Simulation ==========
    sim: SimulationContextCfg = MISSING
    """Simulation configuration."""

    # ========== Scene ==========
    scene: InteractiveSceneCfg = MISSING
    """Scene configuration."""

    # ========== Episode ==========
    decimation: int = MISSING
    """Number of simulation steps per environment step."""

    episode_length_s: float = MISSING
    """Episode length in seconds."""


__all__ = ["DirectRLEnvCfg"]
