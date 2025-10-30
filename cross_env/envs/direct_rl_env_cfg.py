"""Configuration for direct RL environment."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from cross_core.base import InteractiveSceneCfg
from cross_core.utils import configclass

if TYPE_CHECKING:
    from .direct_rl_env import DirectRLEnv


@configclass
class DirectRLEnvCfg:
    """Configuration for direct RL environment (without managers).
    
    Simpler alternative to ManagerBasedRLEnvCfg for basic tasks.
    """

    class_type: type[DirectRLEnv] = MISSING

    # ========== Scene (includes simulation params) ==========
    scene: InteractiveSceneCfg = MISSING
    """Scene configuration (backend-specific, includes sim params)."""

    # ========== Episode ==========
    decimation: int = MISSING
    """Number of simulation steps per environment step."""

    episode_length_s: float = MISSING
    """Episode length in seconds."""
