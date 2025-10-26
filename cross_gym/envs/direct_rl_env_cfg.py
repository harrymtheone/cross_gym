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
    decimation: int = 2
    """Number of simulation steps per environment step."""

    episode_length_s: float = 10.0
    """Episode length in seconds."""

    # ========== Actions ==========
    num_actions: int = MISSING
    """Number of actions (usually equal to num_dof)."""

    # ========== Control (PD Controller) ==========
    @configclass
    class ControlCfg:
        """PD controller configuration."""
        action_scale: float = 0.5
        """Scale factor for actions to joint positions."""

        clip_actions: float = 100.0
        """Clip actions to this range."""

        stiffness: dict = {}
        """Joint stiffness (kp) for PD controller. {joint_pattern: kp}"""

        damping: dict = {}
        """Joint damping (kd) for PD controller. {joint_pattern: kd}"""

    control: ControlCfg = ControlCfg()

    # ========== Initial State ==========
    @configclass
    class InitStateCfg:
        """Initial state configuration."""
        pos: tuple = (0.0, 0.0, 0.6)
        """Initial base position."""

        rot: tuple = (1.0, 0.0, 0.0, 0.0)
        """Initial base orientation (w, x, y, z)."""

        lin_vel: tuple = (0.0, 0.0, 0.0)
        """Initial linear velocity."""

        ang_vel: tuple = (0.0, 0.0, 0.0)
        """Initial angular velocity."""

        default_joint_angles: dict = {}
        """Default joint angles. {joint_pattern: angle}"""

    init_state: InitStateCfg = InitStateCfg()


__all__ = ["DirectRLEnvCfg"]
