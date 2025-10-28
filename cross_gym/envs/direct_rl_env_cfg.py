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

    # ========== Actions ==========
    num_actions: int = MISSING
    """Number of actions (usually equal to num_dof)."""

    # ========== Control (PD Controller) ==========
    @configclass
    class ControlCfg:
        """PD controller configuration."""
        action_scale: float = MISSING
        """Scale factor for actions to joint positions."""

        clip_actions: float = 100.0
        """Clip actions to this range."""

        stiffness: dict = {}
        """DOF stiffness (kp) for PD controller. {dof_pattern: kp}"""

        damping: dict = {}
        """DOF damping (kd) for PD controller. {dof_pattern: kd}"""

    control: ControlCfg = ControlCfg()

    # ========== Domain Randomization ==========
    @configclass
    class DomainRandCfg:
        """Domain randomization configuration for robust training."""
        
        # Root state randomization
        randomize_start_pos_xy: bool = False
        """Randomize initial xy position."""
        randomize_start_pos_xy_range: tuple[float, float] = (-0.5, 0.5)
        """Range for xy position randomization (min, max) in meters."""
        
        randomize_start_pos_z: bool = False
        """Randomize initial z height."""
        randomize_start_pos_z_range: tuple[float, float] = (0.0, 0.1)
        """Range for z height randomization (min, max) in meters."""
        
        randomize_start_yaw: bool = False
        """Randomize initial yaw orientation."""
        randomize_start_yaw_range: tuple[float, float] = (-3.14, 3.14)
        """Range for yaw randomization (min, max) in radians."""
        
        randomize_start_pitch: bool = False
        """Randomize initial pitch orientation."""
        randomize_start_pitch_range: tuple[float, float] = (-0.2, 0.2)
        """Range for pitch randomization (min, max) in radians."""
        
        randomize_start_lin_vel_xy: bool = False
        """Randomize initial linear velocity (xy components only)."""
        randomize_start_lin_vel_xy_range: tuple[float, float] = (-0.5, 0.5)
        """Range for xy linear velocity randomization (min, max) in m/s."""

        # Joint state randomization
        randomize_start_dof_pos: bool = False
        """Randomize initial DOF positions."""
        randomize_start_dof_pos_range: tuple[float, float] = (-0.1, 0.1)
        """Range for DOF position randomization (min, max) in radians."""
        
        randomize_start_dof_vel: bool = False
        """Randomize initial DOF velocities."""
        randomize_start_dof_vel_range: tuple[float, float] = (-0.5, 0.5)
        """Range for DOF velocity randomization (min, max) in rad/s."""
        
        # Torque computation randomization
        randomize_motor_offset: bool = False
        """Randomize motor position offset (simulates calibration error)."""
        motor_offset_range: tuple[float, float] = (-0.02, 0.02)
        """Motor offset range (min, max) in radians."""
        
        randomize_gains: bool = False
        """Randomize PD gains (simulates model uncertainty)."""
        kp_multiplier_range: tuple[float, float] = (0.8, 1.2)
        """Kp gain multiplier range (min, max)."""
        kd_multiplier_range: tuple[float, float] = (0.5, 1.5)
        """Kd gain multiplier range (min, max)."""
        
        randomize_torque: bool = False
        """Randomize output torque (simulates actuator variance)."""
        torque_multiplier_range: tuple[float, float] = (0.9, 1.1)
        """Torque multiplier range (min, max)."""

    domain_rand: DomainRandCfg = DomainRandCfg()


__all__ = ["DirectRLEnvCfg"]
