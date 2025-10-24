"""Configuration classes for simulation."""
from __future__ import annotations

from dataclasses import field
from typing import Tuple

import torch

from cross_gym.utils.configclass import configclass
from .simulator_type import SimulatorType


@configclass
class PhysxCfg:
    """Configuration for PhysX physics engine settings."""

    # Solver settings
    solver_type: int = 1  # 0: PGS, 1: TGS
    min_position_iteration_count: int = 1
    max_position_iteration_count: int = 255
    min_velocity_iteration_count: int = 0
    max_velocity_iteration_count: int = 255

    # Contact settings
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    bounce_threshold_velocity: float = 0.2
    max_depenetration_velocity: float = 1000.0

    # GPU settings
    use_gpu: bool = True
    num_threads: int = 4
    num_subscenes: int = 4

    # Other settings
    enable_ccd: bool = False
    enable_stabilization: bool = False
    friction_offset_threshold: float = 0.04
    friction_correlation_distance: float = 0.025


@configclass
class RenderCfg:
    """Configuration for rendering settings."""

    # Rendering mode
    rendering_mode: str | None = None  # "performance", "balanced", "quality"

    # RT features
    enable_translucency: bool | None = None
    enable_reflections: bool | None = None
    enable_global_illumination: bool | None = None
    enable_shadows: bool | None = None
    enable_ambient_occlusion: bool | None = None
    enable_direct_lighting: bool | None = None

    # Sampling
    samples_per_pixel: int | None = None

    # DLSS
    enable_dlssg: bool | None = None
    enable_dl_denoiser: bool | None = None
    dlss_mode: str | None = None
    antialiasing_mode: str | None = None

    # General carb settings
    carb_settings: dict | None = None


@configclass
class SimulationCfg:
    """Configuration for simulation context.
    
    This is the main configuration class that determines which simulator to use
    and how to configure it.
    """

    # Simulator selection
    simulator: SimulatorType = SimulatorType.ISAACGYM

    # Device
    device: str = "cuda:0"

    # Time settings
    dt: float = 0.01  # Physics timestep in seconds
    render_interval: int = 1  # Render every N physics steps

    # Gravity
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Physics settings
    physx: PhysxCfg = field(default_factory=PhysxCfg)

    # Rendering settings
    render: RenderCfg = field(default_factory=RenderCfg)

    # Simulation control
    headless: bool = True
    enable_scene_query_support: bool = False

    # Other
    physics_prim_path: str = "/physicsScene"

    def validate(self):
        """Validate the configuration."""
        if self.dt <= 0:
            raise ValueError(f"Physics timestep must be positive, got {self.dt}")

        if self.render_interval < 1:
            raise ValueError(f"Render interval must be at least 1, got {self.render_interval}")

        # Convert device string to torch.device for validation
        try:
            torch.device(self.device)
        except Exception as e:
            raise ValueError(f"Invalid device specification '{self.device}': {e}")
