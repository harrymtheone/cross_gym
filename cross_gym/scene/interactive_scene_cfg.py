"""IsaacGym-specific scene configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cross_core.base import SceneBaseCfg
from cross_core.utils import configclass

if TYPE_CHECKING:
    from .interactive_scene import IsaacGymInteractiveScene


@configclass
class IsaacGymSceneCfg(SceneBaseCfg):
    """IsaacGym-specific scene configuration.
    
    Attributes are dynamically added for articulations, sensors, terrain.
    
    Usage:
        scene = scene_cfg.class_type(scene_cfg, sim_context)
    """
    
    class_type: type[IsaacGymInteractiveScene] = None  # Set in __init__.py
    
    num_envs: int = 1024
    """Number of parallel environments."""
    
    env_spacing: float = 2.0
    """Spacing between environments in meters."""
    
    # Additional attributes for terrain, articulations, sensors
    # are added dynamically as needed, e.g.:
    # terrain: TerrainGeneratorCfg | None = None
    # robot: ArticulationCfg = ...
    # height_scanner: HeightScannerCfg = ...

