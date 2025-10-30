"""IsaacGym-specific scene configuration."""

from cross_core.base import SceneConfigBase
from cross_core.utils import configclass


@configclass
class IsaacGymSceneCfg(SceneConfigBase):
    """IsaacGym-specific scene configuration.
    
    Attributes are dynamically added for articulations, sensors, terrain.
    Backend identifier allows task configs to select simulator.
    """
    
    backend: str = "isaacgym"
    """Backend identifier."""
    
    num_envs: int = 1024
    """Number of parallel environments."""
    
    env_spacing: float = 2.0
    """Spacing between environments in meters."""
    
    # Additional attributes for terrain, articulations, sensors
    # are added dynamically as needed, e.g.:
    # terrain: TerrainGeneratorCfg | None = None
    # robot: ArticulationCfg = ...
    # height_scanner: HeightScannerCfg = ...

