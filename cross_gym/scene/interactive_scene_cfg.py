"""Configuration for interactive scene."""

from __future__ import annotations

from dataclasses import MISSING

from cross_gym.utils.configclass import configclass


@configclass
class InteractiveSceneCfg:
    """Configuration for the interactive scene.
    
    The users can inherit from this class to add entities to their scene. This is then parsed by the
    :class:`InteractiveScene` class to create the scene.
    
    Example:
        >>> @configclass
        >>> class MySceneCfg(InteractiveSceneCfg):
        >>>     # Terrain
        >>>     terrain = TerrainImporterCfg(...)
        >>>     
        >>>     # Robot
        >>>     robot = ArticulationCfg(
        >>>         prim_path="{ENV_REGEX_NS}/Robot",
        >>>         file="path/to/robot.urdf",
        >>>     )
        >>>     
        >>>     # Sensor
        >>>     height_scanner = RayCasterCfg(...)
    
    Note:
        The order of attributes matters! Add entities in this order:
        1. Terrain
        2. Physics assets (articulations, rigid objects)
        3. Sensors
        4. Non-physics assets (lights, etc.)
    """

    num_envs: int = MISSING
    """Number of environment instances."""

    lazy_sensor_update: bool = True
    """Whether to update sensors only when accessed.
    
    If True, sensor data is only computed when the sensor's `data` attribute is accessed.
    If False, all sensors are updated every time `scene.update()` is called.
    """

    replicate_physics: bool = True
    """Enable physics replication for faster environment creation.
    
    If True, all environments share the same physics assets, which is faster to create.
    If False, each environment has separate physics assets, which is more flexible but slower.
    
    Note:
        Some asset types (like deformable objects) may require this to be False.
    """
