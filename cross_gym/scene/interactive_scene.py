"""Interactive scene manager."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from cross_gym.assets import AssetBase, Articulation, ArticulationCfg
from cross_gym.sensors import SensorBaseCfg
from cross_gym.sim import SimulationContext
from cross_gym.terrains import TerrainGeneratorCfg
from . import MeshRegistry

if TYPE_CHECKING:
    from . import InteractiveSceneCfg


class InteractiveScene:
    """Interactive scene manager that handles all assets in the environment.
    
    The scene parses a configuration class and creates/manages all assets:
    - Articulations (robots)
    - Rigid objects
    - Sensors (cameras, raycasters, IMUs)
    - Terrain
    
    It handles:
    - Creating and initializing assets
    - Cloning assets across multiple environments
    - Coordinating state updates
    - Providing access to assets by name
    
    Example:
        >>> # Define scene configuration
        >>> @configclass
        >>> class MySceneCfg(InteractiveSceneCfg):
        >>>     robot = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot", ...)
        >>>     
        >>> # Create scene
        >>> scene = InteractiveScene(cfg=MySceneCfg(num_envs=128, env_spacing=4.0))
        >>> 
        >>> # Access robot
        >>> robot = scene["robot"]
        >>> robot = scene.articulations["robot"]
    """

    def __init__(self, cfg: InteractiveSceneCfg):
        """Initialize the interactive scene.
        
        Args:
            cfg: Configuration for the scene.
        """
        # Store configuration
        self.cfg = cfg

        # Get simulation context
        self.sim: SimulationContext = SimulationContext.instance()
        if self.sim is None:
            raise RuntimeError(
                "No SimulationContext found. Create a SimulationContext "
                "before initializing the scene."
            )

        # Containers for different asset types
        self.articulations: dict[str, Articulation] = {}
        self.rigid_objects: dict[str, Any] = {}  # RigidObject not implemented yet
        self.sensors: dict[str, Any] = {}  # Sensors not implemented yet
        self.terrain: Any = None  # Terrain not implemented yet

        # Environment info
        self.num_envs = cfg.num_envs

        # Create mesh registry for sharing meshes between terrain and sensors
        self.mesh_registry = MeshRegistry(device=self.sim.device)

        # Parse configuration and create assets
        self._parse_cfg()

        # Clone environments (if more than one)
        if self.num_envs > 1:
            self._clone_environments()

        # Initialize all assets
        self._initialize_assets()

    def _parse_cfg(self):
        """Parse the configuration and create assets in correct order.
        
        Order matters:
        1. Terrain - needs mesh_registry
        2. Articulations - needs nothing special
        3. Sensors - needs articulations to be created first
        """
        # Create terrain (registers meshes for sensors)
        for name, cfg in self.cfg.__dict__.items():
            if isinstance(cfg, TerrainGeneratorCfg):
                if self.terrain is not None:
                    raise RuntimeError(f"Multiple terrain configurations found: {name}")

                self.terrain = cfg.class_type(cfg)
                print(f"[Scene] Created terrain: {name}")

        # Create articulations
        for name, cfg in self.cfg.__dict__.items():
            if isinstance(cfg, ArticulationCfg):
                self.articulations[name] = cfg.class_type(cfg)
                print(f"[Scene] Created articulation: {name}")

        # Create sensors (attached to articulations)
        for name, cfg in self.cfg.__dict__.items():
            if not isinstance(cfg, SensorBaseCfg):
                continue

            # Find parent articulation by name
            if cfg.articulation_name not in self.articulations:
                raise RuntimeError(
                    f"Sensor '{name}' references articulation '{cfg.articulation_name}' "
                    f"which does not exist. Available articulations: {list(self.articulations.keys())}"
                )

            parent_articulation = self.articulations[cfg.articulation_name]

            self.sensors[name] = cfg.class_type(
                cfg,
                articulation=parent_articulation,
            )

            print(f"[Scene] Created sensor: {name} (attached to {cfg.articulation_name}/{cfg.body_name})")

    def _clone_environments(self):
        """Clone assets across multiple environments.
        
        This creates multiple instances of each asset, one for each environment.
        The actual cloning is handled by the simulator backend.
        """
        # For now, this is a placeholder
        # The actual cloning will be implemented when we add spawners
        # and integrate with the simulator's cloning capabilities
        pass

    def _initialize_assets(self):
        """Initialize all assets after creation."""
        # Create environment IDs tensor
        env_ids = torch.arange(self.num_envs, device=self.sim.device, dtype=torch.long)

        # Initialize articulations
        for articulation in self.articulations.values():
            articulation.initialize(env_ids, self.num_envs)

        # TODO: Initialize rigid objects
        
        # Sensors initialize lazily on first data access (no explicit initialization needed)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset specified environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, reset all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim.device, dtype=torch.long)

        # Reset all articulations
        for articulation in self.articulations.values():
            articulation.reset(env_ids)

        # TODO: Reset rigid objects

        # Reset sensors
        for sensor in self.sensors.values():
            sensor.reset(env_ids)

    def update(self, dt: float):
        """Update all assets in the scene.
        
        This reads the latest state from the simulator for all assets.
        
        Args:
            dt: Time step in seconds
        """
        # Update articulations
        for articulation in self.articulations.values():
            articulation.update(dt)

        # TODO: Update rigid objects

        # Update sensors
        # In lazy mode: sensors only compute when data is accessed
        # In eager mode: all sensors compute every step
        for sensor in self.sensors.values():
            sensor.update(dt, force_recompute=not self.cfg.lazy_sensor_update)

    def write_data_to_sim(self):
        """Write data from all assets to the simulation.
        
        This writes buffered commands (like joint torques) to the simulator.
        """
        # Write articulation data
        for articulation in self.articulations.values():
            articulation.write_data_to_sim()

        # TODO: Write rigid object data

    # ========== Dictionary-style Access ==========

    def __getitem__(self, key: str) -> AssetBase:
        """Access assets by name using dictionary syntax.
        
        Args:
            key: Name of the asset
            
        Returns:
            The requested asset
            
        Raises:
            KeyError: If asset name not found
        """
        # Try articulations
        if key in self.articulations:
            return self.articulations[key]

        # Try rigid objects
        if key in self.rigid_objects:
            return self.rigid_objects[key]

        # Try sensors
        if key in self.sensors:
            return self.sensors[key]

        raise KeyError(f"Asset '{key}' not found in scene")

    def __contains__(self, key: str) -> bool:
        """Check if asset exists in scene.
        
        Args:
            key: Name of the asset
            
        Returns:
            True if asset exists, False otherwise
        """
        return any((key in self.articulations, key in self.rigid_objects, key in self.sensors))

    def keys(self) -> list[str]:
        """Get all asset names in the scene.
        
        Returns:
            List of asset names
        """
        keys = []
        keys.extend(self.articulations.keys())
        keys.extend(self.rigid_objects.keys())
        keys.extend(self.sensors.keys())
        return keys

    def __repr__(self) -> str:
        """String representation of the scene."""
        msg = f"<InteractiveScene with {self.num_envs} environments>\n"
        msg += f"  Articulations: {list(self.articulations.keys())}\n"
        msg += f"  Rigid Objects: {list(self.rigid_objects.keys())}\n"
        msg += f"  Sensors: {list(self.sensors.keys())}\n"
        if self.terrain is not None:
            msg += f"  Terrain: {type(self.terrain).__name__}\n"
        return msg
