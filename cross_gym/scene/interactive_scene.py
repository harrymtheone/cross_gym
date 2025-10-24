"""Interactive scene manager."""

from __future__ import annotations

import inspect
import torch
from typing import Any, Dict, List, TYPE_CHECKING

from cross_gym.assets import AssetBase, Articulation
from cross_gym.sim import SimulationContext

if TYPE_CHECKING:
    from .interactive_scene_cfg import InteractiveSceneCfg


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
        self.articulations: Dict[str, Articulation] = {}
        self.rigid_objects: Dict[str, Any] = {}  # RigidObject not implemented yet
        self.sensors: Dict[str, Any] = {}  # Sensors not implemented yet
        self.terrain: Any = None  # Terrain not implemented yet
        
        # Environment info
        self.num_envs = cfg.num_envs
        self.env_spacing = cfg.env_spacing
        
        # Parse configuration and create assets
        self._parse_config()
        
        # Clone environments (if more than one)
        if self.num_envs > 1:
            self._clone_environments()
        
        # Initialize all assets
        self._initialize_assets()
    
    def _parse_config(self):
        """Parse the configuration and create assets."""
        # Get all attributes from config that are not built-in
        for attr_name in dir(self.cfg):
            # Skip private attributes and methods
            if attr_name.startswith("_"):
                continue
            
            # Skip built-in attributes
            if attr_name in ["num_envs", "env_spacing", "lazy_sensor_update", "replicate_physics"]:
                continue
            
            attr_value = getattr(self.cfg, attr_name)
            
            # Skip methods and other non-config items
            if callable(attr_value) or inspect.ismethod(attr_value):
                continue
            
            # Check if this is an asset configuration
            if hasattr(attr_value, "class_type"):
                self._create_asset(attr_name, attr_value)
    
    def _create_asset(self, name: str, cfg: Any):
        """Create an asset from configuration.
        
        Args:
            name: Name of the asset
            cfg: Configuration object with class_type attribute
        """
        # Get the asset class
        asset_class = cfg.class_type
        
        # Create the asset instance
        asset = asset_class(cfg)
        
        # Store in appropriate container
        if isinstance(asset, Articulation):
            self.articulations[name] = asset
        # elif isinstance(asset, RigidObject):  # TODO: implement
        #     self.rigid_objects[name] = asset
        # elif isinstance(asset, SensorBase):  # TODO: implement
        #     self.sensors[name] = asset
        else:
            # For now, just store in a generic container
            if not hasattr(self, "_other_assets"):
                self._other_assets = {}
            self._other_assets[name] = asset
    
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
        for name, articulation in self.articulations.items():
            articulation.initialize(env_ids, self.num_envs)
        
        # TODO: Initialize rigid objects
        # TODO: Initialize sensors
    
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
        # TODO: Reset sensors
    
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
        
        # Update sensors (if not lazy)
        if not self.cfg.lazy_sensor_update:
            for sensor in self.sensors.values():
                sensor.update(dt)
    
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
        
        # Try other assets
        if hasattr(self, "_other_assets") and key in self._other_assets:
            return self._other_assets[key]
        
        raise KeyError(f"Asset '{key}' not found in scene")
    
    def __contains__(self, key: str) -> bool:
        """Check if asset exists in scene.
        
        Args:
            key: Name of the asset
            
        Returns:
            True if asset exists, False otherwise
        """
        return (
            key in self.articulations
            or key in self.rigid_objects
            or key in self.sensors
            or (hasattr(self, "_other_assets") and key in self._other_assets)
        )
    
    def keys(self) -> List[str]:
        """Get all asset names in the scene.
        
        Returns:
            List of asset names
        """
        keys = []
        keys.extend(self.articulations.keys())
        keys.extend(self.rigid_objects.keys())
        keys.extend(self.sensors.keys())
        if hasattr(self, "_other_assets"):
            keys.extend(self._other_assets.keys())
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

