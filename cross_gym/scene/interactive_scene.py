"""IsaacGym-specific interactive scene."""

from cross_core.base import InteractiveSceneBase


class IsaacGymInteractiveScene(InteractiveSceneBase):
    """IsaacGym-specific scene manager.
    
    Manages articulations, sensors, and terrain for IsaacGym backend.
    """
    
    def __init__(self, cfg, sim):
        """Initialize scene with IsaacGym context.
        
        Args:
            cfg: IsaacGymSceneCfg
            sim: IsaacGymContext
        """
        self.cfg = cfg
        self.sim = sim
        
        # Build scene using sim context
        sim.build_scene(cfg)
        
        # Initialize articulations
        self._articulations = {}
        self._init_articulations()
        
        # Initialize sensors (TODO)
        self._sensors = {}
        
        # Store terrain reference
        self._terrain = sim.get_terrain()
    
    def _init_articulations(self):
        """Create articulation wrappers for configured articulations."""
        from cross_gym.assets.articulation import ArticulationCfg
        
        for attr_name, attr_value in self.cfg.__dict__.items():
            if isinstance(attr_value, ArticulationCfg):
                # Create view from sim context
                view = self.sim.create_articulation_view(attr_value.prim_path)
                
                # For now, we just store the view directly
                # TODO: Wrap in IsaacGymArticulation class
                self._articulations[attr_name] = view
    
    def get_articulation(self, name: str):
        """Get articulation by name."""
        if name not in self._articulations:
            raise ValueError(f"Articulation '{name}' not found. Available: {list(self._articulations.keys())}")
        return self._articulations[name]
    
    def get_sensor(self, name: str):
        """Get sensor by name."""
        if name not in self._sensors:
            raise ValueError(f"Sensor '{name}' not found")
        return self._sensors[name]
    
    def get_terrain(self):
        """Get terrain if configured."""
        return self._terrain
    
    @property
    def num_envs(self):
        """Number of parallel environments."""
        return self.cfg.num_envs

