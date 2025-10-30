"""Abstract base class for simulation context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from . import ArticulationView

if TYPE_CHECKING:
    from . import SimulationContextCfg
    from cross_gym.terrains import TerrainGenerator
    from cross_gym.scene import InteractiveSceneCfg


class SimulationContext(ABC):
    """Abstract base class for simulator contexts.
    
    This class defines the common interface that all simulator-specific contexts
    must implement. It provides a unified API for:
    - Stepping the physics simulation
    - Resetting the simulation
    - Rendering the scene
    - Managing simulation state
    
    Each simulator (IsaacGym, Genesis, IsaacSim) implements this interface
    with its own specific backend.
    """

    # Class-level instance for singleton pattern
    _instance: SimulationContext = None

    def __init__(self, cfg: SimulationContextCfg):
        """Initialize the simulation context.
        
        Args:
            cfg: Configuration for the simulation (SimCfgBase subclass).
        """
        self.cfg = cfg

        # Runtime validation
        if cfg.dt <= 0:
            raise ValueError(f"Physics timestep must be positive, got {cfg.dt}")
        if cfg.render_interval < 1:
            raise ValueError(f"Render interval must be at least 1, got {cfg.render_interval}")

        # Device
        self._device = torch.device(cfg.device)

        # Set as singleton instance
        if SimulationContext._instance is not None:
            raise RuntimeError(
                "Only one SimulationContext instance can exist at a time. "
                "Call SimulationContext.clear_instance() first."
            )
        SimulationContext._instance = self

        # Simulation state
        self._is_playing = False
        self._is_stopped = True
        self._sim_step_counter = 0

    @classmethod
    def instance(cls) -> SimulationContext:
        """Get the singleton instance of SimulationContext.
        
        Returns:
            The current SimulationContext instance, or None if not created.
        """
        return cls._instance

    @classmethod
    def clear_instance(cls):
        """Clear the singleton instance."""
        if cls._instance is not None:
            del cls._instance

    @property
    def device(self):
        return self._device

    # ========== Core Abstract Methods ==========

    @abstractmethod
    def reset(self):
        """Reset the simulation to initial state.
        
        This prepares the simulation for running by:
        - Initializing physics
        - Setting up the scene
        - Resetting all state
        """
        pass

    @abstractmethod
    def step(self, render: bool = True):
        """Step the physics simulation forward by one timestep.
        
        Args:
            render: Whether to render the scene after stepping.
        """
        pass

    @abstractmethod
    def render(self):
        """Render the current scene."""
        pass

    # ========== Asset Spawning (Must be implemented by backends) ==========

    @abstractmethod
    def _add_terrain_to_sim(self, terrain: TerrainGenerator):
        """Add terrain to simulation (BEFORE creating envs).
        
        Args:
            terrain: Terrain generator
        """
        pass

    @abstractmethod
    def load_urdf_asset(self, urdf_path: str, cfg):
        """Load URDF as asset (called once, reused across envs).
        
        Args:
            urdf_path: Path to URDF file
            cfg: Articulation configuration
            
        Returns:
            Gym asset handle
        """
        pass

    @abstractmethod
    def create_envs_with_actors(self, num_envs: int, assets_to_spawn: dict, spacing: float = 2.0):
        """Create environments and add actors (per-env creation required by Isaac Gym).
        
        Args:
            num_envs: Number of environments
            assets_to_spawn: Dict mapping (prim_path, cfg) -> asset
            spacing: Environment spacing
        """
        pass

    @abstractmethod
    def prepare_sim(self):
        """Prepare simulation after spawning (allocate buffers)."""
        pass

    # ========== Scene Management ==========

    @abstractmethod
    def create_articulation_view(self, prim_path: str, num_envs: int) -> ArticulationView:
        """Create a view for articulated bodies (robots).
        
        Args:
            prim_path: Path pattern to the articulation (e.g., "/World/envs/env_.*/robot")
            num_envs: Number of parallel environments
            
        Returns:
            Simulator-specific articulation view object
        """
        pass

    @abstractmethod
    def create_rigid_object_view(self, prim_path: str, num_envs: int) -> Any:
        """Create a view for rigid bodies.
        
        Args:
            prim_path: Path pattern to the rigid objects
            num_envs: Number of parallel environments
            
        Returns:
            Simulator-specific rigid object view object
        """
        pass

    # ========== Scene Building Interface ==========

    @abstractmethod
    def build_scene(self, scene_cfg: InteractiveSceneCfg):
        """Build complete scene from configuration.
        
        This method handles the simulator-specific sequence for creating:
        - Terrain (if configured)
        - All assets (articulations, rigid objects)
        - Multiple environment instances
        
        The exact sequence is simulator-specific:
        - IsaacGym: terrain → load assets → per-env interleaved creation
        - Genesis: scene.add_terrain() → scene.add_entity() → scene.build()
        
        Args:
            scene_cfg: Complete scene configuration
        """
        pass

    @abstractmethod
    def get_terrain(self) -> TerrainGenerator:
        """Get terrain object after scene building.
        
        Returns:
            TerrainGenerator instance if terrain was created, None otherwise
        """
        pass

    @abstractmethod
    def get_articulation_view(self, prim_path: str) -> ArticulationView:
        """Get articulation view by prim path.
        
        This is called by Articulation during initialization to bind to its
        physics representation in the simulator.
        
        Args:
            prim_path: Primitive path of the articulation
            
        Returns:
            Simulator-specific articulation view
            
        Raises:
            ValueError: If articulation with prim_path not found
        """
        pass

    # ========== Utility Methods ==========

    def has_gui(self) -> bool:
        """Check if simulation has a GUI enabled.
        
        Returns:
            True if GUI is enabled, False for headless mode.
        """
        return not self.cfg.headless

    def get_physics_handle(self) -> Any:
        """Get simulator-specific physics handle.
        
        This allows access to simulator-specific features not covered
        by the abstract interface.
        
        Returns:
            Simulator-specific physics simulation handle
        """
        raise NotImplementedError(
            "Simulator-specific physics handle not implemented. "
            "Override this method in the simulator-specific context."
        )
