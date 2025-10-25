"""Configuration for Genesis simulator."""

from __future__ import annotations

from cross_gym.sim.simulation_context_cfg import SimulationContextCfg
from cross_gym.utils.configclass import configclass


@configclass
class GenesisSimOptionsCfg:
    """Genesis SimOptions configuration."""
    
    substeps: int = 1
    """Number of substeps per physics step."""
    
    requires_grad: bool = False
    """Whether to compute gradients."""


@configclass
class GenesisRigidOptionsCfg:
    """Genesis RigidOptions configuration."""
    
    enable_collision: bool = True
    """Enable collision detection."""
    
    enable_joint_limit: bool = True
    """Enable joint limits."""
    
    enable_self_collision: bool = True
    """Enable self-collision."""
    
    max_collision_pairs: int = 100
    """Maximum number of collision pairs."""
    
    constraint_solver: str = "Newton"
    """Constraint solver type ('Newton' or 'CG')."""


@configclass
class GenesisViewerOptionsCfg:
    """Genesis ViewerOptions configuration."""
    
    camera_pos: tuple = (2.0, 0.0, 2.5)
    """Camera position (x, y, z)."""
    
    camera_lookat: tuple = (0.0, 0.0, 0.5)
    """Camera look-at point (x, y, z)."""
    
    camera_fov: float = 40.0
    """Camera field of view in degrees."""
    
    max_FPS: int = 60
    """Maximum rendering FPS."""


@configclass
class GenesisCfg(SimulationContextCfg):
    """Configuration for Genesis simulator.
    
    This contains all the parameters specific to Genesis.
    """
    
    # class_type will be set once GenesisContext is implemented
    # class_type: type = GenesisContext
    
    # Genesis-specific options
    sim_options: GenesisSimOptionsCfg = GenesisSimOptionsCfg()
    """Genesis simulation options."""
    
    rigid_options: GenesisRigidOptionsCfg = GenesisRigidOptionsCfg()
    """Genesis rigid body options."""
    
    viewer_options: GenesisViewerOptionsCfg = GenesisViewerOptionsCfg()
    """Genesis viewer options."""
    
    backend: str = "gpu"
    """Backend to use ('gpu' or 'cpu')."""

