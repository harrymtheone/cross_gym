"""IsaacGym simulation context implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import torch
import numpy as np

from isaacgym import gymapi, gymutil, gymtorch

from cross_gym.sim.simulation_context import SimulationContext

if TYPE_CHECKING:
    from .isaacgym_cfg import IsaacGymCfg


class IsaacGymContext(SimulationContext):
    """IsaacGym-specific implementation of SimulationContext."""
    
    cfg: IsaacGymCfg
    
    def __init__(self, cfg: IsaacGymCfg):
        """Initialize IsaacGym simulation context.
        
        Args:
            cfg: Simulation configuration
        """
        super().__init__(cfg)
        
        # Initialize IsaacGym
        self.gym = gymapi.acquire_gym()
        
        # Create simulation
        self._create_sim()
        
    def _create_sim(self):
        """Create the IsaacGym simulation."""
        # Parse device
        sim_device, sim_device_id = gymutil.parse_device_str(str(self.device))
        
        if self.device.type == 'cuda':
            graphics_device_id = -1 if self.cfg.headless else 0
        else:
            graphics_device_id = sim_device_id
        
        # Create sim params
        sim_params = gymapi.SimParams()
        
        # Set basic parameters
        sim_params.dt = self.cfg.dt
        sim_params.substeps = self.cfg.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z if self.cfg.up_axis == "z" else gymapi.UP_AXIS_Y
        sim_params.gravity = gymapi.Vec3(*self.cfg.gravity)
        
        # GPU pipeline settings
        sim_params.use_gpu_pipeline = self.cfg.use_gpu_pipeline
        
        # PhysX settings
        sim_params.physx.use_gpu = self.cfg.physx.use_gpu
        sim_params.physx.solver_type = self.cfg.physx.solver_type
        sim_params.physx.num_position_iterations = self.cfg.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = self.cfg.physx.num_velocity_iterations
        sim_params.physx.contact_offset = self.cfg.physx.contact_offset
        sim_params.physx.rest_offset = self.cfg.physx.rest_offset
        sim_params.physx.bounce_threshold_velocity = self.cfg.physx.bounce_threshold_velocity
        sim_params.physx.max_depenetration_velocity = self.cfg.physx.max_depenetration_velocity
        sim_params.physx.friction_offset_threshold = self.cfg.physx.friction_offset_threshold
        sim_params.physx.friction_correlation_distance = self.cfg.physx.friction_correlation_distance
        
        # Num threads
        sim_params.physx.num_threads = self.cfg.physx.num_threads
        sim_params.physx.num_subscenes = self.cfg.physx.num_subscenes
        
        # Create the simulation
        self.sim = self.gym.create_sim(
            sim_device_id,
            graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params
        )
        
        if self.sim is None:
            raise RuntimeError("Failed to create IsaacGym simulation")
        
        # Create viewer if not headless
        self.viewer = None
        if not self.cfg.headless:
            self._create_viewer()
    
    def _create_viewer(self):
        """Create viewer for visualization."""
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise RuntimeError("Failed to create viewer")
        
        # Set up camera
        cam_pos = gymapi.Vec3(5.0, 5.0, 3.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def reset(self):
        """Reset the simulation."""
        # Prepare simulation
        self.gym.prepare_sim(self.sim)
        
        # Reset counters
        self._sim_step_counter = 0
        self._is_playing = True
        self._is_stopped = False
        
        # Warm up physics
        for _ in range(2):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
    
    def step(self, render: bool = True):
        """Step physics simulation.
        
        Args:
            render: Whether to render after stepping
        """
        # Simulate
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Increment counter
        self._sim_step_counter += 1
        
        # Render if needed
        if render and (self._sim_step_counter % self.cfg.render_interval == 0):
            self.render()
    
    def render(self):
        """Render the scene."""
        if self.viewer is not None:
            # Check for window close
            if self.gym.query_viewer_has_closed(self.viewer):
                self._is_stopped = True
                return
            
            # Step graphics
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            
            # Handle viewer events
            self.gym.poll_viewer_events(self.viewer)
            
            # Sync frame time
            self.gym.sync_frame_time(self.sim)
    
    def create_articulation_view(self, prim_path: str, num_envs: int) -> Any:
        """Create articulation view for IsaacGym.
        
        Args:
            prim_path: Path pattern to articulation
            num_envs: Number of environments
            
        Returns:
            IsaacGymArticulationView object
        """
        from .isaacgym_articulation_view import IsaacGymArticulationView
        return IsaacGymArticulationView(self.gym, self.sim, prim_path, num_envs, self.device)
    
    def create_rigid_object_view(self, prim_path: str, num_envs: int) -> Any:
        """Create rigid object view for IsaacGym.
        
        Args:
            prim_path: Path pattern to rigid objects
            num_envs: Number of environments
            
        Returns:
            IsaacGymRigidObjectView object
        """
        from .isaacgym_rigid_object_view import IsaacGymRigidObjectView
        return IsaacGymRigidObjectView(self.gym, self.sim, prim_path, num_envs, self.device)
    
    def add_ground_plane(self, static_friction: float = 1.0, dynamic_friction: float = 1.0, 
                         restitution: float = 0.0):
        """Add a ground plane to the simulation.
        
        Args:
            static_friction: Static friction coefficient
            dynamic_friction: Dynamic friction coefficient
            restitution: Restitution coefficient
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = static_friction
        plane_params.dynamic_friction = dynamic_friction
        plane_params.restitution = restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def get_physics_handle(self) -> Any:
        """Get IsaacGym-specific physics handle.
        
        Returns:
            Tuple of (gym, sim) handles
        """
        return (self.gym, self.sim)

