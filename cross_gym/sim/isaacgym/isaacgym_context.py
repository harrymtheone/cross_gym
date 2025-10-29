"""IsaacGym simulation context implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cross_gym.sim import SimulationContext
from . import IsaacGymRigidObjectView, IsaacGymArticulationView

if TYPE_CHECKING:
    from . import IsaacGymCfg
    from cross_gym.assets import ArticulationCfg
    from cross_gym.terrains import TerrainGenerator

try:
    from isaacgym import gymapi, gymutil  # noqa

    ISAACGYM_AVAILABLE = True

except ImportError as e:
    if e.msg == "'PyTorch was imported before isaacgym modules.  Please import torch after isaacgym modules.'":
        raise ImportError(e)
    else:
        ISAACGYM_AVAILABLE = False


class IsaacGymContext(SimulationContext):
    """IsaacGym-specific implementation of SimulationContext."""

    cfg: IsaacGymCfg

    def __init__(self, cfg: IsaacGymCfg):
        """Initialize IsaacGym simulation context.
        
        Args:
            cfg: Simulation configuration
        """
        if not ISAACGYM_AVAILABLE:
            raise RuntimeError("IsaacGym is not available!")

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
        
        # Asset spawning storage
        self.envs = []  # Environment handles
        self.actors = {}  # Actor handles by name
        self._assets = {}  # Loaded gym assets

    def _create_viewer(self):
        """Create viewer for visualization."""
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise RuntimeError("Failed to create viewer")

        # Set up camera
        cam_pos = gymapi.Vec3(5.0, 5.0, 3.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ========== Asset Spawning ==========

    def add_terrain_to_sim(self, terrain: TerrainGenerator):
        """Add terrain as global trimesh to simulation (BEFORE creating envs).
        
        Args:
            terrain: Terrain generator with global trimesh
        """
        import numpy as np
        
        # Get trimesh data
        vertices = np.array(terrain.mesh.vertices, dtype=np.float32)
        triangles = np.array(terrain.mesh.faces, dtype=np.uint32)
        
        # Create triangle mesh parameters
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p = gymapi.Vec3(0, 0, 0)
        tm_params.static_friction = 1.0
        tm_params.dynamic_friction = 1.0
        tm_params.restitution = 0.0
        
        # Add terrain to SIM (global, not per-env)
        self.gym.add_triangle_mesh(
            self.sim,
            vertices.flatten().tolist(),
            triangles.flatten().tolist(),
            tm_params
        )
        
        print(f"[IsaacGym] Added terrain to sim ({vertices.shape[0]} vertices, {triangles.shape[0]} faces)")

    def load_urdf_asset(self, urdf_path: str, cfg: ArticulationCfg):
        """Load URDF as gym asset (called once, reused across envs).
        
        Args:
            urdf_path: Path to URDF file
            cfg: Articulation configuration
            
        Returns:
            Gym asset handle
        """
        import os
        
        # Configure asset options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = cfg.asset_options.fix_base_link
        asset_options.collapse_fixed_joints = cfg.asset_options.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = cfg.asset_options.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = cfg.asset_options.flip_visual_attachments
        asset_options.default_dof_drive_mode = cfg.asset_options.default_dof_drive_mode
        asset_options.density = cfg.asset_options.density
        asset_options.angular_damping = cfg.asset_options.angular_damping
        asset_options.linear_damping = cfg.asset_options.linear_damping
        asset_options.max_angular_velocity = cfg.asset_options.max_angular_velocity
        asset_options.max_linear_velocity = cfg.asset_options.max_linear_velocity
        asset_options.armature = cfg.asset_options.armature
        asset_options.thickness = cfg.asset_options.thickness
        asset_options.disable_gravity = cfg.asset_options.disable_gravity
        
        # Load URDF
        asset = self.gym.load_asset(
            self.sim,
            os.path.dirname(urdf_path),
            os.path.basename(urdf_path),
            asset_options
        )
        
        print(f"[IsaacGym] Loaded URDF asset: {os.path.basename(urdf_path)}")
        return asset

    def create_envs_with_actors(self, num_envs: int, assets_to_spawn: dict, spacing: float = 2.0):
        """Create environments and add actors (Isaac Gym requires per-env creation).
        
        Args:
            num_envs: Number of environments
            assets_to_spawn: Dict mapping (prim_path, cfg) -> gym_asset
            spacing: Environment spacing
        """
        import numpy as np
        import math
        
        # Compute grid
        num_per_row = int(math.sqrt(num_envs))
        
        # Environment bounds
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # Create each env and immediately add all actors to it
        self.envs = []
        self.actors = {prim_path: [] for prim_path in assets_to_spawn.keys()}
        
        for env_id in range(num_envs):
            # Create environment
            env_handle = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_handle)
            
            # Add all actors to THIS env before creating next env
            for (prim_path, cfg), asset in assets_to_spawn.items():
                # Initial pose
                initial_pose = gymapi.Transform()
                initial_pose.p = gymapi.Vec3(*cfg.init_state.pos)
                initial_pose.r = gymapi.Quat(
                    cfg.init_state.rot[1],  # x
                    cfg.init_state.rot[2],  # y
                    cfg.init_state.rot[3],  # z
                    cfg.init_state.rot[0]   # w
                )
                
                # Extract actor name
                actor_name = prim_path.split('/')[-1].replace('.*', '')
                
                # Create actor in this env
                actor_handle = self.gym.create_actor(
                    env_handle,
                    asset,
                    initial_pose,
                    actor_name,
                    env_id,  # Collision group
                    cfg.collision_group,  # Collision filter
                    0  # Segmentation ID
                )
                
                self.actors[prim_path].append(actor_handle)
        
        print(f"[IsaacGym] Created {num_envs} environments with actors ({num_per_row}x{num_per_row} grid)")

    def prepare_sim(self):
        """Prepare simulation after spawning all assets."""
        self.gym.prepare_sim(self.sim)
        print(f"[IsaacGym] Simulation prepared (physics buffers allocated)")

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

    def create_articulation_view(self, prim_path: str, num_envs: int) -> IsaacGymArticulationView:
        """Create articulation view for IsaacGym.
        
        Args:
            prim_path: Path pattern to articulation
            num_envs: Number of environments
            
        Returns:
            IsaacGymArticulationView object
        """
        return IsaacGymArticulationView(self.gym, self.sim, prim_path, num_envs, self.device)

    def create_rigid_object_view(self, prim_path: str, num_envs: int) -> IsaacGymRigidObjectView:
        """Create rigid object view for IsaacGym.
        
        Args:
            prim_path: Path pattern to rigid objects
            num_envs: Number of environments
            
        Returns:
            IsaacGymRigidObjectView object
        """
        return IsaacGymRigidObjectView(self.gym, self.sim, prim_path, num_envs, self.device)

    def add_ground_plane(
            self,
            static_friction: float = 1.0,
            dynamic_friction: float = 1.0,
            restitution: float = 0.0
    ):
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
