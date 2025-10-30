"""IsaacGym-specific interactive scene."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaacgym import gymapi, gymutil

from cross_core.base import InteractiveScene, SensorBaseCfg
from cross_core.terrains import TerrainGeneratorCfg
from cross_gym.assets.articulation import GymArticulationCfg, GymArticulation
from cross_gym.sensors import HeightScannerCfg, RayCasterCfg

if TYPE_CHECKING:
    from . import IsaacGymSceneCfg


class IsaacGymInteractiveScene(InteractiveScene):
    """IsaacGym-specific scene manager.
    
    Owns everything:
    - Initializes IsaacGym simulator directly
    - Builds scene (terrain, assets, envs)
    - Manages articulations and sensors
    - Provides physics control (step, reset, render)
    """
    cfg: IsaacGymSceneCfg

    def __init__(self, cfg: IsaacGymSceneCfg, device: torch.device):
        """Initialize scene with IsaacGym.
        
        Args:
            cfg: IsaacGymSceneCfg
            device: Device to run simulation on
        """
        super().__init__(cfg, device)

        # Initialize IsaacGym directly (no wrapper needed)
        self.gym = gymapi.acquire_gym()  # noqa
        self.sim = self._create_sim()
        self.viewer = self._create_viewer() if not self.cfg.sim.headless else None

        # Scene storage
        self.envs = []  # Environment handles
        self.actors = {}  # Actor handles by prim_path
        self._assets = {}  # Loaded gym assets
        self._terrain = None
        self._articulations = {}
        self._sensors = {}

        # Build scene
        self._create_envs()

        # Initialize articulations and sensors
        self._create_scene_entities()

        # Counters
        self._is_stopped = False

    def _create_sim(self):
        """Create IsaacGym simulation."""
        # Parse device
        sim_device, sim_device_id = gymutil.parse_device_str(str(self.device))

        if self.device.type == 'cuda':
            graphics_device_id = -1 if self.cfg.sim.headless else 0
        else:
            graphics_device_id = sim_device_id

        # Create sim params
        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.sim.dt
        sim_params.substeps = self.cfg.sim.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z if self.cfg.sim.up_axis == "z" else gymapi.UP_AXIS_Y
        sim_params.gravity = gymapi.Vec3(*self.cfg.sim.gravity)
        sim_params.use_gpu_pipeline = self.cfg.sim.use_gpu_pipeline

        # PhysX settings
        sim_params.physx.use_gpu = self.cfg.sim.physx.use_gpu
        sim_params.physx.solver_type = self.cfg.sim.physx.solver_type
        sim_params.physx.num_position_iterations = self.cfg.sim.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = self.cfg.sim.physx.num_velocity_iterations
        sim_params.physx.contact_offset = self.cfg.sim.physx.contact_offset
        sim_params.physx.rest_offset = self.cfg.sim.physx.rest_offset
        sim_params.physx.bounce_threshold_velocity = self.cfg.sim.physx.bounce_threshold_velocity
        sim_params.physx.max_depenetration_velocity = self.cfg.sim.physx.max_depenetration_velocity
        sim_params.physx.friction_offset_threshold = self.cfg.sim.physx.friction_offset_threshold
        sim_params.physx.friction_correlation_distance = self.cfg.sim.physx.friction_correlation_distance
        sim_params.physx.num_threads = self.cfg.sim.physx.num_threads
        sim_params.physx.num_subscenes = self.cfg.sim.physx.num_subscenes

        # Create simulation
        sim = self.gym.create_sim(
            sim_device_id,
            graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params
        )

        if sim is None:
            raise RuntimeError("Failed to create IsaacGym simulation")

        return sim

    def _create_viewer(self):
        """Create viewer for visualization."""
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create viewer")

        # Set up camera
        cam_pos = gymapi.Vec3(5.0, 5.0, 3.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

        return viewer

    def _create_envs(self):
        """Create environments with terrain, assets, and actors.
        
        IsaacGym sequence:
        1. Add terrain (global, before envs)
        2. Load URDF assets
        3. Create envs and add actors (interleaved)
        4. Prepare sim
        """
        # Step 1 & 2: Process terrain and load assets
        assets_to_spawn = {}

        for name, cfg in self.cfg.__dict__.items():
            # Handle terrain
            if isinstance(cfg, TerrainGeneratorCfg):
                self._add_terrain(cfg)

            # Handle articulations
            elif isinstance(cfg, GymArticulationCfg):
                asset = self._load_urdf(cfg)
                assets_to_spawn[name] = (asset, cfg)

        # Step 3: Create envs with actors (IsaacGym interleaved requirement)
        num_per_row = int(math.sqrt(self.cfg.num_envs))
        spacing = self.cfg.env_spacing

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.actors = {name: [] for name in assets_to_spawn.keys()}

        for env_id in range(self.cfg.num_envs):
            # Create environment
            env_handle = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_handle)

            # Add all actors to this env
            for name, (asset, cfg) in assets_to_spawn.items():
                initial_pose = gymapi.Transform()
                initial_pose.p = gymapi.Vec3(*cfg.init_state.pos)
                initial_pose.r = gymapi.Quat(
                    cfg.init_state.rot[1],  # x
                    cfg.init_state.rot[2],  # y
                    cfg.init_state.rot[3],  # z
                    cfg.init_state.rot[0]  # w
                )

                actor_handle = self.gym.create_actor(
                    env_handle,
                    asset,
                    initial_pose,
                    name,
                    env_id,
                    cfg.collision_group,
                    0
                )

                self.actors[name].append(actor_handle)

        # Step 4: Prepare simulation
        self.gym.prepare_sim(self.sim)

        print(f"[IsaacGym] Created {self.cfg.num_envs} environments")

    def _add_terrain(self, terrain_cfg):
        """Add terrain using IsaacGym API directly."""
        if self._terrain is not None:
            raise RuntimeError("Terrain already created")

        # Generate terrain mesh
        self._terrain = terrain_cfg.class_type(terrain_cfg)

        # Get mesh data
        vertices = np.array(self._terrain.mesh.vertices, dtype=np.float32)
        triangles = np.array(self._terrain.mesh.faces, dtype=np.uint32)

        # Create trimesh params
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p = gymapi.Vec3(0, 0, 0)
        tm_params.static_friction = 1.0
        tm_params.dynamic_friction = 1.0
        tm_params.restitution = 0.0

        # Add to sim using direct API
        self.gym.add_triangle_mesh(
            self.sim,
            vertices.flatten().tolist(),
            triangles.flatten().tolist(),
            tm_params
        )

        print(f"[IsaacGym] Added terrain ({vertices.shape[0]} vertices)")

    def _load_urdf(self, cfg: GymArticulationCfg):
        """Load URDF using IsaacGym API directly."""
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

        # Load using direct API
        asset = self.gym.load_asset(
            self.sim,
            os.path.dirname(cfg.file),
            os.path.basename(cfg.file),
            asset_options
        )

        self._assets[cfg.prim_path] = asset
        print(f"[IsaacGym] Loaded URDF: {os.path.basename(cfg.file)}")

        return asset

    def _create_scene_entities(self):
        """Create articulations, sensors, and other scene entities."""
        # Step 1: Create articulations
        for name, cfg in self.cfg.__dict__.items():
            if isinstance(cfg, GymArticulationCfg):
                # Create articulation with direct gym/sim access
                self._articulations[name] = GymArticulation(
                    cfg=cfg,
                    actor_handles=self.actors[name],
                    gym=self.gym,
                    sim=self.sim,
                    device=self.device,
                    num_envs=self.cfg.num_envs
                )

        # Step 2: Create sensors (sensors typically need to reference articulations)
        # For now, we create sensors that are defined in the config
        # Sensors may need to be attached to specific articulations - this is
        # a simplified implementation that can be extended based on sensor requirements
        for name, cfg in self.cfg.__dict__.items():
            if isinstance(cfg, SensorBaseCfg):
                # Find the articulation this sensor should attach to
                # This assumes sensors have a way to identify their target articulation
                # For now, we'll need to extend this based on sensor configuration
                # TODO: Implement sensor creation logic based on sensor type and articulation assignment
                if isinstance(cfg, HeightScannerCfg):
                    # HeightScanner needs an articulation - determine which one
                    # For now, skip or implement based on requirements
                    pass
                elif isinstance(cfg, RayCasterCfg):
                    # RayCaster needs an articulation - determine which one  
                    # For now, skip or implement based on requirements
                    pass

    def step(self, render: bool = True):
        """Step physics using direct IsaacGym API."""
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def render(self):
        """Render using direct IsaacGym API."""
        if self.viewer is not None:
            if self.gym.query_viewer_has_closed(self.viewer):
                self._is_stopped = True
                return

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.poll_viewer_events(self.viewer)
            self.gym.sync_frame_time(self.sim)

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self.cfg.num_envs

    @property
    def is_stopped(self) -> bool:
        """Whether simulation has been stopped."""
        return self._is_stopped
