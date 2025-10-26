"""Ray caster sensor implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from cross_gym.sensors import SensorBase
from cross_gym.utils import math as math_utils
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from cross_gym.assets import Articulation
    from cross_gym.sim import SimulationContext
    from .ray_caster_cfg import RayCasterCfg


class RayCaster(SensorBase):
    """Ray caster sensor for distance measurements.
    
    Performs geometric raycasting to measure distances to obstacles/terrain.
    Uses GPU-accelerated raycasting via Open3D for performance.
    
    Features:
    - Multiple ray patterns (grid, lidar, circle)
    - Configurable range and filtering
    - Optional surface normal computation
    - Per-environment sensor positioning
    """

    def __init__(
        self,
        cfg: RayCasterCfg,
        articulation: Articulation,
        sim: SimulationContext,
    ):
        """Initialize ray caster sensor.
        
        Args:
            cfg: Ray caster configuration
            articulation: Articulation to attach sensor to
            sim: Simulation context
            
        Raises:
            ImportError: If Open3D is not installed
        """
        if not HAS_OPEN3D:
            raise ImportError(
                "Open3D is required for RayCaster sensor. "
                "Install it with: pip install open3d"
            )
        
        super().__init__(cfg, articulation, sim)
        
        self.cfg: RayCasterCfg = cfg
        
        # Generate ray pattern
        self._pattern = cfg.pattern.class_type(cfg.pattern)
        self._ray_directions_local = self._pattern.generate(self.device)
        self._num_rays = self._pattern.num_rays
        
        # Initialize data container
        self._data = RayCasterData()
        self._init_data_buffers()
        
        # Raycasting scene (will be initialized on first use)
        self._raycasting_scene: o3d.t.geometry.RaycastingScene | None = None
        self._scene_needs_update = True
    
    @property
    def data(self) -> RayCasterData:
        """Get ray caster data."""
        return self._data
    
    @property
    def num_rays(self) -> int:
        """Number of rays per sensor."""
        return self._num_rays
    
    def _init_data_buffers(self):
        """Initialize data container buffers."""
        # Sensor pose
        self._data.pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._data.quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._data.quat_w[:, 0] = 1.0  # Identity quaternion
        
        # Ray directions in world frame
        self._data.ray_directions_w = torch.zeros(
            self.num_envs, self._num_rays, 3,
            device=self.device
        )
        
        # Measurements
        self._data.distances = torch.full(
            (self.num_envs, self._num_rays),
            self.cfg.max_distance,
            device=self.device
        )
        
        self._data.hit_mask = torch.zeros(
            self.num_envs, self._num_rays,
            dtype=torch.bool,
            device=self.device
        )
        
        if self.cfg.return_hit_points:
            self._data.hit_points_w = torch.zeros(
                self.num_envs, self._num_rays, 3,
                device=self.device
            )
        
        if self.cfg.return_normals:
            self._data.hit_normals_w = torch.zeros(
                self.num_envs, self._num_rays, 3,
                device=self.device
            )
    
    def _update_buffers(self, dt: float):
        """Update ray caster measurements.
        
        Args:
            dt: Time step (unused)
        """
        # Update sensor pose in data
        self._data.pos_w.copy_(self.pos_w)
        self._data.quat_w.copy_(self.quat_w)
        
        # Transform ray directions to world frame
        for env_idx in range(self.num_envs):
            self._data.ray_directions_w[env_idx] = math_utils.quat_rotate(
                self.quat_w[env_idx:env_idx+1],
                self._ray_directions_local
            )
        
        # Perform raycasting
        self._raycast()
        
        # Store results in buffer
        self._buffer.append(self._data.distances)
    
    def _raycast(self):
        """Perform raycasting for all environments."""
        # Build raycasting scene if needed
        if self._raycasting_scene is None or self._scene_needs_update:
            self._build_raycasting_scene()
        
        # Batch raycasting for all environments
        for env_idx in range(self.num_envs):
            # Get ray origins and directions for this environment
            ray_origins = self.pos_w[env_idx:env_idx+1].expand(self._num_rays, 3)
            ray_directions = self._data.ray_directions_w[env_idx]
            
            # Convert to numpy for Open3D
            origins_np = ray_origins.cpu().numpy().astype(np.float32)
            directions_np = ray_directions.cpu().numpy().astype(np.float32)
            
            # Prepare rays: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
            rays = np.concatenate([origins_np, directions_np], axis=1)
            rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
            
            # Cast rays
            result = self._raycasting_scene.cast_rays(rays_tensor)
            
            # Extract results
            hit_t = torch.from_numpy(result['t_hit'].numpy()).to(self.device)
            
            # Check valid hits
            valid_hits = torch.isfinite(hit_t) & (hit_t >= self.cfg.min_distance)
            
            # Update distances
            self._data.distances[env_idx] = torch.where(
                valid_hits,
                torch.clamp(hit_t, max=self.cfg.max_distance),
                torch.tensor(self.cfg.max_distance, device=self.device)
            )
            
            # Update hit mask
            self._data.hit_mask[env_idx] = valid_hits
            
            # Compute hit points if requested
            if self.cfg.return_hit_points:
                hit_points = ray_origins + ray_directions * hit_t.unsqueeze(-1)
                self._data.hit_points_w[env_idx] = torch.where(
                    valid_hits.unsqueeze(-1),
                    hit_points,
                    ray_origins + ray_directions * self.cfg.max_distance
                )
            
            # Compute normals if requested
            if self.cfg.return_normals:
                if 'primitive_normals' in result:
                    normals = torch.from_numpy(
                        result['primitive_normals'].numpy()
                    ).to(self.device)
                    self._data.hit_normals_w[env_idx] = torch.where(
                        valid_hits.unsqueeze(-1),
                        normals,
                        torch.zeros(self._num_rays, 3, device=self.device)
                    )
                else:
                    # Normals not available
                    self._data.hit_normals_w[env_idx] = 0.0
    
    def _build_raycasting_scene(self):
        """Build Open3D raycasting scene from simulation meshes."""
        # TODO: Get mesh data from simulation
        # For now, create a simple ground plane for testing
        
        # Create ground plane mesh
        vertices = np.array([
            [-100, -100, 0],
            [100, -100, 0],
            [100, 100, 0],
            [-100, 100, 0],
        ], dtype=np.float32)
        
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)
        
        # Convert to Open3D tensors
        verts_tensor = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
        tris_tensor = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.Int32)
        
        # Create triangle mesh
        mesh = o3d.t.geometry.TriangleMesh(verts_tensor, tris_tensor)
        
        # Build raycasting scene
        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        _ = self._raycasting_scene.add_triangles(mesh)
        
        self._scene_needs_update = False


__all__ = ["RayCaster"]

