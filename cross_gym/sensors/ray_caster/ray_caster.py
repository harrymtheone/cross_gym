"""Ray caster sensor implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from cross_gym.scene import MeshRegistry
from cross_gym.sensors import SensorBase
from cross_gym.utils import math as math_utils
from . import RayCasterData
from .warp_kernel import raycast_mesh

if TYPE_CHECKING:
    from cross_gym.assets import Articulation
    from . import RayCasterCfg


class RayCaster(SensorBase):
    """Ray caster sensor for distance measurements.
    
    Performs geometric raycasting to measure distances to obstacles/terrain.
    Uses GPU-accelerated raycasting via NVIDIA Warp.
    
    Features:
    - Multiple ray patterns (grid, lidar, circle)
    - Configurable range and filtering
    - Per-environment sensor positioning
    """

    cfg: RayCasterCfg
    _data: RayCasterData

    def __init__(
            self,
            cfg: RayCasterCfg,
            articulation: Articulation,
            **kwargs,
    ):
        """Initialize ray caster sensor.
        
        Args:
            cfg: Ray caster configuration
            articulation: Articulation to attach sensor to
            **kwargs: Additional arguments passed to parent
            
        Raises:
            RuntimeError: If Warp is not available
        """
        # Validate and setup mesh registry
        self._validate_mesh_configuration()

        # Generate ray pattern (before calling super().__init__ which calls _init_data)
        self._pattern = cfg.pattern.class_type(cfg.pattern)
        self._ray_directions_local = self._pattern.generate(articulation.device)
        self._num_rays = self._pattern.num_rays

        self._data = RayCasterData()
        super().__init__(cfg, articulation, **kwargs)

    @property
    def data(self) -> RayCasterData:
        """Get ray caster data.
        
        This triggers lazy evaluation: the sensor only computes
        if it's outdated based on update_period.
        """
        self._update_outdated_buffers()
        return self._data

    @property
    def num_rays(self) -> int:
        """Number of rays per sensor."""
        return self._num_rays

    def _validate_mesh_configuration(self):
        """Validate mesh registry and configuration.
        
        Checks:
        - MeshRegistry singleton exists
        - At least one mesh name is configured
        - Only one mesh is configured (current limitation)
        - The configured mesh exists in the registry
        
        Raises:
            RuntimeError: If MeshRegistry not found
            ValueError: If mesh configuration is invalid
            NotImplementedError: If multiple meshes are configured
        """
        # Get mesh registry from singleton
        self.mesh_registry = MeshRegistry.instance()

        if self.mesh_registry is None:
            raise RuntimeError(
                "MeshRegistry not found. Ensure InteractiveScene is created before sensors."
            )

        # Validate mesh configuration
        if len(self.cfg.mesh_names) == 0:
            raise ValueError("RayCaster requires at least one mesh name in cfg.mesh_names")

        if len(self.cfg.mesh_names) > 1:
            raise NotImplementedError(
                f"RayCaster currently only supports single mesh raycasting. "
                f"Got {len(self.cfg.mesh_names)} meshes: {self.cfg.mesh_names}. "
                f"Use only one mesh name for now."
            )

        # Validate mesh exists in registry
        mesh_name = self.cfg.mesh_names[0]
        if not self.mesh_registry.mesh_exists(mesh_name):
            raise ValueError(
                f"Mesh '{mesh_name}' not found in registry. "
                f"Available meshes: {self.mesh_registry.list_meshes()}"
            )

    def _init_data(self):
        """Initialize sensor data container.
        
        Creates the RayCasterData container, initializes base fields via super(),
        then initializes ray caster-specific fields.
        """
        # Initialize base fields (body_idx, pose, offsets)
        super()._init_data()

        # Initialize ray caster-specific fields
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

    def _update_buffers_impl(self, env_ids: Sequence[int] | None = None):
        """Update ray caster measurements for specified environments.
        
        Note: Sensor pose (pos_w, quat_w) is automatically updated by the base class.
        
        Args:
            env_ids: Indices of environments to update
        """
        # Transform ray directions to world frame (vectorized)
        # Expand local directions to all environments: (num_envs, num_rays, 3)
        ray_dirs = self._ray_directions_local.unsqueeze(0).expand(
            self.num_envs, self._num_rays, 3
        )

        # Expand quaternions for each ray: (num_envs, num_rays, 4)
        quat = self._data.quat_w.unsqueeze(1).expand(
            self.num_envs, self._num_rays, 4
        )

        # Reshape to (num_envs * num_rays, 3) and (num_envs * num_rays, 4)
        ray_dirs_flat = ray_dirs.reshape(-1, 3)
        quat_flat = quat.reshape(-1, 4)

        # Rotate and reshape back
        rotated = math_utils.quat_rotate(quat_flat, ray_dirs_flat)
        self._data.ray_directions_w[:] = rotated.reshape(self.num_envs, self._num_rays, 3)

        # Perform raycasting
        # TODO: Optimize to only raycast specified env_ids
        self._raycast()

        # Store distances in buffer for delay/history tracking
        self._buffer.append(self.sim.time, self._data.distances)

    def _raycast(self):
        """Perform raycasting for all environments using Warp."""
        # Get mesh from registry (handles caching and version checking internally)
        mesh_name = self.cfg.mesh_names[0]
        mesh = self.mesh_registry.get_warp_mesh(mesh_name)

        # Expand ray origins for all rays
        # Shape: (num_envs, num_rays, 3)
        ray_origins = self._data.pos_w.unsqueeze(1).expand(-1, self._num_rays, -1)

        # Ray directions already in world frame
        # Shape: (num_envs, num_rays, 3)
        ray_directions = self._data.ray_directions_w

        # Perform raycasting
        ray_hits, distances = raycast_mesh(
            ray_origins,
            ray_directions,
            mesh=mesh,
            max_dist=self.cfg.max_distance,
            return_distance=True,
        )

        # Check valid hits (finite distances)
        valid_hits = torch.isfinite(distances) & (distances >= self.cfg.min_distance)

        # Update distances
        self._data.distances[:] = torch.where(
            valid_hits,
            torch.clamp(distances, max=self.cfg.max_distance),
            torch.tensor(self.cfg.max_distance, device=self.device)
        )

        # Update hit mask
        self._data.hit_mask[:] = valid_hits

        # Update hit points if requested
        if self.cfg.return_hit_points:
            self._data.hit_points_w[:] = torch.where(
                valid_hits.unsqueeze(-1),
                ray_hits,
                ray_origins + ray_directions * self.cfg.max_distance
            )
