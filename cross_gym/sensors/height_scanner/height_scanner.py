"""Height scanner sensor implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cross_core.base import SensorBase
from cross_core.utils import math as math_utils
from . import HeightScannerData

if TYPE_CHECKING:
    from cross_gym.assets import Articulation
    from . import HeightScannerCfg


class HeightScanner(SensorBase):
    """Height scanner sensor.
    
    Reads height values from terrain heightmap at scan points around the sensor.
    Useful for:
    - Terrain-aware locomotion
    - Foothold selection
    - Elevation mapping
    - Obstacle detection
    
    The sensor:
    1. Generates scan points in sensor local frame (from pattern)
    2. Transforms points to world frame (with alignment mode)
    3. Queries terrain heightmap at those world positions
    4. Returns height measurements (absolute or relative)
    """

    cfg: HeightScannerCfg
    _data: HeightScannerData

    def __init__(
            self,
            cfg: HeightScannerCfg,
            articulation: Articulation,
            **kwargs,
    ):
        """Initialize height scanner sensor.
        
        Args:
            cfg: Sensor configuration
            articulation: Articulation to attach sensor to
            **kwargs: Additional arguments passed to parent
        """

        self._data = HeightScannerData()
        super().__init__(cfg, articulation, **kwargs)

    @property
    def data(self) -> HeightScannerData:
        """Get height scanner data.
        
        This triggers lazy evaluation: the sensor only computes
        if it's outdated based on update_period.
        """
        self._update_outdated_buffers()
        return self._data

    @property
    def num_points(self) -> int:
        """Number of scan points."""
        return self._scan_points_local.size(1)

    def _init_data(self):
        """Initialize sensor data container.
        
        Creates the HeightScannerData container, initializes base fields via super(),
        then initializes height scanner-specific fields.
        """
        # Initialize base fields (body_idx, pose, offsets)
        super()._init_data()

        # Expand scan points to all environments
        pattern_cfg = self.cfg.pattern_cfg
        self._scan_points_local = pattern_cfg.func(pattern_cfg, self.device)
        self._scan_points_local = self._scan_points_local.unsqueeze(0).repeat(self.num_envs, 1, 1)

        # Initialize height scanner-specific fields
        self._data.scan_points_w = torch.zeros(
            self.num_envs, self.num_points, 3,
            device=self.device
        )

        self._data.heights = torch.zeros(
            self.num_envs, self.num_points,
            device=self.device
        )

        self._data.heights_relative = torch.zeros(
            self.num_envs, self.num_points,
            device=self.device
        )

        # Get terrain heightmap from simulation
        if self.cfg.use_guidance_map:
            self._heightmap = torch.from_numpy(self.sim.terrain.height_map_guidance).to(self.device)
        else:
            self._heightmap = torch.from_numpy(self.sim.terrain.height_map).to(self.device)

        self._horizontal_scale = self.sim.terrain.cfg.horizontal_scale
        self._heightmap_size = self._heightmap.shape

    def _update_buffers_impl(self, env_ids: torch.Tensor | None = None):
        """Update height scanner measurements for specified environments.
        
        Args:
            env_ids: Environment indices to update. If None, update all.
        """
        # Resolve env_ids
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        # Step 1: Transform scan points to world frame based on alignment mode
        if self.cfg.alignment == "base":
            # Full 6DOF orientation
            # Flatten to (num_envs * num_points, 3) and (num_envs * num_points, 4)
            points_flat = self._scan_points_local[env_ids].reshape(-1, 3)
            quat_flat = self._data.quat_w[env_ids].repeat_interleave(self.num_points, dim=0)

            # Transform and reshape back
            points_rotated = math_utils.quat_rotate(
                points_flat, quat_flat
            ).reshape(num_envs, self.num_points, 3)

        elif self.cfg.alignment == "yaw":
            # Only yaw alignment (gravity-aligned, but rotated in XY)
            yaw = math_utils.quat_to_euler_xyz(self._data.quat_w[env_ids])[:, 2]

            # Flatten to (num_envs * num_points, 3) and (num_envs * num_points,)
            points_flat = self._scan_points_local[env_ids].reshape(-1, 3)
            yaw_flat = yaw.repeat_interleave(self.num_points)

            # Transform and reshape back
            points_rotated = math_utils.transform_by_yaw(
                points_flat, yaw_flat
            ).reshape(num_envs, self.num_points, 3)

        elif self.cfg.alignment == "gravity":
            # No rotation - keep gravity-aligned
            points_rotated = self._scan_points_local[env_ids]
        else:
            raise ValueError(f"Unknown alignment mode: {self.cfg.alignment}")

        # Add sensor position to get world coordinates
        self._data.scan_points_w[env_ids] = points_rotated + self._data.pos_w[env_ids].unsqueeze(1)

        # Step 2: Query heightmap at scan points
        self._query_heightmap(env_ids, num_envs)

        # Step 3: Compute relative heights if requested
        if self.cfg.measure_relative_height:
            sensor_z = self._data.pos_w[env_ids, 2:3]  # (num_envs, 1)
            self._data.heights_relative[env_ids] = self._data.heights[env_ids] - sensor_z
        else:
            self._data.heights_relative[env_ids] = self._data.heights[env_ids]

        # Store in buffer for delay/history
        self._buffer.append(self.sim.time, self._data.heights_relative)

    def _query_heightmap(self, env_ids, num_envs):
        """Query heightmap at scan points using specified interpolation.
        
        Args:
            env_ids: Environment indices
            num_envs: Number of environments to query
        """
        # Convert world positions to heightmap indices
        points = self._data.scan_points_w[env_ids] / self._horizontal_scale

        # Flatten for indexing: (num_envs * num_points)
        px = points[:, :, 0].flatten()
        py = points[:, :, 1].flatten()

        if self.cfg.clamp_to_terrain_bounds:
            # Clamp to valid heightmap range
            px = torch.clamp(px, 0, self._heightmap_size[0] - 1.001)
            py = torch.clamp(py, 0, self._heightmap_size[1] - 1.001)

        if self.cfg.interpolation == "nearest":
            # Nearest neighbor
            px_int = px.long()
            py_int = py.long()
            heights = self._heightmap[px_int, py_int]

        elif self.cfg.interpolation == "bilinear":
            # Bilinear interpolation
            px_floor = torch.floor(px).long()
            py_floor = torch.floor(py).long()
            px_ceil = torch.ceil(px).long()
            py_ceil = torch.ceil(py).long()

            # Clamp ceiling indices
            px_ceil = torch.clamp(px_ceil, 0, self._heightmap_size[0] - 1)
            py_ceil = torch.clamp(py_ceil, 0, self._heightmap_size[1] - 1)

            # Get 4 corner heights
            h00 = self._heightmap[px_floor, py_floor]
            h01 = self._heightmap[px_floor, py_ceil]
            h10 = self._heightmap[px_ceil, py_floor]
            h11 = self._heightmap[px_ceil, py_ceil]

            # Compute interpolation weights
            wx = (px - px_floor.float()).clamp(0, 1)
            wy = (py - py_floor.float()).clamp(0, 1)

            # Bilinear interpolation
            heights = (
                    h00 * (1 - wx) * (1 - wy) +
                    h01 * (1 - wx) * wy +
                    h10 * wx * (1 - wy) +
                    h11 * wx * wy
            )

        elif self.cfg.interpolation == "minimum":
            # Minimum of 4 neighbors (conservative for collision)
            px_floor = torch.floor(px).long()
            py_floor = torch.floor(py).long()
            px_ceil = (px_floor + 1).clamp(0, self._heightmap_size[0] - 1)
            py_ceil = (py_floor + 1).clamp(0, self._heightmap_size[1] - 1)

            h00 = self._heightmap[px_floor, py_floor]
            h01 = self._heightmap[px_floor, py_ceil]
            h10 = self._heightmap[px_ceil, py_floor]
            h11 = self._heightmap[px_ceil, py_ceil]

            heights = torch.minimum(torch.minimum(h00, h01), torch.minimum(h10, h11))
        else:
            raise ValueError(f"Unknown interpolation method: {self.cfg.interpolation}")

        # Reshape back to (num_envs, num_points)
        self._data.heights[env_ids] = heights.view(num_envs, self.num_points)
