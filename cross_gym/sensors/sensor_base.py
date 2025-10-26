"""Base class for all sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import torch

from cross_gym.utils import math as math_utils
from . import SensorBuffer

if TYPE_CHECKING:
    from cross_gym.assets import Articulation
    from cross_gym.sim import SimulationContext
    from . import SensorBaseCfg


class SensorBase(ABC):
    """Abstract base class for all sensors.
    
    Features:
    - Lazy evaluation: Updates only when data is accessed or forced
    - Configurable update rate: Can run at lower frequency than simulation
    - History buffering: Store past measurements
    - Delay simulation: Per-environment measurement delays
    - Attachment: Mount sensor to articulation bodies with offsets
    - Randomization: Per-environment offset/rotation randomization
    
    Subclasses must implement:
    - _update_buffers(): Update sensor measurements
    """

    def __init__(
            self,
            cfg: SensorBaseCfg,
            articulation: Articulation,
            sim: SimulationContext,
    ):
        """Initialize sensor base.
        
        Args:
            cfg: Sensor configuration
            articulation: Articulation to attach sensor to
            sim: Simulation context
        """
        # Validate configuration
        cfg.validate()  # noqa

        self.cfg = cfg
        self._articulation = articulation
        self.sim = sim

        # Find body index
        self._body_idx = self._find_body_index(cfg.body_name)

        # Initialize transform buffers
        self._init_transforms()

        # Create sensor buffer for history/delay management
        self._buffer = SensorBuffer(self)

        # Update tracking
        self._time_since_update = 0.0
        self._is_initialized = False

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._articulation.num_envs

    @property
    def device(self) -> torch.device:
        """Device for tensor operations."""
        return self._articulation.device

    @property
    def is_initialized(self) -> bool:
        """Whether sensor has been initialized."""
        return self._is_initialized

    @property
    def pos_w(self) -> torch.Tensor:
        """Sensor position in world frame. Shape: (num_envs, 3)."""
        return self._pos_w

    @property
    def quat_w(self) -> torch.Tensor:
        """Sensor orientation in world frame (w, x, y, z). Shape: (num_envs, 4)."""
        return self._quat_w

    @property
    @abstractmethod
    def data(self):
        """Sensor data container. Must be implemented by subclasses."""
        pass

    def update(self, dt: float, force: bool = False):
        """Update sensor measurements.
        
        Args:
            dt: Time step in seconds
            force: Force update regardless of update period
        """
        # Mark data as stale at start of step
        self._buffer.step()

        # Check if we should update based on update period
        self._time_since_update += dt
        should_update = force or (self._time_since_update >= self.cfg.update_period)

        if should_update:
            # Update sensor pose in world frame
            self._update_sensor_pose()

            # Update sensor measurements (implemented by subclasses)
            self._update_buffers(dt)

            # Mark as initialized after first update
            self._is_initialized = True

            # Reset update timer
            if self.cfg.update_period > 0:
                self._time_since_update = 0.0

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset sensor for specified environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        self._buffer.reset(env_ids)

        # Re-randomize offsets for reset environments
        if env_ids is not None and (self.cfg.offset_range is not None or self.cfg.rotation_range is not None):
            self._randomize_transforms(env_ids)

    @abstractmethod
    def _update_buffers(self, dt: float):
        """Update sensor measurements. Must be implemented by subclasses.
        
        Args:
            dt: Time step in seconds
        """
        pass

    def _find_body_index(self, body_name: str) -> int:
        """Find body index in articulation.
        
        Args:
            body_name: Name of body to attach to
            
        Returns:
            Body index
            
        Raises:
            ValueError: If body not found
        """
        try:
            return self._articulation.body_names.index(body_name)
        except ValueError:
            raise ValueError(
                f"Body '{body_name}' not found in articulation. "
                f"Available bodies: {self._articulation.body_names}"
            )

    def _init_transforms(self):
        """Initialize sensor transform buffers."""
        # Nominal offsets (from config)
        self._offset_pos = torch.tensor(
            self.cfg.offset,
            dtype=torch.float32,
            device=self.device
        ).repeat(self.num_envs, 1)

        self._offset_quat = math_utils.quat_from_euler_xyz(
            torch.deg2rad(torch.tensor(
                self.cfg.rotation,
                dtype=torch.float32,
                device=self.device
            ))
        ).repeat(self.num_envs, 1)

        # Actual offsets (with randomization)
        self._offset_pos_actual = self._offset_pos.clone()
        self._offset_quat_actual = self._offset_quat.clone()

        # Randomize if configured
        if self.cfg.offset_range is not None or self.cfg.rotation_range is not None:
            self._randomize_transforms(slice(None))

        # Sensor pose in world frame
        self._pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._quat_w[:, 0] = 1.0  # Initialize to identity (w=1)

    def _randomize_transforms(self, env_ids: Sequence[int] | slice):
        """Randomize sensor offsets for specified environments.
        
        Args:
            env_ids: Environment IDs to randomize
        """
        # Randomize position offset
        if self.cfg.offset_range is not None:
            for axis, (min_val, max_val) in enumerate(self.cfg.offset_range):
                if min_val != max_val:
                    self._offset_pos_actual[env_ids, axis] = (
                            self._offset_pos[env_ids, axis] +
                            torch.rand(
                                len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids),
                                device=self.device
                            ) * (max_val - min_val) + min_val
                    )

        # Randomize rotation offset
        if self.cfg.rotation_range is not None:
            euler = torch.zeros(
                len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids),
                3,
                device=self.device
            )
            base_euler = torch.deg2rad(torch.tensor(self.cfg.rotation, device=self.device))

            for axis, (min_val, max_val) in enumerate(self.cfg.rotation_range):
                if min_val != max_val:
                    euler[:, axis] = (
                            base_euler[axis] +
                            torch.deg2rad(
                                torch.rand(euler.shape[0], device=self.device) * (max_val - min_val) + min_val
                            )
                    )
                else:
                    euler[:, axis] = base_euler[axis]

            self._offset_quat_actual[env_ids] = math_utils.quat_from_euler_xyz(euler)

    def _update_sensor_pose(self):
        """Update sensor pose in world frame based on body pose."""
        # Get body pose in world frame
        body_pos_w = self._articulation.data.body_pos_w[:, self._body_idx]
        body_quat_w = self._articulation.data.body_quat_w[:, self._body_idx]

        # Transform sensor offset to world frame
        self._pos_w[:] = math_utils.quat_rotate(body_quat_w, self._offset_pos_actual) + body_pos_w
        self._quat_w[:] = math_utils.quat_mul(body_quat_w, self._offset_quat_actual)


__all__ = ["SensorBase"]
