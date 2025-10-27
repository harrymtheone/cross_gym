"""Base class for all sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import torch

from cross_gym.utils import math as math_utils
from . import SensorBuffer, SensorBaseData

if TYPE_CHECKING:
    from cross_gym.assets import Articulation
    from cross_gym.sim import SimulationContext
    from . import SensorBaseCfg


class SensorBase(ABC):
    """Abstract base class for all sensors.
    
    This implementation follows IsaacLab's sensor update mechanism with
    improved delay simulation using vectorized operations.
    
    Subclasses must implement:
    - _update_buffers_impl(env_ids): Update sensor measurements for given environments
    - data property: Return sensor data (calls _update_outdated_buffers())
    """

    cfg: SensorBaseCfg
    _data: SensorBaseData = None

    def __init__(
            self,
            cfg: SensorBaseCfg,
            articulation: Articulation,
            **kwargs,
    ):
        """Initialize sensor base.
        
        Args:
            cfg: Sensor configuration
            articulation: Articulation to attach sensor to
            **kwargs: Additional arguments (e.g., mesh_registry for RayCaster)
        """
        # Validate configuration
        cfg.validate()  # noqa
        self.cfg = cfg
        self._articulation = articulation
        self.sim = SimulationContext.instance()

        # Initialize sensor data container
        if self._data is None:
            raise RuntimeError("self._data must be initialized")
        self._init_data()

        # Create sensor buffer for history/delay management
        self._buffer = SensorBuffer(self)

        # Per-environment update tracking (IsaacLab-style)
        self._is_outdated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._timestamp = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._timestamp_last_update = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Initialization flag
        self._initialized = False

        # Debug visualization flag
        self._is_visualizing = False

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._articulation.num_envs

    @property
    def device(self) -> torch.device:
        """Device for tensor operations."""
        return self._articulation.device

    @property
    def initialized(self) -> bool:
        """Whether sensor has been initialized."""
        return self._initialized

    @property
    @abstractmethod
    def data(self) -> SensorBaseData:
        """Sensor data container. Must be implemented by subclasses.
        
        The data container should inherit from SensorBaseData which provides
        pos_w and quat_w fields.
        
        Subclasses should implement this as:
        
        @property
        def data(self) -> MySensorData:
            self._update_outdated_buffers()
            return self._data
        """
        pass

    def update(self, dt: float, force_recompute: bool = False):
        """Update sensor state and check if recomputation is needed.
        
        This method is called every simulation step. It updates timestamps and
        determines which environments are outdated. Actual sensor computation
        only happens if:
        1. force_recompute=True (eager mode)
        2. Debug visualization is enabled
        3. History tracking is enabled (needs consistent updates)
        4. Or when data is accessed (lazy mode, in _update_outdated_buffers)
        
        Args:
            dt: Time step in seconds
            force_recompute: If True, force update regardless of update period
        """
        # Update timestamps for all environments
        self._timestamp.add_(dt)

        # Mark environments as outdated if enough time has passed
        # Add small epsilon (1e-6) for floating point comparison
        self._is_outdated[:] |= torch.ge(
            self._timestamp - self._timestamp_last_update + 1e-6,
            self.cfg.update_period
        )

        # Compute immediately if:
        # 1. Force recompute (eager mode from InteractiveScene)
        # 2. Debug visualization enabled (need fresh data for vis)
        # 3. History tracking enabled (need consistent history buffer)
        if force_recompute or self._is_visualizing or self.cfg.history_length > 0:
            self._update_outdated_buffers()

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset sensor for specified environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        # Resolve env_ids
        if env_ids is None:
            env_ids = slice(None)

        # Reset timestamps for the sensors
        self._timestamp[env_ids] = 0.0
        self._timestamp_last_update[env_ids] = 0.0

        # Set all reset sensors to outdated so they update when accessed
        self._is_outdated[env_ids] = True

        # Reset buffer
        self._buffer.reset(env_ids)

        # Re-randomize offsets for reset environments (if configured)
        if self._has_randomization():
            self._apply_randomization(env_ids)

    def _update_outdated_buffers(self):
        """Update sensor data for environments that are outdated.
        
        This is the core of lazy evaluation: only compute sensor data for
        environments whose timestamps indicate they need updating.
        """
        # Find which environments are outdated
        if not torch.any(self._is_outdated):
            return

        outdated_env_ids = self._is_outdated.nonzero(as_tuple=False).squeeze(-1)

        # Update sensor pose in world frame
        self._update_sensor_pose()

        # Call subclass-specific update (only for outdated environments)
        # Note: Subclass should call self._buffer.append() if using delay/history
        self._update_buffers_impl(outdated_env_ids)

        # Update timestamps for the environments that were just updated
        self._timestamp_last_update[outdated_env_ids] = self._timestamp[outdated_env_ids]

        # Clear outdated flags
        self._is_outdated[outdated_env_ids] = False

        # Mark as initialized after first update
        if not self._initialized:
            self._initialized = True

    def _init_data(self):
        """Initialize base sensor data fields.
        
        This base implementation initializes common fields that all sensors have:
        - body_idx: Index of the body the sensor is attached to
        - pos_w, quat_w: Sensor pose in world frame
        - offset_pos, offset_quat: Nominal offsets from config
        - offset_pos_sim, offset_quat_sim: Actual offsets (will be randomized)
        
        Subclasses must override this method and:
        1. Create their specific data container
        2. Call super()._init_data() to initialize base fields
        3. Initialize sensor-specific fields
        
        Example:
            def _init_data(self):
                # Create sensor-specific data container
                self._data = ImuData()
                
                # Initialize base fields (body_idx, pose, offsets)
                super()._init_data()
                
                # Initialize sensor-specific fields
                self._data.lin_acc_b = torch.zeros(self.num_envs, 3, device=self.device)
                self._data.ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        """
        # Find and store body index
        self._data.body_idx = self._find_body_index(self.cfg.body_name)

        # Initialize sensor pose in world frame
        self._data.pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._data.quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._data.quat_w[:, 0] = 1.0  # Identity quaternion (w=1)

        # Initialize nominal offsets from config
        self._data.offset_pos = torch.tensor(
            self.cfg.offset, dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)

        self._data.offset_quat = math_utils.quat_from_euler_xyz(
            torch.deg2rad(torch.tensor(
                self.cfg.rotation, dtype=torch.float, device=self.device
            ))
        ).repeat(self.num_envs, 1)

        # Initialize actual offsets (will be randomized if configured)
        self._data.offset_pos_sim = self._data.offset_pos.clone()
        self._data.offset_quat_sim = self._data.offset_quat.clone()

        # Apply randomization if configured
        if self._has_randomization():
            self._apply_randomization()

    @abstractmethod
    def _update_buffers_impl(self, env_ids: Sequence[int] | torch.Tensor):
        """Update sensor measurements for specified environments.
        
        This method must be implemented by subclasses to compute sensor-specific
        measurements. It will only be called for environments that are outdated.
        
        Args:
            env_ids: Indices of environments to update. Can be a sequence or tensor.
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

    def _apply_randomization(self, env_ids: Sequence[int] | None = None):
        """Apply randomization to sensor offsets for specified environments.
        
        This can be called during initialization (for all envs) or during reset
        (for specific envs).
        
        Args:
            env_ids: Environment IDs to randomize. If None, randomize all environments.
        """
        if env_ids is None:
            env_ids = slice(None)
            num_env_ids = self.num_envs
        else:
            num_env_ids = len(env_ids)

        # Randomize position offset
        offset_cfg = self.cfg.offset_range
        for axis, axis_range in enumerate([offset_cfg.x, offset_cfg.y, offset_cfg.z]):
            if axis_range is None:
                continue

            min_val, max_val = axis_range
            if min_val != max_val:
                # Sample random offset for this axis
                random_offset = math_utils.torch_rand_float(min_val, max_val, num_env_ids, self.device)
                self._data.offset_pos_sim[env_ids, axis] = self._data.offset_pos[env_ids, axis] + random_offset

        # Randomize rotation offset
        euler = torch.zeros(num_env_ids, 3, device=self.device)
        base_euler = torch.deg2rad(torch.tensor(self.cfg.rotation, device=self.device))

        # Check each axis (roll=0, pitch=1, yaw=2)
        rotation_cfg = self.cfg.rotation_range
        for axis, axis_range in enumerate([rotation_cfg.roll, rotation_cfg.pitch, rotation_cfg.yaw]):
            if axis_range is None:
                continue

            min_val, max_val = axis_range
            if min_val == max_val:
                # No randomization for this axis - use base value
                euler[:, axis] = base_euler[axis]
                continue

            # Sample random rotation for this axis (in degrees, then convert to radians)
            random_angle = math_utils.torch_rand_float(min_val, max_val, num_env_ids, self.device)
            euler[:, axis] = base_euler[axis] + torch.deg2rad(random_angle)

        self._data.offset_quat_sim[env_ids] = math_utils.quat_from_euler_xyz(euler)

    def _update_sensor_pose(self):
        """Update sensor pose in world frame based on body pose.
        
        Uses the body_idx and offset transforms from self._data to compute
        the sensor pose in world frame.
        """
        # Get body pose in world frame
        body_pos_w = self._articulation.data.body_pos_w[:, self._data.body_idx]
        body_quat_w = self._articulation.data.body_quat_w[:, self._data.body_idx]

        # Transform sensor offset to world frame
        self._data.pos_w[:] = math_utils.quat_rotate(body_quat_w, self._data.offset_pos_sim) + body_pos_w
        self._data.quat_w[:] = math_utils.quat_mul(body_quat_w, self._data.offset_quat_sim)

    def _has_randomization(self) -> bool:
        """Check if any randomization is configured.
        
        Returns:
            True if offset or rotation randomization is enabled for any axis.
        """
        # Check if ANY position axis has randomization
        cfg = self.cfg.offset_range
        rand_offset = any([cfg.x, cfg.y, cfg.z])

        # Check if ANY rotation axis has randomization
        cfg = self.cfg.rotation_range
        rand_rotation = any([cfg.roll, cfg.pitch, cfg.yaw])

        return rand_offset or rand_rotation
