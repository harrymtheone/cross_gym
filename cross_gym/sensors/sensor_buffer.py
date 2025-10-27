"""Buffer for sensor measurements with history, delay, and staleness tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from cross_gym.utils.buffers import CircularBuffer
from cross_gym.utils.math import torch_rand_float

if TYPE_CHECKING:
    from .sensor_base import SensorBase


class SensorBuffer:
    """Buffer for managing sensor measurements with realistic delay and history simulation.
    
    Design Philosophy:
    In real robotics, sensor data goes through: Measurement → Delay → History Buffer
    This mimics real systems where data arrives at callbacks after network/processing delay,
    then gets appended to history for temporal reasoning.
    
    Features:
    - Delay simulation: Per-environment configurable delays (mimics sensor latency)
    - History buffer: Store N past measurements (after delay is applied)
    - Raw data access: Get fresh measurements before delay/history
    
    Behavior Modes:
    1. No delay, no history: data = raw_data (instant passthrough)
    2. Delay, no history: data = delayed measurement
    3. No delay, with history: data = history buffer (fresh measurements)
    4. Delay + history: data = history buffer (delayed measurements)
    
    Convention:
    All data uses batch-first format: [num_envs, ...] or [num_envs, history_length, ...]
    This is consistent with modern PyTorch and easier to work with.
    """

    def __init__(self, sensor: SensorBase):
        """Initialize sensor buffer.
        
        Args:
            sensor: Parent sensor instance
        """
        self._cfg = sensor.cfg
        self._num_envs = sensor.num_envs
        self._device = sensor.device
        self._sim = sensor.sim

        # Delay simulation (only initialize if delay is enabled)
        self._delay_enabled = self._cfg.delay_range is not None

        if self._delay_enabled:
            min_delay, max_delay = self._cfg.delay_range

            # Per-environment delays (sampled once, stays fixed)
            self._env_delays = torch_rand_float(
                min_delay, max_delay, (self._num_envs,), self._device
            )

            # Calculate buffer size based on max delay and update period
            if sensor.cfg.update_period > 0:
                update_rate = sensor.cfg.update_period
            else:
                update_rate = sensor.sim.dt

            self._delay_buf_size = max(10, int(max_delay / update_rate) + 5)

            # Tensor-based circular buffer for delay simulation
            self._timestamps = torch.full(
                (self._delay_buf_size,),
                -1e6,  # Initialize with very old timestamps
                dtype=torch.float32,
                device=self._device
            )
            self._delay_buf = None  # Initialized on first append (need data shape)
            self._delay_buf_idx = 0  # Circular buffer pointer

        # Current fresh measurement (before delay)
        self._measurement: torch.Tensor | None = None

        # History buffer (if enabled) - stores past measurements AFTER delay
        self._history_enabled = self._cfg.history_length > 0
        if self._history_enabled:
            self._history_buffer = CircularBuffer(
                max_len=self._cfg.history_length,
                batch_size=self._num_envs,
                device=str(self._device)
            )
        else:
            self._history_buffer = None

        # Cache for current processed data (after delay, ready for history)
        self._processed_measurement: torch.Tensor | None = None

    @property
    def data(self) -> torch.Tensor:
        """Get sensor data after delay and history simulation.
        
        Returns:
            Processed sensor data in batch-first format:
            - If history enabled: [num_envs, history_length, ...] 
            - If only delay enabled: [num_envs, ...]
            - If neither: [num_envs, ...]
        
        Raises:
            RuntimeError: If buffer is not initialized or if enabled features are not properly set up.
        """
        if self._measurement is None:
            raise RuntimeError("Buffer has no data. Call append() first.")

        # If history enabled, return history buffer (already contains delayed data if delay enabled)
        if self._history_enabled:
            if self._history_buffer is None:
                raise RuntimeError("History is enabled but history buffer is not initialized.")
            return self._history_buffer.buffer  # Shape: [num_envs, history_length, ...]

        # If only delay enabled (no history), return current delayed measurement
        if self._delay_enabled:
            if self._processed_measurement is None:
                raise RuntimeError("Delay is enabled but processed measurement is not initialized. Call append() first.")
            return self._processed_measurement

        # No delay, no history: return fresh measurement
        return self._measurement

    @property
    def raw_data(self) -> torch.Tensor:
        """Get fresh measurement before any delay or history processing.
        
        Returns:
            Fresh measurement tensor in batch-first format: [num_envs, ...]
        """
        if self._measurement is None:
            raise RuntimeError("Buffer has no data. Call append() first.")
        return self._measurement

    def append(self, timestamp: float, data: torch.Tensor):
        """Add new measurement to buffer with realistic delay and history chaining.
        
        Important: This is called once per sensor update (at update_period rate), NOT every timestep.
        This ensures measurements are added to history exactly once, avoiding duplicates.
        
        Processing pipeline:
        1. Store fresh measurement (raw_data)
        2. Store in delay buffer if delay enabled  
        3. Retrieve current delayed measurement
        4. Append delayed measurement to history (happens ONCE per sensor update)
        
        Args:
            timestamp: Current simulation time
            data: New fresh measurement tensor
        """
        # Initialize measurement buffer on first append
        if self._measurement is None:
            self._measurement = torch.zeros_like(data)
            self._processed_measurement = torch.zeros_like(data)

        # Step 1: Store fresh measurement (raw data)
        self._measurement.copy_(data)

        # Step 2: Handle delay and history
        if self._delay_enabled:
            # Initialize delay buffer on first append (now we know the data shape)
            if self._delay_buf is None:
                self._delay_buf = torch.zeros(
                    (self._delay_buf_size, *data.shape), device=self._device, dtype=data.dtype
                )

            # Store fresh measurement in delay buffer
            self._timestamps[self._delay_buf_idx] = timestamp
            self._delay_buf[self._delay_buf_idx].copy_(data)
            self._delay_buf_idx = (self._delay_buf_idx + 1) % self._delay_buf_size

            # Get current delayed measurement for .data access
            delayed_data = self._get_delayed_measurement()
            self._processed_measurement.copy_(delayed_data)

            # Step 3: Append delayed measurement to history
            # This happens once per sensor update (at update_period rate)
            # Each environment gets its appropriately delayed measurement
            if self._history_enabled:
                self._history_buffer.append(delayed_data)
        else:
            # No delay: append fresh measurement to history immediately
            if self._history_enabled:
                self._history_buffer.append(data)

    def _get_delayed_measurement(self) -> torch.Tensor:
        """Vectorized retrieval of delayed measurements.
        
        This method uses torch.searchsorted to find the appropriate past measurements
        for each environment based on their individual delays.
        
        Returns:
            Delayed measurement tensor with per-environment delays applied.
        """
        if self._delay_buf is None:
            return self._measurement

        # Current simulation time
        current_time = self._sim.time

        # Target times for each environment (vectorized subtraction)
        target_times = current_time - self._env_delays  # Shape: [num_envs]

        # Find valid timestamps (not initialized to -1e6)
        valid_mask = self._timestamps > -1e5
        if not valid_mask.any():
            # No valid data yet, return current measurement
            return self._measurement

        # Get valid timestamps (sorted due to circular buffer insertion)
        valid_timestamps = self._timestamps[valid_mask]  # Shape: [num_valid]

        # Sort timestamps to ensure monotonicity for searchsorted
        sorted_indices = torch.argsort(valid_timestamps)
        sorted_timestamps = valid_timestamps[sorted_indices]

        # Use searchsorted to find insertion points for target times
        # This gives us the index of the first timestamp >= target_time
        # We want the index before that (last timestamp <= target_time)
        indices = torch.searchsorted(
            sorted_timestamps,
            target_times,
            right=False
        )  # Shape: [num_envs]

        # Clamp indices to valid range [0, num_valid-1]
        # If index is 0 and target_time < first timestamp, use first measurement
        # If index >= num_valid, use last measurement
        indices = torch.clamp(indices - 1, min=0, max=sorted_timestamps.numel() - 1)

        # Map back to original buffer indices
        valid_indices = torch.where(valid_mask)[0]
        original_indices = valid_indices[sorted_indices]
        buffer_indices = original_indices[indices]  # Shape: [num_envs]

        # Gather delayed data using advanced indexing
        # delay_buffer shape: [buffer_size, num_envs, *data_shape]
        # We want: for each env, get data from buffer_indices[env]
        env_range = torch.arange(self._num_envs, device=self._device)
        delayed_data = self._delay_buf[buffer_indices, env_range]

        return delayed_data

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset buffer for specified environments.
        
        Note: Delays are NOT resampled on reset - they stay fixed.
        
        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        if self._measurement is None:
            return

        if env_ids is None:
            env_ids = slice(None)

        # Clear fresh measurements for reset environments
        self._measurement[env_ids] = 0.0

        # Clear processed measurements
        if self._processed_measurement is not None:
            self._processed_measurement[env_ids] = 0.0

        # Clear history buffer using CircularBuffer's reset method
        if self._history_enabled:
            if self._history_buffer is None:
                raise RuntimeError("History is enabled but history buffer is not initialized.")
            self._history_buffer.reset(env_ids)

        # Clear delay buffer for reset environments
        if self._delay_enabled:
            if self._delay_buf is None:
                raise RuntimeError("Delay is enabled but delay buffer is not initialized.")
            self._delay_buf[:, env_ids] = 0.0

        # Note: We do NOT resample delays - they stay fixed throughout training
