"""Buffer for sensor measurements with history, delay, and staleness tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from cross_gym.utils.math import torch_rand_float

if TYPE_CHECKING:
    from .sensor_base import SensorBase


class SensorBuffer:
    """Buffer for managing sensor measurements.
    
    Features:
    - History buffer: Store N past measurements
    - Delay simulation: Per-environment configurable delays using tensor buffers
    - Fully vectorized delay retrieval using torch.searchsorted
    
    The delay mechanism works by storing measurements in circular tensor buffers
    with timestamps, then using vectorized operations to retrieve measurements
    from (current_time - delay) ago for each environment.
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

        # Per-environment delays (sampled once, stays fixed)
        if self._cfg.delay_range is None:
            self._env_delays = None
            self._buffer_size = 0
            self._timestamps = None
            self._data_buffer = None
            self._buffer_idx = 0
        else:
            min_delay, max_delay = self._cfg.delay_range
            self._env_delays = torch_rand_float(
                min_delay, max_delay, (self._num_envs,), self._device
            )

            # Calculate buffer size based on max delay and update period
            if sensor.cfg.update_period > 0:
                update_rate = sensor.cfg.update_period
            else:
                update_rate = sensor.sim.dt
            self._buffer_size = max(10, int(max_delay / update_rate) + 5)

            # Tensor-based circular buffer for delay simulation
            self._timestamps = torch.full(
                (self._buffer_size,),
                -1e6,  # Initialize with very old timestamps
                dtype=torch.float32,
                device=self._device
            )
            self._data_buffer = None  # Initialized on first append (need data shape)
            self._buffer_idx = 0  # Circular buffer pointer

        # Current measurement (no delay)
        self._measurement: torch.Tensor | None = None

        # History buffer (if enabled) - stores past measurements at update rate
        self._history_length = self._cfg.history_length
        if self._history_length > 0:
            self._history_buffer = None  # Initialized on first append
            self._history_idx = 0

    @property
    def data(self) -> torch.Tensor:
        """Get current measurement (with delay if enabled).
        
        Returns:
            Measurement tensor with per-environment delays applied.
        """
        if self._measurement is None:
            raise RuntimeError("Buffer has no data. Call append() first.")

        # If no delay configured, return current measurement
        if self._env_delays is None:
            return self._measurement

        # Return delayed measurements (fully vectorized)
        return self._get_delayed_measurement()

    @property
    def history(self) -> torch.Tensor | None:
        """Get history buffer if enabled.
        
        Returns:
            History tensor of shape (history_length, num_envs, *data_shape), or None if disabled.
        """
        if self._history_length > 0 and self._history_buffer is not None:
            return self._history_buffer
        return None

    def append(self, timestamp: float, data: torch.Tensor):
        """Add new measurement to buffer.
        
        Args:
            timestamp: Current simulation time
            data: New measurement tensor
        """
        # Initialize measurement buffer on first append
        if self._measurement is None:
            self._measurement = torch.zeros_like(data)

        # Update current measurement (no delay)
        self._measurement.copy_(data)

        # Update delay buffer if enabled
        if self._env_delays is not None:
            # Initialize data buffer on first append (now we know the data shape)
            if self._data_buffer is None:
                self._data_buffer = torch.zeros(
                    (self._buffer_size, *data.shape),
                    device=self._device,
                    dtype=data.dtype
                )

            # Store in circular buffer
            self._timestamps[self._buffer_idx] = timestamp
            self._data_buffer[self._buffer_idx].copy_(data)
            self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size

        # Update history buffer if enabled
        if self._history_length > 0:
            if self._history_buffer is None:
                self._history_buffer = torch.zeros(
                    (self._history_length, *data.shape),
                    device=self._device,
                    dtype=data.dtype
                )

            self._history_buffer[self._history_idx].copy_(data)
            self._history_idx = (self._history_idx + 1) % self._history_length

    def _get_delayed_measurement(self) -> torch.Tensor:
        """Vectorized retrieval of delayed measurements.
        
        This method uses torch.searchsorted to find the appropriate past measurements
        for each environment based on their individual delays. This is much faster
        than iterating through environments.
        
        Returns:
            Delayed measurement tensor with per-environment delays applied.
        """
        if self._data_buffer is None:
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
        # data_buffer shape: [buffer_size, num_envs, *data_shape]
        # We want: for each env, get data from buffer_indices[env]
        env_range = torch.arange(self._num_envs, device=self._device)
        delayed_data = self._data_buffer[buffer_indices, env_range]

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

        # Clear measurements for reset environments
        self._measurement[env_ids] = 0.0

        # Clear history
        if self._history_length > 0 and self._history_buffer is not None:
            self._history_buffer[:, env_ids] = 0.0

        # Clear delay buffer for reset environments
        if self._env_delays is not None and self._data_buffer is not None:
            self._data_buffer[:, env_ids] = 0.0

        # Note: We do NOT resample delays - they stay fixed throughout training
