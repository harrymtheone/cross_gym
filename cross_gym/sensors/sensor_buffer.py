"""Buffer implementations for sensor measurements with delay and history simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cross_gym.utils import math as math_utils
from cross_gym.utils.buffers import CircularBuffer

if TYPE_CHECKING:
    from . import SensorBase


class DelayBuffer:
    """Simple delay buffer for sensors without history.
    
    Uses efficient lookback: retrieves measurement from (current_time - delay).
    No duplicate tracking needed since we're not building a history.
    """

    def __init__(self, sensor: SensorBase):
        """Initialize delay buffer.
        
        Args:
            sensor: Parent sensor instance
        """
        self._sim = sensor.sim
        self._num_envs = sensor.num_envs
        self._device = sensor.device

        self._ALL_ENVS = torch.arange(self._num_envs, device=sensor.device)

        # Per-environment delays (sampled once, stays fixed)
        min_delay, max_delay = sensor.cfg.delay_range
        self._env_delays = torch_rand_float(
            min_delay, max_delay, (self._num_envs,), self._device
        )

        # Calculate buffer size based on max delay and update period
        update_rate = sensor.cfg.update_period if sensor.cfg.update_period > 0 else sensor.sim.dt
        buffer_size = max(10, int(max_delay / update_rate) + 5)

        # Circular buffer for storing measurements with timestamps
        self._timestamps = torch.full(
            (buffer_size,), -1e6, dtype=torch.float32, device=self._device
        )
        self._data_buffer: torch.Tensor | None = None  # [buffer_size, num_envs, *data_shape]
        self._buffer_idx = 0

        # Current measurement (most recent)
        self._current_data: torch.Tensor | None = None

    @property
    def data(self) -> torch.Tensor:
        """Get delayed measurement (lookback from current_time - delay)."""
        return self._get_delayed_measurement()

    @property
    def raw_data(self) -> torch.Tensor:
        """Get fresh measurement without delay."""
        if self._current_data is None:
            raise RuntimeError("Buffer has no data. Call append() first.")
        return self._current_data

    def update(self):
        """Update buffer state (no-op for delay-only buffer)."""
        pass

    def append(self, timestamp: float, data: torch.Tensor):
        """Store fresh measurement in circular buffer.
        
        Args:
            timestamp: Simulation time when measurement was taken
            data: Fresh measurement data
        """
        # Initialize buffers on first append
        if self._data_buffer is None:
            self._data_buffer = torch.zeros(
                (len(self._timestamps), *data.shape),
                device=self._device,
                dtype=data.dtype
            )
            self._current_data = torch.zeros_like(data)

        # Store fresh measurement
        self._current_data.copy_(data)

        # Add to circular buffer with timestamp
        self._timestamps[self._buffer_idx] = timestamp
        self._data_buffer[self._buffer_idx].copy_(data)
        self._buffer_idx = (self._buffer_idx + 1) % len(self._timestamps)

    def _get_delayed_measurement(self) -> torch.Tensor:
        """Retrieve measurement from (current_time - delay) using vectorized lookback.
        
        Returns:
            Delayed measurement for each environment
        """
        current_time = self._sim.time
        target_times = current_time - self._env_delays  # [num_envs]

        # Find valid timestamps
        valid_mask = self._timestamps > -1e5
        if not valid_mask.any():
            return self._current_data

        # Sort timestamps and use searchsorted for efficient lookback
        valid_timestamps = self._timestamps[valid_mask]
        sorted_indices = torch.argsort(valid_timestamps)
        sorted_timestamps = valid_timestamps[sorted_indices]

        # Find closest past measurement for each environment
        indices = torch.searchsorted(sorted_timestamps, target_times, right=False)
        indices = torch.clamp(indices - 1, min=0, max=sorted_timestamps.numel() - 1)

        # Map back to buffer indices and gather data
        valid_indices = torch.where(valid_mask)[0]
        buffer_indices = valid_indices[sorted_indices[indices]]

        return self._data_buffer[buffer_indices, self._ALL_ENVS]

    def reset(self, env_ids=None):
        """Reset buffer for specified environments."""
        if self._current_data is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._current_data[env_ids] = 0.0
        if self._data_buffer is not None:
            self._data_buffer[:, env_ids] = 0.0


class HistoryBuffer:
    """Simple history buffer for sensors without delay.
    
    Thin wrapper around CircularBuffer - stores past N measurements in chronological order.
    """

    def __init__(self, sensor: SensorBase):
        """Initialize history buffer.
        
        Args:
            sensor: Parent sensor instance
        """
        self._circular_buffer = CircularBuffer(
            max_len=sensor.cfg.history_length,
            batch_size=sensor.num_envs,
            device=str(sensor.device)
        )

    @property
    def data(self) -> torch.Tensor:
        """Get ordered history (oldest to newest).
        
        Returns:
            [num_envs, history_len, *data_shape]
        """
        return self._circular_buffer.buffer

    @property
    def raw_data(self) -> torch.Tensor:
        """Get most recent measurement.
        
        Returns:
            [num_envs, *data_shape]
        """
        # Index 0 gets the most recent entry (LIFO)
        zeros = torch.zeros(self._circular_buffer.batch_size, dtype=torch.long, device=self._circular_buffer.device)
        return self._circular_buffer[zeros]

    def update(self):
        """Update buffer state (no-op for history-only buffer)."""
        pass

    def append(self, timestamp: float, data: torch.Tensor):
        """Append measurement to history.
        
        Args:
            timestamp: Simulation time (unused for history-only)
            data: Measurement data to append
        """
        self._circular_buffer.append(data)

    def reset(self, env_ids=None):
        """Reset buffer for specified environments."""
        self._circular_buffer.reset(env_ids)


class DelayedHistoryBuffer:
    """Buffer for sensors with both delay AND history.
    
    Uses ready queue design to prevent duplicate measurements in history.
    Each measurement is queued with time_ready = timestamp + delay, then
    appended to history exactly once when time_ready <= current_time.
    """

    def __init__(self, sensor: SensorBase):
        """Initialize delayed history buffer.
        
        Args:
            sensor: Parent sensor instance
        """
        self._sim = sensor.sim
        self._num_envs = sensor.num_envs
        self._device = sensor.device

        # Per-environment delays
        min_delay, max_delay = sensor.cfg.delay_range
        self._env_delays = math_utils.torch_rand_float_1d(
            min_delay, max_delay, self._num_envs, self._device
        )

        # Ready queue size
        update_rate = sensor.cfg.update_period if sensor.cfg.update_period > 0 else sensor.sim.dt
        self._queue_size = max(10, int(max_delay / update_rate) + 5)

        # Ready queue: [queue_size, num_envs, *data_shape]
        self._delay_buf: torch.Tensor | None = None
        self._ready_times = torch.full(
            (self._queue_size, self._num_envs),
            float('inf'),
            dtype=torch.float32,
            device=self._device
        )
        self._queue_idx = 0

        # History buffer
        self._history_len = sensor.cfg.history_length
        self._history_buf: torch.Tensor | None = None
        self._history_idx = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._history_count = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        # Current fresh measurement
        self._current_data: torch.Tensor | None = None

    @property
    def data(self) -> torch.Tensor:
        """Get ordered history with delayed measurements."""
        self.update()  # Safety check
        if self._history_buf is None:
            raise RuntimeError("Buffer has no data. Call append() first.")
        return self._get_ordered_history()

    @property
    def raw_data(self) -> torch.Tensor:
        """Get fresh measurement without delay."""
        if self._current_data is None:
            raise RuntimeError("Buffer has no data. Call append() first.")
        return self._current_data

    def update(self):
        """Check and append ready measurements to history."""
        self._check_and_append_ready()

    def append(self, timestamp: float, data: torch.Tensor):
        """Queue measurement with its time_ready.
        
        Args:
            timestamp: Time when measurement was taken
            data: Fresh measurement data
        """
        # Initialize buffers on first append
        if self._delay_buf is None:
            self._delay_buf = torch.zeros(
                (self._queue_size, *data.shape),
                device=self._device,
                dtype=data.dtype
            )
            self._current_data = torch.zeros_like(data)

        # Store fresh data
        self._current_data.copy_(data)

        # Queue with time_ready = timestamp + delay
        time_ready = timestamp + self._env_delays
        self._delay_buf[self._queue_idx] = data
        self._ready_times[self._queue_idx] = time_ready
        self._queue_idx = (self._queue_idx + 1) % self._queue_size

        # Check for ready measurements
        self._check_and_append_ready()

    def _check_and_append_ready(self):
        """Check ready queue and append measurements to history."""
        current_time = self._sim.time

        # Find ready measurements
        ready_mask = current_time >= self._ready_times  # [queue_size, num_envs]

        if not ready_mask.any():
            return

        # Sort by time_ready for chronological appending
        times_for_sort = torch.where(
            ready_mask,
            self._ready_times,
            torch.tensor(float('inf'), device=self._device)
        )

        sorted_indices = torch.argsort(times_for_sort, dim=0)
        sorted_times = torch.gather(times_for_sort, 0, sorted_indices)
        valid_mask = sorted_times < float('inf')
        num_ready_per_env = valid_mask.sum(dim=0)

        if num_ready_per_env.max() == 0:
            return

        # Batch append to history
        self._batch_append_to_history(sorted_indices, valid_mask, num_ready_per_env)

        # Mark as consumed
        self._ready_times[ready_mask] = float('inf')

    def _batch_append_to_history(self, sorted_indices, valid_mask, num_ready_per_env):
        """Vectorized batch append to history."""
        # Initialize history buffer if needed
        if self._history_buf is None:
            first_valid = valid_mask.any(dim=1).nonzero()[0].item()
            first_env = valid_mask[first_valid].nonzero()[0].item()
            sample_data = self._delay_buf[sorted_indices[first_valid, first_env], first_env]
            self._history_buf = torch.zeros(
                (self._history_len, self._num_envs, *sample_data.shape),
                device=self._device,
                dtype=sample_data.dtype
            )

        # Compute write positions
        position_offsets = torch.arange(self._queue_size, device=self._device).unsqueeze(1).expand(-1, self._num_envs)
        position_offsets = torch.where(valid_mask, position_offsets, torch.tensor(-1, device=self._device))
        write_positions = (self._history_idx.unsqueeze(0) + position_offsets) % self._history_len

        # Flatten and gather valid entries
        valid_flat = valid_mask.flatten().nonzero(as_tuple=True)[0]

        if valid_flat.numel() > 0:
            sorted_flat = sorted_indices.flatten()
            write_pos_flat = write_positions.flatten()
            env_flat = torch.arange(self._queue_size * self._num_envs, device=self._device) % self._num_envs

            valid_queue = sorted_flat[valid_flat]
            valid_write = write_pos_flat[valid_flat]
            valid_env = env_flat[valid_flat]

            # Write all at once
            self._history_buf[valid_write, valid_env] = self._delay_buf[valid_queue, valid_env]

        # Update pointers
        self._history_idx = (self._history_idx + num_ready_per_env) % self._history_len
        self._history_count = torch.minimum(
            self._history_count + num_ready_per_env,
            torch.tensor(self._history_len, device=self._device)
        )

    def _get_ordered_history(self) -> torch.Tensor:
        """Get history in chronological order."""
        base_indices = torch.arange(self._history_len, device=self._device)
        shifted_indices = (base_indices.unsqueeze(0) + self._history_idx.unsqueeze(1)) % self._history_len

        buf = self._history_buf.transpose(0, 1)
        gather_indices = shifted_indices

        if buf.dim() > 2:
            for _ in range(buf.dim() - 2):
                gather_indices = gather_indices.unsqueeze(-1)
            gather_indices = gather_indices.expand_as(buf)

        return torch.gather(buf, dim=1, index=gather_indices)

    def reset(self, env_ids=None):
        """Reset buffer."""
        if env_ids is None:
            env_ids = slice(None)

        if self._current_data is not None:
            self._current_data[env_ids] = 0.0
        if self._delay_buf is not None:
            self._delay_buf[:, env_ids] = 0.0
        if self._history_buf is not None:
            self._history_buf[:, env_ids] = 0.0

        self._ready_times[:, env_ids] = float('inf')
        self._history_idx[env_ids] = 0
        self._history_count[env_ids] = 0


class PassthroughBuffer:
    """Simple buffer for sensors without delay or history.

    Just stores and returns the current measurement.
    """

    def __init__(self, sensor: SensorBase):
        """Initialize passthrough buffer.

        Args:
            sensor: Parent sensor instance
        """
        self._data: torch.Tensor | None = None

    @property
    def data(self) -> torch.Tensor:
        """Get current measurement."""
        if self._data is None:
            raise RuntimeError("Buffer has no data. Call append() first.")
        return self._data

    @property
    def raw_data(self) -> torch.Tensor:
        """Get current measurement (same as data)."""
        return self.data

    def update(self):
        """Update buffer state (no-op)."""
        pass

    def append(self, timestamp: float, data: torch.Tensor):
        """Store measurement.

        Args:
            timestamp: Simulation time (unused)
            data: Measurement data
        """
        if self._data is None:
            self._data = torch.zeros_like(data)
        self._data.copy_(data)

    def reset(self, env_ids=None):
        """Reset buffer."""
        if self._data is not None:
            if env_ids is None:
                env_ids = slice(None)
            self._data[env_ids] = 0.0


class SensorBuffer:
    """Factory that creates appropriate buffer based on sensor configuration.

    Automatically selects:
    - DelayBuffer: if delay_range set, history_length = 0
    - HistoryBuffer: if history_length > 0, delay_range = None
    - DelayedHistoryBuffer: if both delay_range and history_length set
    - PassthroughBuffer: if neither set (just stores current measurement)
    """

    def __new__(cls, sensor: SensorBase):
        """Create appropriate buffer type based on sensor config."""
        has_delay = sensor.cfg.delay_range is not None
        has_history = sensor.cfg.history_length > 0

        if has_delay and has_history:
            return DelayedHistoryBuffer(sensor)
        elif has_delay:
            return DelayBuffer(sensor)
        elif has_history:
            return HistoryBuffer(sensor)
        else:
            # No delay, no history: simple passthrough
            return PassthroughBuffer(sensor)
