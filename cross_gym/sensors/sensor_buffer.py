"""Buffer for sensor measurements with history, delay, and staleness tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from .sensor_base import SensorBase


class SensorBuffer:
    """Buffer for managing sensor measurements.
    
    Features:
    - History buffer: Store N past measurements
    - Delay simulation: Per-environment configurable delays
    - Staleness tracking: Mark when data needs updating
    """

    def __init__(self, sensor: SensorBase):
        """Initialize sensor buffer.
        
        Args:
            sensor: Parent sensor instance
        """
        self._cfg = sensor.cfg
        self._num_envs = sensor.num_envs
        self._device = sensor.device
        
        # Delay configuration
        self._delay_range = self._cfg.delay_range
        self._update_period = self._cfg.update_period
        
        # Current measurement
        self._measurement: torch.Tensor | None = None
        self._is_stale = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        
        # History buffer (if enabled)
        self._history_length = self._cfg.history_length
        if self._history_length > 0:
            self._history_buffer: list[torch.Tensor] = []
        
        # Delay buffer (if enabled)
        if self._delay_range is not None:
            # Calculate buffer length based on max delay and update period
            update_steps = max(1, int(self._update_period / sensor.sim.dt)) if self._update_period > 0 else 1
            max_delay_steps = int(self._delay_range[1] / (sensor.sim.dt * update_steps))
            self._delay_buffer_length = max_delay_steps + 2
            
            # Initialize delay buffer and delay indices per environment
            self._delay_buffer: list[torch.Tensor] = []
            self._delay_steps = torch.randint(
                int(self._delay_range[0] / (sensor.sim.dt * update_steps)),
                int(self._delay_range[1] / (sensor.sim.dt * update_steps)) + 1,
                (self._num_envs,),
                device=self._device
            )
    
    @property
    def data(self) -> torch.Tensor:
        """Get current measurement (with delay if enabled).
        
        Returns:
            Measurement tensor. Shape depends on sensor type.
        """
        if self._measurement is None:
            raise RuntimeError("Buffer has no data. Call append() first.")
        
        if self._delay_range is None:
            return self._measurement.clone()
        else:
            # Return delayed measurements per environment
            delayed_data = torch.zeros_like(self._measurement)
            for env_idx in range(self._num_envs):
                delay_idx = self._delay_steps[env_idx].item()
                if delay_idx < len(self._delay_buffer):
                    delayed_data[env_idx] = self._delay_buffer[-(delay_idx + 1)][env_idx]
                else:
                    # Not enough history, use oldest available
                    delayed_data[env_idx] = self._delay_buffer[0][env_idx]
            return delayed_data
    
    @property
    def is_stale(self) -> torch.Tensor:
        """Get staleness flag for each environment.
        
        Returns:
            Boolean tensor (num_envs,). True if data is stale.
        """
        return self._is_stale.clone()
    
    @property
    def history(self) -> list[torch.Tensor] | None:
        """Get history buffer if enabled.
        
        Returns:
            List of past measurements, or None if history disabled.
        """
        if self._history_length > 0:
            return self._history_buffer.copy()
        return None
    
    def append(self, data: torch.Tensor):
        """Add new measurement to buffer.
        
        Args:
            data: New measurement tensor
        """
        # Initialize measurement buffer on first append
        if self._measurement is None:
            self._measurement = torch.zeros_like(data)
        
        # Update current measurement
        self._measurement.copy_(data)
        self._is_stale.fill_(False)
        
        # Update history buffer
        if self._history_length > 0:
            self._history_buffer.append(data.clone())
            if len(self._history_buffer) > self._history_length:
                self._history_buffer.pop(0)
        
        # Update delay buffer
        if self._delay_range is not None:
            self._delay_buffer.append(data.clone())
            if len(self._delay_buffer) > self._delay_buffer_length:
                self._delay_buffer.pop(0)
    
    def step(self):
        """Mark data as stale (called at start of each step)."""
        self._is_stale.fill_(True)
    
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset buffer for specified environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        if self._measurement is None:
            return
        
        if env_ids is None:
            env_ids = slice(None)
        
        # Clear measurements for reset environments
        self._measurement[env_ids] = 0.0
        self._is_stale[env_ids] = True
        
        # Clear history
        if self._history_length > 0:
            for hist_data in self._history_buffer:
                hist_data[env_ids] = 0.0
        
        # Clear delay buffer
        if self._delay_range is not None:
            for delay_data in self._delay_buffer:
                delay_data[env_ids] = 0.0


__all__ = ["SensorBuffer"]

