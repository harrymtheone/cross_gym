"""Abstract base class for sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch


class SensorConfigBase(ABC):
    """Base class for sensor configuration."""
    pass


class SensorBase(ABC):
    """Abstract base class for sensors.
    
    This defines a common interface for sensors across different simulators.
    """
    
    @abstractmethod
    def update(self):
        """Update sensor measurements."""
        pass
    
    @abstractmethod
    def get_data(self) -> torch.Tensor:
        """Get current sensor data.
        
        Returns:
            Sensor measurements
        """
        pass
    
    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of parallel environments."""
        pass

