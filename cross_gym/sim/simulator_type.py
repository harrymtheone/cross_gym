"""Simulator type enumeration."""

from enum import Enum


class SimulatorType(Enum):
    """Enumeration of supported simulators."""
    
    ISAACGYM = "isaacgym"
    GENESIS = "genesis"
    ISAACSIM = "isaacsim"
    
    def __str__(self):
        return self.value

