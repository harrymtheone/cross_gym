"""Buffer utilities for efficient data caching."""

from __future__ import annotations

import torch


class TimestampedBuffer:
    """Buffer that caches data with a timestamp.
    
    Used to avoid recomputing expensive transformations (e.g., velocities in base frame)
    multiple times per simulation step.
    
    Pattern (from IsaacLab):
        if buffer.timestamp < sim_timestamp:
            # Recompute
            buffer.data = expensive_computation()
            buffer.timestamp = sim_timestamp
        return buffer.data
    """

    def __init__(self):
        """Initialize empty buffer."""
        self.data: torch.Tensor = None
        self.timestamp: float = -1.0

    def reset(self):
        """Reset buffer."""
        self.data = None
        self.timestamp = -1.0
