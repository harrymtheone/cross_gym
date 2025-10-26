"""Base class and utilities for vectorized environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, Dict, Tuple

import torch

# ============================================================================
# Type Aliases
# ============================================================================

VecEnvObs = Dict[str, torch.Tensor]
"""Type alias for observations from a vectorized environment."""

VecEnvStepReturn = Tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]
"""Type alias for the return of the step function of a vectorized environment.

Returns:
    A tuple containing:
    - observations (dict): Dictionary of observation tensors
    - rewards (torch.Tensor): Reward tensor (num_envs,)
    - terminated (torch.Tensor): Termination flags (num_envs,)
    - truncated (torch.Tensor): Truncation flags (num_envs,)
    - info (dict): Additional information
"""


# ============================================================================
# Base Class
# ============================================================================

class VecEnv(ABC):
    """Base class for vectorized environments in Cross-Gym.
    
    This class extends gym.Env to support vectorized (parallel) environments.
    All Cross-Gym environments inherit from this class.
    
    Key additions over gym.Env:
    - num_envs: Number of parallel environments
    - device: Torch device
    - Vectorized observations/actions
    """

    is_vector_env: bool = True
    """Whether this is a vectorized environment."""

    def __init__(self, num_envs: int, device: torch.device):
        """Initialize vectorized environment base.
        
        Args:
            num_envs: Number of parallel environments
            device: Torch device
        """
        super().__init__()

        self._num_envs = num_envs
        self._device = device

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def device(self) -> torch.device:
        """Device on which environment tensors are stored."""
        return self._device

    @abstractmethod
    def step(self, action: dict[str, torch.Tensor]) -> VecEnvStepReturn:
        pass

    @abstractmethod
    def reset(self, env_ids: Sequence[int]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        pass
