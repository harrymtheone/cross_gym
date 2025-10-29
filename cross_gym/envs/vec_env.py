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
    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of parallel environments."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device on which environment tensors are stored."""
        pass

    @abstractmethod
    def step(self, action: dict[str, torch.Tensor]) -> VecEnvStepReturn:
        pass

    @abstractmethod
    def reset(self, env_ids: Sequence[int]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        pass
