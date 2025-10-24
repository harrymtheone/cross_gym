"""Common types and utilities for environments."""

from __future__ import annotations

import torch
from typing import Any, Dict, Tuple

# Type alias for vectorized environment step return
VecEnvStepReturn = Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]
"""Type alias for the return of the step function of a vectorized environment.

Returns:
    A tuple containing:
    - observations (dict): Dictionary of observation tensors
    - rewards (torch.Tensor): Reward tensor (num_envs,)
    - terminated (torch.Tensor): Termination flags (num_envs,)
    - truncated (torch.Tensor): Truncation flags (num_envs,)
    - info (dict): Additional information
"""

VecEnvObs = Dict[str, torch.Tensor]
"""Type alias for observations from a vectorized environment."""
