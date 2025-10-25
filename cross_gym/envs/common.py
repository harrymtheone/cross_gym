"""Common types and utilities for environments."""

from __future__ import annotations

from typing import Any

import torch

# Type alias for vectorized environment step return
VecEnvStepReturn = tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
"""Type alias for the return of the step function of a vectorized environment.

Returns:
    A tuple containing:
    - observations (dict): Dictionary of observation tensors
    - rewards (torch.Tensor): Reward tensor (num_envs,)
    - terminated (torch.Tensor): Termination flags (num_envs,)
    - truncated (torch.Tensor): Truncation flags (num_envs,)
    - info (dict): Additional information
"""

VecEnvObs = dict[str, torch.Tensor]
"""Type alias for observations from a vectorized environment."""
