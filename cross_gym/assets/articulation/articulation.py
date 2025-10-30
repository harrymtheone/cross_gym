"""Template articulation class for IsaacGym."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cross_core.base import ArticulationBase

if TYPE_CHECKING:
    from . import ArticulationCfg


class Articulation(ArticulationBase):
    """Template articulation class for IsaacGym backend.
    
    Provides common functionality and structure that specific implementations
    can build upon. This is a template that gym-specific articulations inherit from.
    """

    def __init__(self, cfg: ArticulationCfg):
        """Initialize articulation template.
        
        Args:
            cfg: Articulation configuration
        """
        super().__init__(cfg)

        # These will be set by child classes
        self._device: torch.device | None = None
        self._num_envs: int = 0
        self._num_dof: int = 0
        self._num_bodies: int = 0

    @property
    def num_dof(self) -> int:
        """Number of degrees of freedom."""
        return self._num_dof

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        return self._num_bodies

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    # Child classes must implement the actual control methods
    # using their specific gym API access
