"""Articulation assets for IsaacGym."""

from .articulation_cfg import ArticulationCfg
from .isaacgym_articulation_view import IsaacGymArticulationView

# Note: Articulation and ArticulationData are part of the old structure
# and need to be migrated or simplified for the new architecture

__all__ = [
    "ArticulationCfg",
    "IsaacGymArticulationView",
]
