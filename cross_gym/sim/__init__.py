"""Simulator abstraction layer."""

from .simulation_context import SimulationContext
from .sim_cfg_base import SimCfgBase

# Import simulator-specific configs when available
try:
    from .isaacgym import IsaacGymCfg, PhysxCfg, IsaacGymContext
except ImportError:
    IsaacGymCfg = None
    PhysxCfg = None
    IsaacGymContext = None

try:
    from .genesis import GenesisCfg
except ImportError:
    GenesisCfg = None

__all__ = [
    "SimulationContext",
    "SimCfgBase",
]

# Add simulator-specific exports if available
if IsaacGymCfg is not None:
    __all__.extend(["IsaacGymCfg", "PhysxCfg", "IsaacGymContext"])

if GenesisCfg is not None:
    __all__.append("GenesisCfg")
