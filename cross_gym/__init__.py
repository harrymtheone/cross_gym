"""Cross-Gym: A cross-platform robot reinforcement learning framework.

Cross-Gym provides a unified interface for robot RL across multiple simulators
(IsaacGym, Genesis, IsaacSim) with an architecture inspired by IsaacLab.
"""

__version__ = "0.1.0"

# Core simulator imports
from cross_gym.sim import (
    SimulationContext,
    SimulationCfg,
    PhysxCfg,
    RenderCfg,
    SimulatorType,
)

# Asset imports
from cross_gym.assets import (
    AssetBase,
    AssetBaseCfg,
    Articulation,
    ArticulationCfg,
    ArticulationData,
)

__all__ = [
    # Simulation
    "SimulationContext",
    "SimulationCfg",
    "PhysxCfg",
    "RenderCfg",
    "SimulatorType",
    # Assets
    "AssetBase",
    "AssetBaseCfg",
    "Articulation",
    "ArticulationCfg",
    "ArticulationData",
]
