"""Cross-Gym: A cross-platform robot reinforcement learning framework.

Cross-Gym provides a unified interface for robot RL across multiple simulators
(IsaacGym, Genesis, IsaacSim) with an architecture inspired by IsaacLab.
"""

__version__ = "0.1.0"

# Core simulator imports
from cross_gym.sim import (
    SimulationContext,
    SimCfgBase,
)

# Simulator-specific imports (when available)
try:
    from cross_gym.sim import IsaacGymCfg, PhysxCfg, IsaacGymContext
except ImportError:
    IsaacGymCfg = None
    PhysxCfg = None

try:
    from cross_gym.sim import GenesisCfg
except ImportError:
    GenesisCfg = None

# Asset imports
from cross_gym.assets import (
    AssetBase,
    AssetBaseCfg,
    Articulation,
    ArticulationCfg,
    ArticulationData,
)

# Scene imports
from cross_gym.scene import (
    InteractiveScene,
    InteractiveSceneCfg,
)

# Environment imports
from cross_gym.envs import (
    ManagerBasedEnv,
    ManagerBasedEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
)

# Manager imports
from cross_gym.managers import (
    ActionManager,
    ActionManagerCfg,
    ObservationManager,
    ObservationManagerCfg,
    ObservationGroupCfg,
    RewardManager,
    RewardManagerCfg,
    TerminationManager,
    TerminationManagerCfg,
    CommandManager,
    CommandManagerCfg,
    EventManager,
    EventManagerCfg,
    ManagerTermCfg,
)

__all__ = [
    # Simulation
    "SimulationContext",
    "SimCfgBase",
    # Assets
    "AssetBase",
    "AssetBaseCfg",
    "Articulation",
    "ArticulationCfg",
    "ArticulationData",
    # Scene
    "InteractiveScene",
    "InteractiveSceneCfg",
    # Environments
    "ManagerBasedEnv",
    "ManagerBasedEnvCfg",
    "ManagerBasedRLEnv",
    "ManagerBasedRLEnvCfg",
    # Managers
    "ActionManager",
    "ActionManagerCfg",
    "ObservationManager",
    "ObservationManagerCfg",
    "ObservationGroupCfg",
    "RewardManager",
    "RewardManagerCfg",
    "TerminationManager",
    "TerminationManagerCfg",
    "CommandManager",
    "CommandManagerCfg",
    "EventManager",
    "EventManagerCfg",
    "ManagerTermCfg",
]

# Add simulator-specific exports if available
if IsaacGymCfg is not None:
    __all__.extend(["IsaacGymCfg", "PhysxCfg", "IsaacGymContext"])
if GenesisCfg is not None:
    __all__.append("GenesisCfg")
