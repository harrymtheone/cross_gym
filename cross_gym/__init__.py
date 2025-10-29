"""Cross-Gym: A cross-platform robot reinforcement cross_rl framework.

Cross-Gym provides a unified interface for robot RL across multiple simulators
(IsaacGym, Genesis, IsaacSim) with an architecture inspired by IsaacLab.
"""

__version__ = "0.1.0"

# Core simulator imports
from .sim import *

# Actuators
from .actuators import *

# Asset imports
from .assets import *

# Scene imports
from .scene import *

# Environment imports
from .envs import *

# Manager imports
from .managers import *

# MDP terms
from .envs import mdp

# Terrain
from . import terrains
