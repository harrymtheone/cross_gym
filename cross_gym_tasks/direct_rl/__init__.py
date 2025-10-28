"""Direct RL tasks.

Tasks defined using the direct RL workflow (without managers).
For simpler tasks or when you need more direct control.

Example tasks:
- Locomotion (simple forward walking)
"""

from .locomotion_env import LocomotionEnv, LocomotionEnvCfg
from .parkour_env import ParkourEnv, ParkourEnvCfg

from . import rewards
