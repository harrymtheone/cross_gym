"""Direct RL tasks.

Tasks defined using the direct RL workflow (without managers).
For simpler tasks or when you need more direct control.

Example tasks:
- Locomotion (simple forward walking)
- Parkour (terrain navigation with curriculum)
- Humanoid (bipedal locomotion with gait control)
"""

from cross_gym_tasks.direct_rl.base.locomotion_env import LocomotionEnv
from cross_gym_tasks.direct_rl.base.env_cfgs import LocomotionEnvCfg

