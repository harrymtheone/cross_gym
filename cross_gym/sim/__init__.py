"""IsaacGym simulation context."""

from .isaacgym_context import IsaacGymContext
from .isaacgym_cfg import IsaacGymCfg, PhysXCfg

# Set class_type to enable cfg.class_type(cfg) pattern
IsaacGymCfg.class_type = IsaacGymContext

