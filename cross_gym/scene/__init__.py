"""IsaacGym scene management."""

from .interactive_scene import IsaacGymInteractiveScene
from .interactive_scene_cfg import IsaacGymSceneCfg

# Set class_type to enable cfg.class_type(cfg, sim) pattern
IsaacGymSceneCfg.class_type = IsaacGymInteractiveScene

