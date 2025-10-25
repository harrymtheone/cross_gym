"""Task definitions for Cross-Gym.

This package contains task definitions organized by type:
- manager_based_rl: Tasks using the manager-based workflow
- direct_rl: Tasks using the direct RL workflow

It also provides the task registry for centralized task management.
"""

from .task_registry import TaskRegistry, task_registry

# Import task modules (users can add their tasks here)
from . import manager_based_rl
from . import direct_rl

__all__ = [
    "TaskRegistry",
    "task_registry",
    "manager_based_rl",
    "direct_rl",
]

__version__ = "0.1.0"

