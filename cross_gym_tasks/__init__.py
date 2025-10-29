"""Task definitions for Cross-Gym.

This package contains:
1. TaskCfg - Base configuration for tasks (env + algorithm + runner)
2. TaskRegistry - Creates env, algorithm, runner from TaskCfg
3. Example tasks in manager_based_rl/ and direct_rl/

Usage:
    >>> task_cfg = MyTaskCfg()  # Contains env + algorithm + runner settings
    >>> task_registry = TaskRegistry(task_cfg)
    >>> runner = task_registry.make()  # Creates and wires everything
    >>> runner.learn()
"""

from .task_cfg import TaskCfg
from .task_registry import TaskRegistry

from .direct_rl import direct_tasks
