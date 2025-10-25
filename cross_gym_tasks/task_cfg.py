"""Base configuration for tasks."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from cross_gym.envs import ManagerBasedRLEnvCfg
from cross_gym.utils.configclass import configclass

if TYPE_CHECKING:
    from cross_gym_rl.algorithms.ppo import PPOCfg
    from cross_gym_rl.runners import OnPolicyRunnerCfg


@configclass
class TaskCfg:
    """Base configuration for a complete task.
    
    A task config contains:
    1. env - Environment configuration
    2. algorithm - Algorithm configuration (from cross_gym_rl)
    3. runner - Runner configuration (from cross_gym_rl)
    
    Everything needed for training in one unified config!
    
    Example:
        >>> from cross_gym_rl.algorithms.ppo import PPOCfg
        >>> from cross_gym_rl.runners import OnPolicyRunnerCfg
        >>> 
        >>> @configclass
        >>> class MyTaskCfg(TaskCfg):
        >>>     # Environment
        >>>     env: ManagerBasedRLEnvCfg = MyEnvCfg()
        >>>     
        >>>     # Algorithm
        >>>     algorithm: PPOCfg = PPOCfg(gamma=0.99, ...)
        >>>     
        >>>     # Runner
        >>>     runner: OnPolicyRunnerCfg = OnPolicyRunnerCfg(
        >>>         max_iterations=1000,
        >>>         project_name="my_project",
        >>>         experiment_name="exp001",
        >>>     )
        >>> 
        >>> # Use with TaskRegistry
        >>> from cross_gym_tasks import TaskRegistry
        >>> task_registry = TaskRegistry(MyTaskCfg())
        >>> runner = task_registry.make()
        >>> runner.learn()
    """

    # ========== Environment ==========
    env: ManagerBasedRLEnvCfg = MISSING
    """Environment configuration."""
    
    # ========== Algorithm ==========
    algorithm: PPOCfg = MISSING
    """Algorithm configuration (PPOCfg from cross_gym_rl)."""
    
    # ========== Runner ==========
    runner: OnPolicyRunnerCfg = MISSING
    """Runner configuration (OnPolicyRunnerCfg from cross_gym_rl)."""
    
    def __post_init__(self):
        """Post-initialization to link configs.
        
        Sets num_steps_per_update in algorithm config from runner config.
        """
        # Link num_steps_per_update from runner to algorithm
        if hasattr(self.algorithm, 'num_steps_per_update'):
            self.algorithm.num_steps_per_update = self.runner.num_steps_per_update


__all__ = ["TaskCfg"]
