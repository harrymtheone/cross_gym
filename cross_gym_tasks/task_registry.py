"""Task registry for managing and creating tasks."""

from __future__ import annotations

from . import TaskCfg


class TaskRegistry:
    """Registry for creating RL training components.
    
    Usage:
        1. Create task config (contains env + algorithm + runner settings)
        2. Pass to TaskRegistry(task_cfg)
        3. Call make() to get runner (env and algorithm are created and passed to runner)
        4. Call runner.learn()
    
    Example:
        >>> task_cfg = LocomotionTaskCfg()
        >>> task_registry = TaskRegistry(task_cfg)
        >>> runner = task_registry.make()
        >>> runner.learn()
    """

    def __init__(self, task_cfg: TaskCfg):
        """Initialize task registry with task configuration.
        
        Args:
            task_cfg: Complete task configuration
        """
        task_cfg.validate()  # noqa
        self.cfg = task_cfg

    def make(self):
        """Create runner with env and algorithm.
        
        This method:
        1. Creates environment from task_cfg.env
        2. Creates algorithm from task_cfg.algorithm, passing env
        3. Creates runner from task_cfg.runner, passing env and algorithm
        4. Returns runner ready for training
        
        Returns:
            Runner instance ready to call .learn()
        """
        # Step 1: Create environment
        env = self.cfg.env.class_type(self.cfg.env)

        print(f"[TaskRegistry] Environment created: {env.num_envs} parallel environments")

        # Step 2: Create algorithm, passing environment
        algorithm_cfg = self.cfg.algorithm

        # Set num_steps_per_update in algorithm config
        algorithm_cfg.num_steps_per_update = self.cfg.runner.num_steps_per_update

        algorithm = algorithm_cfg.class_type(algorithm_cfg, env)
        print(f"[TaskRegistry] Algorithm created: {algorithm_cfg.class_type.__name__}")

        # Step 3: Create runner using runner config from task_cfg
        runner_cfg = self.cfg.runner

        # Create runner, passing env and algorithm
        runner = runner_cfg.class_type(runner_cfg, env=env, algorithm=algorithm)
        print(f"[TaskRegistry] Runner created: {runner_cfg.class_type.__name__}")

        return runner

    def make_env(self):
        """Create environment only (for inference/evaluation).
        
        Returns:
            Environment instance
        """
        # validate() already checked that class_type exists
        env = self.cfg.env.class_type(self.cfg.env)

        return env


__all__ = ["TaskRegistry"]
