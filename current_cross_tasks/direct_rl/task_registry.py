"""Task registry for managing and creating tasks."""

from __future__ import annotations

import argparse

from . import TaskCfg


class TaskRegistry:
    """Registry for creating RL training components.
    
    Usage:
        1. Create task config
        2. Pass config and args to TaskRegistry
        3. Registry applies args overrides
        4. Call make() to get runner
    
    Example:
        >>> task_cfg = T1DreamWaqCfg()
        >>> registry = TaskRegistry(task_cfg, args)
        >>> runner = registry.make()
        >>> runner.learn()
    """

    def __init__(self, task_cfg: TaskCfg, args: argparse.Namespace | None = None):
        """Initialize task registry with task configuration and args.
        
        Args:
            task_cfg: Complete task configuration
            args: Command-line arguments (optional, for overrides)
        """
        # Apply command-line overrides if args provided
        if args is not None:
            self._apply_args_overrides(task_cfg, args)

        task_cfg.validate()  # noqa
        self.cfg = task_cfg

    @staticmethod
    def _apply_args_overrides(task_cfg: TaskCfg, args: argparse.Namespace):
        """Apply command-line argument overrides to task config.
        
        Args:
            task_cfg: Task configuration to modify
            args: Parsed command-line arguments
        """
        # Device override
        if hasattr(args, 'device') and args.device is not None:
            task_cfg.env.sim.device = args.device

        # Headless mode
        if hasattr(args, 'headless') and args.headless:
            task_cfg.env.sim.headless = True

        # Debug mode: reduce complexity
        if hasattr(args, 'debug') and args.debug:
            print("[DEBUG MODE] Reducing environment complexity for faster iteration...")
            task_cfg.env.scene.num_envs = 16

            # Reduce terrain size if present
            if hasattr(task_cfg.env.scene, 'terrain') and task_cfg.env.scene.terrain is not None:
                task_cfg.env.scene.terrain.num_rows = 5
                task_cfg.env.scene.terrain.num_cols = 5

            # Disable logger in debug
            if hasattr(task_cfg, 'runner') and hasattr(task_cfg.runner, 'logger_backend'):
                task_cfg.runner.logger_backend = None

        # Experiment tracking
        if hasattr(args, 'proj_name'):
            task_cfg.runner.project_name = args.proj_name
        if hasattr(args, 'exptid'):
            task_cfg.runner.experiment_name = args.exptid
        if hasattr(args, 'log_dir'):
            task_cfg.runner.log_root_dir = args.log_dir

        # Resume from checkpoint
        if hasattr(args, 'resumeid') and args.resumeid is not None:
            task_cfg.resume_id = args.resumeid
            if hasattr(args, 'checkpoint') and args.checkpoint is not None:
                task_cfg.checkpoint = args.checkpoint

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
        return self.cfg.env.class_type(self.cfg.env)
