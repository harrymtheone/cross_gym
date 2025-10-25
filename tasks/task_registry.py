"""Task registry for managing and creating tasks.

The task registry provides a clean interface for:
1. Registering tasks
2. Creating environments
3. Creating runners/algorithms
4. Managing both manager-based and direct RL workflows
"""

from __future__ import annotations

from typing import Dict, Optional, Callable
import argparse

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv


class TaskRegistry:
    """Registry for RL tasks.
    
    Provides a centralized way to:
    - Register task configurations
    - Create environments from task names
    - Create runners/algorithms
    - Parse common arguments
    
    Example:
        >>> # Register a task
        >>> task_registry = TaskRegistry()
        >>> task_registry.register("locomotion", LocomotionTaskCfg)
        >>> 
        >>> # Create environment
        >>> env = task_registry.make_env("locomotion", args)
        >>> 
        >>> # Create runner
        >>> runner = task_registry.make_runner("locomotion", args)
        >>> runner.learn()
    """
    
    def __init__(self):
        """Initialize task registry."""
        self._task_cfgs: Dict[str, Callable] = {}
        self._task_types: Dict[str, str] = {}  # "manager_based" or "direct"
    
    def register(
        self,
        name: str,
        task_cfg_class: Callable,
        task_type: str = "manager_based",
    ):
        """Register a task.
        
        Args:
            name: Task name
            task_cfg_class: Task configuration class (not instance!)
            task_type: "manager_based" or "direct"
        """
        if name in self._task_cfgs:
            print(f"[WARNING] Task '{name}' already registered, overwriting.")
        
        self._task_cfgs[name] = task_cfg_class
        self._task_types[name] = task_type
        print(f"[INFO] Registered task: {name} ({task_type})")
    
    def get_task_cfg_class(self, name: str) -> Callable:
        """Get task configuration class by name.
        
        Args:
            name: Task name
            
        Returns:
            Task configuration class
        """
        if name not in self._task_cfgs:
            raise ValueError(
                f"Task '{name}' not registered. "
                f"Available tasks: {list(self._task_cfgs.keys())}"
            )
        return self._task_cfgs[name]
    
    def make_env(
        self,
        name: str,
        args: Optional[argparse.Namespace] = None,
        num_envs: Optional[int] = None,
        **kwargs
    ):
        """Create environment from task name.
        
        Args:
            name: Task name
            args: Command line arguments (optional)
            num_envs: Override number of environments (optional)
            **kwargs: Additional arguments to pass to env config
            
        Returns:
            Environment instance
        """
        # Get task config class
        task_cfg_class = self.get_task_cfg_class(name)
        
        # Create config instance
        task_cfg = task_cfg_class()
        
        # Apply overrides from args
        if args is not None:
            if hasattr(args, 'num_envs') and args.num_envs is not None:
                task_cfg.scene.num_envs = args.num_envs
            if hasattr(args, 'device') and args.device is not None:
                task_cfg.sim.device = args.device
            if hasattr(args, 'headless') and args.headless is not None:
                task_cfg.sim.headless = args.headless
        
        # Apply kwargs overrides
        if num_envs is not None:
            task_cfg.scene.num_envs = num_envs
        
        for key, value in kwargs.items():
            if hasattr(task_cfg, key):
                setattr(task_cfg, key, value)
        
        # Create environment (import here to avoid circular dependency)
        if hasattr(task_cfg, 'class_type'):
            env = task_cfg.class_type(task_cfg)
        else:
            from cross_gym.envs import ManagerBasedRLEnv
            env = ManagerBasedRLEnv(task_cfg)
        
        return env
    
    def make_runner(
        self,
        name: str,
        runner_cfg_class: Callable,
        args: Optional[argparse.Namespace] = None,
        **kwargs
    ):
        """Create runner for training.
        
        Args:
            name: Task name
            runner_cfg_class: Runner configuration class
            args: Command line arguments (optional)
            **kwargs: Additional arguments
            
        Returns:
            Runner instance
        """
        # Get task config
        task_cfg_class = self.get_task_cfg_class(name)
        task_cfg = task_cfg_class()
        
        # Create runner config
        runner_cfg = runner_cfg_class()
        runner_cfg.env_cfg = task_cfg
        
        # Apply overrides from args
        if args is not None:
            if hasattr(args, 'max_iterations'):
                runner_cfg.max_iterations = args.max_iterations
            if hasattr(args, 'experiment_name'):
                runner_cfg.experiment_name = args.experiment_name
            if hasattr(args, 'resume_path'):
                runner_cfg.resume_path = args.resume_path
        
        # Apply kwargs
        for key, value in kwargs.items():
            if hasattr(runner_cfg, key):
                setattr(runner_cfg, key, value)
        
        # Create runner
        runner = runner_cfg.class_type(runner_cfg)
        
        return runner
    
    def list_tasks(self) -> list[str]:
        """List all registered tasks.
        
        Returns:
            List of task names
        """
        return list(self._task_cfgs.keys())
    
    @staticmethod
    def get_arg_parser() -> argparse.ArgumentParser:
        """Get argument parser with common RL arguments.
        
        Returns:
            ArgumentParser instance
        """
        parser = argparse.ArgumentParser(description="Cross-Gym RL Training")
        
        # Task selection
        parser.add_argument('--task', type=str, required=True, help='Task name')
        
        # Environment
        parser.add_argument('--num_envs', type=int, help='Number of parallel environments')
        parser.add_argument('--device', type=str, help='Device (cuda:0, cpu, etc.)')
        parser.add_argument('--headless', action='store_true', help='Run without GUI')
        
        # Training
        parser.add_argument('--max_iterations', type=int, help='Maximum training iterations')
        parser.add_argument('--experiment_name', type=str, help='Experiment name/ID')
        
        # Resume
        parser.add_argument('--resume_path', type=str, help='Path to checkpoint to resume from')
        
        return parser


# Global task registry instance
task_registry = TaskRegistry()


__all__ = ["TaskRegistry", "task_registry"]

