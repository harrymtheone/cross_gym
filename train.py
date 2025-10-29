"""Training script for Cross-Gym RL tasks."""
try:
    import isaacgym, torch
except ImportError:
    import torch

import argparse

from cross_gym_tasks import TaskRegistry, TaskCfg, direct_tasks


def get_arg_parser():
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(description="Cross-Gym RL Training Framework")

    # Environment settings
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., t1_dreamwaq)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on (cuda:0, cpu, etc.)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no rendering)')

    # Experiment tracking
    parser.add_argument('--proj_name', type=str, required=True, help='Project name for logging')
    parser.add_argument('--exptid', type=str, required=True, help='Experiment ID')
    parser.add_argument('--resumeid', type=str, default=None, help='Resume from experiment ID')
    parser.add_argument('--checkpoint', type=int, default=None, help='Checkpoint iteration to resume from')

    # Logging
    parser.add_argument('--log_dir', type=str, default='logs', help='Root directory for logs')
    parser.add_argument('--debug', action='store_true', help='Debug mode (fewer envs, smaller terrain)')

    return parser


def main():
    """Main training function."""
    # Parse arguments
    args = get_arg_parser().parse_args()

    # Get task configuration
    if args.task not in direct_tasks:
        available_tasks = ', '.join(direct_tasks.keys())
        raise ValueError(f"Task '{args.task}' not found. Available tasks: {available_tasks}")

    task_cfg: TaskCfg = direct_tasks[args.task]()

    # Create registry (applies args overrides automatically)
    registry = TaskRegistry(task_cfg, args)

    # Print configuration summary
    print("=" * 80)
    print(f"Cross-Gym RL Training")
    print("=" * 80)
    print(f"Task:             {args.task}")
    print(f"Device:           {registry.cfg.env.sim.device}")
    print(f"Num Environments: {registry.cfg.env.scene.num_envs}")
    print(f"Headless:         {registry.cfg.env.sim.headless}")
    print(f"Project:          {args.proj_name}/{args.exptid}")
    print(f"Log Directory:    {args.log_root}")
    print("=" * 80)

    # Create runner (TaskRegistry handles env + algorithm creation)
    runner = registry.make()
    
    # Start training
    runner.learn()


if __name__ == "__main__":
    main()
