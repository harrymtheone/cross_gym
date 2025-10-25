"""Example training script using task registry.

This demonstrates the clean task registry pattern for training.
"""

try:
    import isaacgym, torch
except ImportError:
    import torch

from cross_gym.utils.configclass import configclass
from learning import OnPolicyRunnerCfg, PPOCfg
from tasks import task_registry


# ============================================================================
# Training Configuration (Generic for all tasks)
# ============================================================================

@configclass
class DefaultRunnerCfg(OnPolicyRunnerCfg):
    """Default training configuration."""

    # Algorithm
    algorithm_cfg: PPOCfg = PPOCfg(
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        num_mini_batches=4,
        num_learning_epochs=5,
        learning_rate=1e-3,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
    )

    # Training
    max_iterations: int = 1000
    num_steps_per_update: int = 24

    # Logging
    log_interval: int = 1
    save_interval: int = 100
    logger_backend: str = "tensorboard"
    log_dir: str = "logs"
    project_name: str = "cross_gym"
    experiment_name: str = "default_exp"


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main training function using task registry."""

    # Parse arguments
    parser = task_registry.get_arg_parser()
    args = parser.parse_args()

    print("=" * 80)
    print("Cross-Gym Training with Task Registry")
    print("=" * 80)
    print(f"  Task: {args.task}")
    print(f"  Available tasks: {task_registry.list_tasks()}")
    print("=" * 80)

    # Option 1: Create environment only
    # env = task_registry.make_env(args.task, args)
    # # Use env for inference, data collection, etc.

    # Option 2: Create runner for training
    runner = task_registry.make_runner(
        name=args.task,
        runner_cfg_class=DefaultRunnerCfg,
        args=args,
    )

    # Start training
    print("\n[INFO] Starting training...")
    runner.learn()


if __name__ == "__main__":
    # For testing without args
    print("=" * 80)
    print("Cross-Gym Task Registry Example")
    print("=" * 80)
    print(f"\nRegistered tasks: {task_registry.list_tasks()}")
    print("\nTo train, run:")
    print("  python examples/train_with_registry.py --task locomotion --headless")
    print("=" * 80)
