"""Complete training example for Cross-Gym.

This example demonstrates the clean TaskRegistry pattern:
1. TaskCfg contains env + algorithm + runner (everything in one config)
2. TaskRegistry(task_cfg) receives the complete configuration
3. task_registry.make() creates env → algorithm → runner
4. runner.learn() starts training
"""

try:
    import isaacgym, torch
except ImportError:
    import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cross_gym import *
from cross_gym.utils.configclass import configclass
from cross_gym_tasks import TaskRegistry, TaskCfg


# ============================================================================
# Scene Configuration
# ============================================================================

@configclass
class ExampleSceneCfg(InteractiveSceneCfg):
    """Scene with robot."""
    
    num_envs: int = 4096
    env_spacing: float = 4.0
    
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        file=None,  # TODO: Provide actual robot URDF path
        init_state=ArticulationCfg.InitStateCfg(
            pos=(0.0, 0.0, 0.6),
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
        ),
    )


# ============================================================================
# Environment Configuration
# ============================================================================

@configclass
class ExampleEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration."""
    
    # Simulation
    sim: IsaacGymCfg = IsaacGymCfg(
        dt=0.01,  # 100 Hz physics
        device="cuda:0",
        headless=True,
    )
    
    # Scene
    scene: ExampleSceneCfg = ExampleSceneCfg()
    
    # Episode settings
    decimation: int = 2  # 50 Hz control
    episode_length_s: float = 10.0
    
    # Actions (TODO: Add action terms when implemented)
    actions: ActionManagerCfg = ActionManagerCfg()
    # actions.joint_effort = ManagerTermCfg(func=mdp.actions.JointEffortAction, ...)
    
    # Observations - use MDP library
    observations: ObservationManagerCfg = ObservationManagerCfg()
    observations.policy = ObservationGroupCfg(concatenate=True)
    observations.policy.base_lin_vel = ManagerTermCfg(func=mdp.observations.base_lin_vel)
    observations.policy.base_ang_vel = ManagerTermCfg(func=mdp.observations.base_ang_vel)
    observations.policy.joint_pos = ManagerTermCfg(func=mdp.observations.joint_pos)
    observations.policy.joint_vel = ManagerTermCfg(func=mdp.observations.joint_vel)
    
    # Rewards - use MDP library
    rewards: RewardManagerCfg = RewardManagerCfg()
    rewards.alive = ManagerTermCfg(func=mdp.rewards.alive_reward, weight=1.0)
    rewards.forward = ManagerTermCfg(
        func=mdp.rewards.lin_vel_tracking_reward,
        weight=2.0,
        params={"target_x": 1.0, "target_y": 0.0}
    )
    rewards.energy = ManagerTermCfg(func=mdp.rewards.energy_penalty, weight=-0.01)
    rewards.upright = ManagerTermCfg(func=mdp.rewards.upright_reward, weight=0.5)
    
    # Terminations - use MDP library
    terminations: TerminationManagerCfg = TerminationManagerCfg()
    terminations.time_out = ManagerTermCfg(func=mdp.terminations.time_out)
    terminations.base_height = ManagerTermCfg(
        func=mdp.terminations.base_height_termination,
        params={"min_height": 0.3}
    )


# ============================================================================
# Complete Task Configuration (env + algorithm + runner)
# ============================================================================

@configclass
class ExampleTaskCfg(TaskCfg):
    """Complete task configuration.
    
    This single config contains everything needed for training:
    - Environment (simulation, scene, managers)
    - Algorithm (PPO with hyperparameters)
    - Runner (training iterations, logging)
    """
    
    # ========== Environment ==========
    env: ExampleEnvCfg = ExampleEnvCfg()
    
    # ========== Algorithm ==========
    from cross_gym_rl.algorithms.ppo import PPOCfg
    
    algorithm: PPOCfg = PPOCfg(
        # RL parameters
        gamma=0.99,
        lam=0.95,
        # PPO parameters
        clip_param=0.2,
        num_mini_batches=4,
        num_learning_epochs=5,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        # Network architecture
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation='elu',
        # Learning
        learning_rate=1e-3,
        learning_rate_schedule='adaptive',
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
    # ========== Runner ==========
    from cross_gym_rl.runners import OnPolicyRunnerCfg
    
    runner: OnPolicyRunnerCfg = OnPolicyRunnerCfg(
        max_iterations=1000,
        num_steps_per_update=24,
        log_interval=1,
        save_interval=100,
        logger_backend="tensorboard",
        log_dir="logs",
        project_name="cross_gym_example",
        experiment_name="test_run",
    )


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function."""
    
    print("=" * 80)
    print("Cross-Gym Training Example")
    print("=" * 80)
    
    # Step 1: Create task configuration
    print("\n[Step 1] Creating task configuration...")
    task_cfg = ExampleTaskCfg()
    print(f"  ✓ Task configured")
    print(f"    - Environments: {task_cfg.env.scene.num_envs}")
    print(f"    - Algorithm: {task_cfg.algorithm.class_type.__name__}")
    print(f"    - Max iterations: {task_cfg.runner.max_iterations}")
    
    # Step 2: Create TaskRegistry with config
    print("\n[Step 2] Creating TaskRegistry...")
    task_registry = TaskRegistry(task_cfg)
    print(f"  ✓ TaskRegistry created")
    
    # Step 3: Make runner (creates env → algorithm → runner)
    print("\n[Step 3] Making runner...")
    print("  This will create:")
    print("    1. Environment (simulation + scene + managers)")
    print("    2. Algorithm (PPO with actor-critic)")
    print("    3. Runner (training loop + logging)")
    
    # Uncomment to actually run (requires IsaacGym + robot URDF)
    # runner = task_registry.make()
    
    # Step 4: Train!
    # print("\n[Step 4] Starting training...")
    # runner.learn()
    
    print("\n" + "=" * 80)
    print("NOTE: This is a configuration example.")
    print("\nTo actually train, you need:")
    print("  1. IsaacGym installed")
    print("  2. Robot URDF file provided in ExampleSceneCfg")
    print("  3. Action terms implemented")
    print("\nThen uncomment the runner.make() and runner.learn() lines above.")
    print("=" * 80)
    print("\nClean pattern demonstrated:")
    print("  task_cfg = MyTaskCfg()           # Everything in one config")
    print("  registry = TaskRegistry(task_cfg)  # Pass config")
    print("  runner = registry.make()          # Creates env→alg→runner")
    print("  runner.learn()                    # Train!")
    print("=" * 80)


if __name__ == "__main__":
    main()

