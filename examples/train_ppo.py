"""Example training script for PPO with Cross-Gym.

This demonstrates how to train a policy using PPO on a Cross-Gym task.
"""
try:
    import isaacgym, torch  # noqa
except ImportError:
    import torch

from cross_gym import *
from cross_gym.utils.configclass import configclass
from cross_gym_rl.algorithms.ppo import PPOCfg
from cross_gym_rl.runners import OnPolicyRunnerCfg


# Import the task (in real use, this would be your custom task)
# from my_task import MyTaskCfg

# For this example, we'll create a minimal task config
@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    """Simple scene configuration."""
    num_envs: int = 16  # Small for testing
    env_spacing: float = 4.0

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        file=None,  # Would need actual URDF
        init_state=ArticulationCfg.InitStateCfg(
            pos=(0.0, 0.0, 0.6),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class SimpleTaskCfg(ManagerBasedRLEnvCfg):
    """Simple task configuration."""

    # Simulation
    sim: IsaacGymCfg = IsaacGymCfg(
        dt=0.01,
        device="cuda:0",
        headless=True,
    )

    # Scene
    scene: SimpleSceneCfg = SimpleSceneCfg()

    # Episode
    decimation: int = 2
    episode_length_s: float = 10.0

    # Actions
    actions: ActionManagerCfg = ActionManagerCfg()
    # Would need to add action terms

    # Observations
    observations: ObservationManagerCfg = ObservationManagerCfg()
    observations.policy = ObservationGroupCfg()
    observations.policy.base_vel = ManagerTermCfg(func=mdp.observations.base_lin_vel)
    observations.policy.joint_pos = ManagerTermCfg(func=mdp.observations.joint_pos)

    # Rewards
    rewards: RewardManagerCfg = RewardManagerCfg()
    rewards.alive = ManagerTermCfg(func=mdp.rewards.alive_reward, weight=1.0)

    # Terminations
    terminations: TerminationManagerCfg = TerminationManagerCfg()
    terminations.time_out = ManagerTermCfg(func=mdp.terminations.time_out)


@configclass
class TrainCfg(OnPolicyRunnerCfg):
    """Training configuration."""

    # Environment
    env_cfg: SimpleTaskCfg = SimpleTaskCfg()

    # Algorithm
    algorithm_cfg: PPOCfg = PPOCfg(
        # RL parameters
        gamma=0.99,
        lam=0.95,
        # PPO parameters
        clip_param=0.2,
        num_mini_batches=4,
        num_learning_epochs=5,
        # Network
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation='elu',
        # Learning
        learning_rate=1e-3,
        learning_rate_schedule='adaptive',
        desired_kl=0.01,
        # Training settings
        max_grad_norm=1.0,
        use_amp=False,
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
    experiment_name: str = "test_ppo"


def main():
    """Main training function."""

    print("=" * 80)
    print("Cross-Gym PPO Training Example")
    print("=" * 80)

    # NOTE: This is a configuration example
    # To actually train, you need:
    # 1. A real robot URDF
    # 2. Action term implementation
    # 3. Proper task setup

    print("\nTraining Configuration:")
    cfg = TrainCfg()
    print(f"  Algorithm: {cfg.algorithm_cfg.class_type.__name__}")
    print(f"  Environments: {cfg.env_cfg.scene.num_envs}")
    print(f"  Max iterations: {cfg.max_iterations}")
    print(f"  Steps per update: {cfg.num_steps_per_update}")

    # Uncomment to actually run (requires proper setup)
    # runner = OnPolicyRunner(cfg)
    # runner.learn()

    print("\n" + "=" * 80)
    print("NOTE: This is a configuration example.")
    print("To actually train, implement action terms and provide robot URDF.")
    print("=" * 80)


if __name__ == "__main__":
    main()
