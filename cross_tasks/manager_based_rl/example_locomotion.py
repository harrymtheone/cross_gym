"""Example locomotion task using manager-based workflow.

This demonstrates how to define a complete task with both
environment settings and training settings in one config.
"""

from dataclasses import MISSING

from cross_gym import *
from cross_gym.utils.configclass import configclass
from cross_gym_tasks import TaskCfg


# ============================================================================
# Scene Configuration
# ============================================================================

@configclass
class LocomotionSceneCfg(InteractiveSceneCfg):
    """Scene configuration for locomotion task."""

    num_envs: int = 4096
    env_spacing: float = 4.0

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        file=MISSING,  # User must provide URDF path
        init_state=ArticulationCfg.InitStateCfg(
            pos=(0.0, 0.0, 0.6),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# ============================================================================
# Environment Configuration
# ============================================================================

@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for locomotion."""

    # Simulation
    sim: IsaacGymCfg = IsaacGymCfg(
        dt=0.01,
        device="cuda:0",
        headless=True,
    )

    # Scene
    scene: LocomotionSceneCfg = LocomotionSceneCfg()

    # Episode
    decimation: int = 2
    episode_length_s: float = 10.0

    # Actions (TODO: Implement action terms)
    actions: ActionManagerCfg = ActionManagerCfg()

    # Observations
    observations: ObservationManagerCfg = ObservationManagerCfg()
    observations.policy = ObservationGroupCfg(concatenate=True)
    observations.policy.base_lin_vel = ManagerTermCfg(func=mdp.observations.base_lin_vel)
    observations.policy.base_ang_vel = ManagerTermCfg(func=mdp.observations.base_ang_vel)
    observations.policy.joint_pos = ManagerTermCfg(func=mdp.observations.joint_pos)
    observations.policy.joint_vel = ManagerTermCfg(func=mdp.observations.joint_vel)

    # Rewards
    rewards: RewardManagerCfg = RewardManagerCfg()
    rewards.alive = ManagerTermCfg(func=mdp.rewards.alive_reward, weight=1.0)
    rewards.forward = ManagerTermCfg(
        func=mdp.rewards.lin_vel_tracking_reward,
        weight=2.0,
        params={"target_x": 1.0, "target_y": 0.0}
    )
    rewards.energy = ManagerTermCfg(func=mdp.rewards.energy_penalty, weight=-0.01)
    rewards.upright = ManagerTermCfg(func=mdp.rewards.upright_reward, weight=0.5)

    # Terminations
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
class LocomotionTaskCfg(TaskCfg):
    """Complete locomotion task configuration.
    
    Contains environment, algorithm, and runner configurations.
    """

    # ========== Environment ==========
    env: LocomotionEnvCfg = LocomotionEnvCfg()

    # ========== Algorithm ========== 
    from cross_gym_rl.algorithms.ppo import PPOCfg

    algorithm: PPOCfg = PPOCfg(
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        num_mini_batches=4,
        num_learning_epochs=5,
        learning_rate=1e-3,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
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
        project_name="cross_gym",
        experiment_name="locomotion_v1",
    )


__all__ = ["LocomotionTaskCfg"]
