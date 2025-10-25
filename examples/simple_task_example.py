"""
Simple Example Task for Cross-Gym

This example demonstrates how to create a basic RL task using Cross-Gym's
manager-based workflow. It shows:
1. Defining a scene with a robot
2. Setting up observations, actions, rewards, and terminations
3. Creating and running the environment

Note: This is a minimal example for demonstration. A real task would need:
- Actual robot URDF file
- Proper observation/reward/action implementations
- More sophisticated termination conditions
"""

try:
    import isaacgym, torch
except ImportError:
    import torch

from cross_gym import (
    # Scene
    InteractiveSceneCfg,
    ArticulationCfg,
    # Environment
    ManagerBasedRLEnvCfg,
    # Managers
    ActionManagerCfg,
    ObservationManagerCfg,
    ObservationGroupCfg,
    RewardManagerCfg,
    TerminationManagerCfg,
    ManagerTermCfg,
    # MDP terms - use from library!
    mdp,
)
# Check if IsaacGym is available
from cross_gym import IsaacGymCfg
from cross_gym.utils.configclass import configclass


# ============================================================================
# Task Configuration
# ============================================================================

@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""

    # Number of environments
    num_envs: int = 4096
    env_spacing: float = 4.0

    # Robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        file=None,  # TODO: Provide actual URDF path
        # Initial state
        init_state=ArticulationCfg.InitStateCfg(
            pos=(0.0, 0.0, 0.6),  # Start 0.6m above ground
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
        ),
    )


@configclass
class SimpleTaskCfg(ManagerBasedRLEnvCfg):
    """Configuration for the simple task."""

    # Simulation - Use simulator-specific config!
    sim: IsaacGymCfg = IsaacGymCfg(
        dt=0.01,  # 100 Hz physics
        device="cuda:0",
        headless=True,
        physx=IsaacGymCfg.PhysxCfg(
            solver_type=1,
            num_position_iterations=4,
            num_velocity_iterations=1,
        ),
    )

    # Scene
    scene: SimpleSceneCfg = SimpleSceneCfg()

    # Environment
    decimation: int = 2  # 50 Hz control
    episode_length_s: float = 10.0  # 10 second episodes

    # Actions
    actions: ActionManagerCfg = ActionManagerCfg()
    # actions.joint_effort = ... # TODO: Define action term

    # Observations - use library functions!
    observations: ObservationManagerCfg = ObservationManagerCfg()
    observations.policy = ObservationGroupCfg(concatenate=True)
    observations.policy.base_lin_vel = ManagerTermCfg(func=mdp.observations.base_lin_vel)
    observations.policy.base_ang_vel = ManagerTermCfg(func=mdp.observations.base_ang_vel)
    observations.policy.joint_pos = ManagerTermCfg(func=mdp.observations.joint_pos)
    observations.policy.joint_vel = ManagerTermCfg(func=mdp.observations.joint_vel)

    # Rewards - use library functions!
    rewards: RewardManagerCfg = RewardManagerCfg()
    rewards.alive = ManagerTermCfg(func=mdp.rewards.alive_reward, weight=1.0)
    rewards.energy = ManagerTermCfg(func=mdp.rewards.energy_penalty, weight=-0.001)

    # Terminations - use library functions!
    terminations: TerminationManagerCfg = TerminationManagerCfg()
    terminations.time_out = ManagerTermCfg(func=mdp.terminations.time_out)
    terminations.base_height = ManagerTermCfg(
        func=mdp.terminations.base_height_termination,
        params={"min_height": 0.3}
    )


# ============================================================================
# Main Script
# ============================================================================

def main():
    """Main function to run the example."""

    print("=" * 80)
    print("Cross-Gym Simple Task Example")
    print("=" * 80)

    # Create task configuration
    cfg = SimpleTaskCfg()

    # Note: This example won't actually run without:
    # 1. A real robot URDF file
    # 2. Action term implementation
    # 3. Proper simulator setup

    print("\nTask Configuration:")
    print(f"  Simulator: {cfg.sim.class_type.__name__}")
    print(f"  Device: {cfg.sim.device}")
    print(f"  Num Envs: {cfg.scene.num_envs}")
    print(f"  Physics dt: {cfg.sim.dt}s")
    print(f"  Control dt: {cfg.decimation * cfg.sim.dt}s")
    print(f"  Episode length: {cfg.episode_length_s}s")

    print("\nManagers:")
    print(f"  Observations: {list(cfg.observations.policy.__dict__.keys())}")
    print(f"  Rewards: {list(cfg.rewards.__dict__.keys())}")
    print(f"  Terminations: {list(cfg.terminations.__dict__.keys())}")

    # Uncomment to actually create environment (requires proper setup)
    # print("\nCreating environment...")
    # env = ManagerBasedRLEnv(cfg)
    # 
    # print("\nRunning environment...")
    # obs, info = env.reset()
    # for step in range(100):
    #     # Random actions
    #     actions = torch.randn(env.num_envs, env.single_action_space.shape[0], device=env.device)
    #     obs, reward, terminated, truncated, info = env.step(actions)
    #     
    #     if step % 10 == 0:
    #         print(f"  Step {step}: reward = {reward.mean().item():.3f}")
    # 
    # env.close()
    # print("\nDone!")

    print("\n" + "=" * 80)
    print("NOTE: This is a configuration example.")
    print("\nTo actually run, you need:")
    print("  1. Robot URDF file")
    print("  2. Action term implementation")
    print("=" * 80)


if __name__ == "__main__":
    main()
