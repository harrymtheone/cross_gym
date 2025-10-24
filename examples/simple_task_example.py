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

import torch

from cross_gym import (
    # Simulation
    SimulationCfg,
    SimulatorType,
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
)
from cross_gym.utils.configclass import configclass


# ============================================================================
# MDP Terms (Observations, Actions, Rewards, Terminations)
# ============================================================================

def base_lin_vel(env) -> torch.Tensor:
    """Observation: Base linear velocity in world frame.
    
    Returns:
        Linear velocity (num_envs, 3)
    """
    robot = env.scene["robot"]
    return robot.data.root_vel_w


def base_ang_vel(env) -> torch.Tensor:
    """Observation: Base angular velocity in world frame.
    
    Returns:
        Angular velocity (num_envs, 3)
    """
    robot = env.scene["robot"]
    return robot.data.root_ang_vel_w


def joint_pos(env) -> torch.Tensor:
    """Observation: Joint positions.
    
    Returns:
        Joint positions (num_envs, num_dof)
    """
    robot = env.scene["robot"]
    return robot.data.joint_pos


def joint_vel(env) -> torch.Tensor:
    """Observation: Joint velocities.
    
    Returns:
        Joint velocities (num_envs, num_dof)
    """
    robot = env.scene["robot"]
    return robot.data.joint_vel


def alive_reward(env) -> torch.Tensor:
    """Reward: Constant reward for staying alive.
    
    Returns:
        Reward (num_envs,)
    """
    return torch.ones(env.num_envs, device=env.device)


def energy_penalty(env) -> torch.Tensor:
    """Reward: Penalty for using energy (torque squared).
    
    Returns:
        Penalty (num_envs,)
    """
    robot = env.scene["robot"]
    torques = robot.data.applied_torques
    return -torch.sum(torques ** 2, dim=-1)


def time_out(env) -> torch.Tensor:
    """Termination: Episode timeout.
    
    Returns:
        Boolean tensor (num_envs,)
    """
    return env.episode_length_buf >= env.max_episode_length


def base_height_termination(env, min_height: float = 0.3) -> torch.Tensor:
    """Termination: Robot base fell below minimum height.
    
    Args:
        min_height: Minimum allowed base height
    
    Returns:
        Boolean tensor (num_envs,)
    """
    robot = env.scene["robot"]
    base_height = robot.data.root_pos_w[:, 2]
    return base_height < min_height


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
            rot=(0.0, 0.0, 0.0, 1.0),  # Identity quaternion
        ),
    )


@configclass
class SimpleTaskCfg(ManagerBasedRLEnvCfg):
    """Configuration for the simple task."""

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        simulator=SimulatorType.ISAACGYM,
        dt=0.01,  # 100 Hz physics
        device="cuda:0",
        headless=True,
    )

    # Scene
    scene: SimpleSceneCfg = SimpleSceneCfg()

    # Environment
    decimation: int = 2  # 50 Hz control
    episode_length_s: float = 10.0  # 10 second episodes

    # Actions
    actions: ActionManagerCfg = ActionManagerCfg()
    # actions.joint_effort = ... # TODO: Define action term

    # Observations
    observations: ObservationManagerCfg = ObservationManagerCfg()
    observations.policy = ObservationGroupCfg(concatenate=True)
    observations.policy.base_lin_vel = ManagerTermCfg(func=base_lin_vel)
    observations.policy.base_ang_vel = ManagerTermCfg(func=base_ang_vel)
    observations.policy.joint_pos = ManagerTermCfg(func=joint_pos)
    observations.policy.joint_vel = ManagerTermCfg(func=joint_vel)

    # Rewards
    rewards: RewardManagerCfg = RewardManagerCfg()
    rewards.alive = ManagerTermCfg(func=alive_reward, weight=1.0)
    rewards.energy = ManagerTermCfg(func=energy_penalty, weight=-0.001)

    # Terminations
    terminations: TerminationManagerCfg = TerminationManagerCfg()
    terminations.time_out = ManagerTermCfg(func=time_out)
    terminations.base_height = ManagerTermCfg(
        func=base_height_termination,
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
    print(f"  Simulator: {cfg.sim.simulator.name}")
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
    print("NOTE: This is a configuration example only.")
    print("To actually run, you need:")
    print("  1. Robot URDF file")
    print("  2. Action term implementation")
    print("  3. Proper simulator setup")
    print("=" * 80)


if __name__ == "__main__":
    main()
