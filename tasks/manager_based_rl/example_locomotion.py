"""Example locomotion task using manager-based workflow.

This is an example of how to define a locomotion task.
Users can copy and modify this for their own tasks.
"""

from dataclasses import MISSING

from cross_gym import *
from cross_gym.utils.configclass import configclass
from tasks import task_registry


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
# Task Configuration
# ============================================================================

@configclass
class LocomotionTaskCfg(ManagerBasedRLEnvCfg):
    """Locomotion task configuration.
    
    This task trains a quadruped/biped robot to walk forward.
    """
    
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
    # actions.joint_effort = ManagerTermCfg(func=mdp.actions.JointEffortAction, ...)
    
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
# Register Task
# ============================================================================

# Register this task so it can be used with task_registry
task_registry.register("locomotion", LocomotionTaskCfg, task_type="manager_based")


__all__ = ["LocomotionTaskCfg"]

