"""Configuration for manager-based RL environment."""

from __future__ import annotations

from dataclasses import MISSING

from cross_gym.managers import RewardManagerCfg, TerminationManagerCfg, CommandManagerCfg
from cross_gym.utils import configclass
from .manager_based_env_cfg import ManagerBasedEnvCfg


@configclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    """Configuration for manager-based RL environment.
    
    This extends ManagerBasedEnvCfg with RL-specific managers:
    - RewardManager
    - TerminationManager
    - CommandManager (optional)
    
    Example:
        >>> from cross_gym import IsaacGymCfg, PhysxCfg
        >>> 
        >>> @configclass
        >>> class MyTaskCfg(ManagerBasedRLEnvCfg):
        >>>     # Simulation - use simulator-specific config
        >>>     sim = IsaacGymCfg(
        >>>         dt=0.01,
        >>>         device="cuda:0",
        >>>         physx=PhysxCfg(...),
        >>>     )
        >>>     
        >>>     # Scene
        >>>     scene = MySceneCfg(num_envs=4096, env_spacing=4.0)
        >>>     
        >>>     # Actions
        >>>     actions = ActionManagerCfg()
        >>>     actions.joint_pos = JointPositionActionCfg(...)
        >>>     
        >>>     # Observations
        >>>     observations = ObservationManagerCfg()
        >>>     observations.policy = ObservationGroupCfg()
        >>>     observations.policy.base_lin_vel = ObservationTermCfg(...)
        >>>     
        >>>     # Rewards
        >>>     rewards = RewardManagerCfg()
        >>>     rewards.tracking = ManagerTermCfg(func=tracking_reward, weight=1.0)
        >>>     
        >>>     # Terminations
        >>>     terminations = TerminationManagerCfg()
        >>>     terminations.time_out = ManagerTermCfg(func=time_out)
    """

    # RL-specific managers
    rewards: RewardManagerCfg = MISSING
    """Reward manager configuration."""

    terminations: TerminationManagerCfg = MISSING
    """Termination manager configuration."""

    commands: CommandManagerCfg | None = None
    """Command manager configuration (optional)."""

    # Episode settings
    episode_length_s: float = MISSING
    """Episode length in seconds."""

    # Rendering
    rerender_on_reset: bool = True
    """Whether to re-render after reset (for sensor updates)."""
