"""Bridge RL: Reinforcement learning framework for Cross-Gym and other Gym environments.

This is a standalone RL framework that works with Cross-Gym environments
but can also be used with other Gymnasium-compatible environments.
"""

from .algorithms import AlgorithmBase, PPO, PPOCfg
from .algorithms.ppo import ActorCritic, ActorCriticCfg
from .modules import make_mlp, get_activation
from .runners import OnPolicyRunner, OnPolicyRunnerCfg
from .storage import RolloutStorage
from .utils import Logger, EpisodeLogger, masked_mean, masked_MSE

__all__ = [
    # Base classes
    "AlgorithmBase",
    # Algorithms
    "PPO",
    "PPOCfg",
    # Modules
    "ActorCritic",
    "ActorCriticCfg",
    "make_mlp",
    "get_activation",
    # Runners
    "OnPolicyRunner",
    "OnPolicyRunnerCfg",
    # Storage
    "RolloutStorage",
    # Utils
    "Logger",
    "EpisodeLogger",
    "masked_mean",
    "masked_MSE",
]

__version__ = "0.1.0"
