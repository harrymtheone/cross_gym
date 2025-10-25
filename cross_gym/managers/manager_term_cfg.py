"""Configuration for manager terms."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from cross_gym.utils.configclass import configclass


@configclass
class ManagerTermCfg:
    """Base configuration for manager terms.
    
    A term is a component of a manager (e.g., a specific reward function, observation, or action).
    """

    func: Callable = MISSING
    """The function/class to call for this term.
    
    For most terms, this should be a function that takes the environment as first argument.
    For action terms, this should be an ActionTerm subclass.
    """

    params: dict = {}
    """Parameters to pass to the function.
    
    These are passed as keyword arguments to the function.
    """

    weight: float = 1.0
    """Weight for this term (used in rewards/observations).
    
    For rewards: final_reward = sum(weight_i * reward_i)
    For observations: can be used for scaling
    """
