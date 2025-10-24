"""Observation manager for computing observations."""

from __future__ import annotations

import torch
from typing import Any, Callable, Dict, TYPE_CHECKING, Tuple

from cross_gym.utils.configclass import configclass
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ManagerTermCfg

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


@configclass
class ObservationGroupCfg:
    """Configuration for an observation group.
    
    An observation group is a collection of observation terms that are
    concatenated together (e.g., "policy" observations, "critic" observations).
    """
    
    concatenate: bool = True
    """Whether to concatenate observations into a single tensor."""
    
    enable_corruption: bool = False
    """Whether to apply noise/corruption to observations."""


@configclass
class ObservationManagerCfg:
    """Configuration for observation manager.
    
    Example:
        >>> observations = ObservationManagerCfg()
        >>> observations.policy = ObservationGroupCfg()
        >>> observations.policy.base_lin_vel = ObservationTermCfg(func=base_lin_vel, ...)
        >>> observations.policy.joint_pos = ObservationTermCfg(func=joint_pos, ...)
    """
    pass  # Groups are added as attributes dynamically


class ObservationTerm(ManagerTermBase):
    """Base class for observation terms."""
    
    def __init__(self, cfg: ManagerTermCfg, env: ManagerBasedEnv):
        """Initialize observation term.
        
        Args:
            cfg: Term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)
        self._func: Callable = cfg.func
        self._params: dict = cfg.params
    
    def compute(self) -> torch.Tensor:
        """Compute the observation.
        
        Returns:
            Observation tensor (num_envs, obs_dim)
        """
        return self._func(self._env, **self._params)


class ObservationManager(ManagerBase):
    """Manager for computing observations.
    
    The observation manager computes observations from the current state of
    the environment. Observations can be grouped (e.g., policy vs. critic observations).
    """
    
    def __init__(self, cfg: ObservationManagerCfg, env: ManagerBasedEnv):
        """Initialize observation manager.
        
        Args:
            cfg: Configuration
            env: Environment instance
        """
        super().__init__(cfg, env)
        
        # Observation groups
        self.groups: Dict[str, Dict[str, ObservationTerm]] = {}
        self.group_obs_dim: Dict[str, Tuple[int, ...]] = {}
        self.group_obs_concatenate: Dict[str, bool] = {}
        
        # Parse configuration
        self._prepare_terms()
    
    def _prepare_terms(self):
        """Parse configuration and create observation terms."""
        # Get all groups from config
        for attr_name in dir(self.cfg):
            if attr_name.startswith("_"):
                continue
            
            attr_value = getattr(self.cfg, attr_name)
            
            # Check if this is an observation group
            if isinstance(attr_value, ObservationGroupCfg):
                group_name = attr_name
                self.groups[group_name] = {}
                self.group_obs_concatenate[group_name] = attr_value.concatenate
                
                # Parse terms in this group
                for term_name in dir(attr_value):
                    if term_name.startswith("_") or term_name in ["concatenate", "enable_corruption"]:
                        continue
                    
                    term_cfg = getattr(attr_value, term_name)
                    if isinstance(term_cfg, ManagerTermCfg):
                        term = ObservationTerm(term_cfg, self._env)
                        self.groups[group_name][term_name] = term
                        self.active_terms[f"{group_name}/{term_name}"] = term
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute all observations.
        
        Returns:
            Dictionary mapping group names to observations
        """
        obs_dict = {}
        
        for group_name, terms in self.groups.items():
            # Compute each term in the group
            term_obs = []
            for term in terms.values():
                obs = term.compute()
                term_obs.append(obs)
            
            # Concatenate or return as dict
            if self.group_obs_concatenate[group_name]:
                obs_dict[group_name] = torch.cat(term_obs, dim=-1)
            else:
                obs_dict[group_name] = {
                    name: obs for name, obs in zip(terms.keys(), term_obs)
                }
        
        return obs_dict
    
    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        """Reset observation manager.
        
        Args:
            env_ids: Environment IDs to reset
            
        Returns:
            Dictionary of reset information
        """
        info = {}
        for name, term in self.active_terms.items():
            term_info = term.reset(env_ids)
            if term_info:
                info[name] = term_info
        return info

