"""Termination manager for checking episode terminations."""

from __future__ import annotations

import torch
from typing import Callable, Dict, TYPE_CHECKING, Tuple

from cross_gym.utils.configclass import configclass
from .manager_base import ManagerBase
from .manager_term_cfg import ManagerTermCfg

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


@configclass
class TerminationManagerCfg:
    """Configuration for termination manager.
    
    Example:
        >>> terminations = TerminationManagerCfg()
        >>> terminations.time_out = ManagerTermCfg(func=time_out, ...)
        >>> terminations.base_contact = ManagerTermCfg(func=base_contact, ...)
    """
    pass  # Terms are added as attributes dynamically


class TerminationManager(ManagerBase):
    """Manager for checking termination conditions.
    
    The termination manager checks various conditions and determines if
    episodes should be terminated. It distinguishes between:
    - Terminations: Task failure (e.g., robot fell)
    - Time-outs: Episode length limit reached
    """
    
    def __init__(self, cfg: TerminationManagerCfg, env: ManagerBasedEnv):
        """Initialize termination manager.
        
        Args:
            cfg: Configuration
            env: Environment instance
        """
        super().__init__(cfg, env)
        
        # Termination terms
        self.termination_terms: Dict[str, Tuple[Callable, dict]] = {}
        
        # Parse configuration
        self._prepare_terms()
        
        # Termination buffers
        self.terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.time_outs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    def _prepare_terms(self):
        """Parse configuration and create termination terms."""
        for attr_name in dir(self.cfg):
            if attr_name.startswith("_"):
                continue
            
            attr_value = getattr(self.cfg, attr_name)
            
            if isinstance(attr_value, ManagerTermCfg):
                func = attr_value.func
                params = attr_value.params if hasattr(attr_value, 'params') else {}
                
                self.termination_terms[attr_name] = (func, params)
                self.active_terms[attr_name] = attr_value
    
    def compute(self) -> torch.Tensor:
        """Check termination conditions.
        
        Returns:
            Boolean tensor indicating which environments should be reset (num_envs,)
        """
        # Reset buffers
        self.terminated.fill_(False)
        self.time_outs.fill_(False)
        
        # Check each termination term
        for name, (func, params) in self.termination_terms.items():
            # Call termination function
            term_result = func(self._env, **params)
            
            # Check if this is a time-out
            if "time_out" in name.lower():
                self.time_outs |= term_result
            else:
                self.terminated |= term_result
        
        # Return combined reset signal
        return self.terminated | self.time_outs
    
    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        """Reset termination manager.
        
        Args:
            env_ids: Environment IDs to reset
            
        Returns:
            Dictionary of termination statistics
        """
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        
        # Clear termination flags for reset environments
        if len(env_ids) > 0:
            self.terminated[env_ids] = False
            self.time_outs[env_ids] = False
        
        return {}

