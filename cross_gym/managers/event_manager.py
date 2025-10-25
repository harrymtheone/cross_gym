"""Event manager for handling randomization and events."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Tuple

import torch

from cross_gym.utils.configclass import configclass
from .manager_base import ManagerBase
from .manager_term_cfg import ManagerTermCfg

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


@configclass
class EventManagerCfg:
    """Configuration for event manager.
    
    Events are operations triggered at specific times:
    - startup: Once at environment creation
    - reset: When environments are reset
    - interval: At fixed intervals during episode
    """
    pass  # Event terms are added as attributes


class EventManager(ManagerBase):
    """Manager for handling randomization events.
    
    Events are operations that happen at specific times:
    - Domain randomization (e.g., randomize masses, friction)
    - External perturbations (e.g., push robot)
    - Scene modifications (e.g., spawn/remove objects)
    
    Events are organized by mode:
    - "startup": Called once when environment is created
    - "reset": Called when environments are reset
    - "interval": Called at fixed time intervals
    """

    def __init__(self, cfg: EventManagerCfg, env: ManagerBasedEnv):
        """Initialize event manager.
        
        Args:
            cfg: Configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Event terms organized by mode
        self.event_terms: Dict[str, Dict[str, Tuple[Callable, dict]]] = {
            "startup": {},
            "reset": {},
            "interval": {},
        }

        # Parse configuration
        self._prepare_terms()

    def _prepare_terms(self):
        """Parse configuration and create event terms."""
        for attr_name in dir(self.cfg):
            if attr_name.startswith("_"):
                continue

            attr_value = getattr(self.cfg, attr_name)

            if isinstance(attr_value, ManagerTermCfg):
                func = attr_value.func
                params = attr_value.params if hasattr(attr_value, 'params') else {}

                # Determine mode from name or params
                mode = params.get("mode", "reset")

                self.event_terms[mode][attr_name] = (func, params)
                self.active_terms[attr_name] = attr_value

    @property
    def available_modes(self) -> List[str]:
        """Get list of modes that have active events."""
        return [mode for mode, terms in self.event_terms.items() if len(terms) > 0]

    def apply(
            self,
            mode: str,
            env_ids: torch.Tensor | None = None,
            dt: Optional[float] = None,
            **kwargs
    ):
        """Apply events for specified mode.
        
        Args:
            mode: Event mode ("startup", "reset", "interval")
            env_ids: Environment IDs to apply events to (for "reset" mode)
            dt: Time step (for "interval" mode)
            **kwargs: Additional arguments
        """
        if mode not in self.event_terms:
            return

        # Apply each event term for this mode
        for name, (func, params) in self.event_terms[mode].items():
            # Build arguments for the function
            func_args = {"env": self._env}
            if env_ids is not None:
                func_args["env_ids"] = env_ids
            if dt is not None:
                func_args["dt"] = dt
            func_args.update(params)
            func_args.update(kwargs)

            # Call the event function
            func(**func_args)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        """Reset event manager.
        
        Args:
            env_ids: Environment IDs to reset
            
        Returns:
            Dictionary of reset information
        """
        return {}
