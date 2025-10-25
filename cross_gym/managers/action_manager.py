"""Action manager for processing actions."""

from __future__ import annotations

from abc import abstractmethod
from typing import List, TYPE_CHECKING

import torch

from cross_gym.utils.configclass import configclass
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ManagerTermCfg

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


@configclass
class ActionManagerCfg:
    """Configuration for action manager."""
    pass  # Terms are added as attributes dynamically


class ActionTerm(ManagerTermBase):
    """Base class for action terms.
    
    An action term processes raw actions from the policy and converts them to
    simulator commands (e.g., joint torques, position targets).
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action space for this term."""
        pass

    @property
    @abstractmethod
    def raw_actions(self) -> torch.Tensor:
        """The raw actions sent to this term."""
        pass

    @abstractmethod
    def process_actions(self, actions: torch.Tensor):
        """Process raw actions.
        
        This is called once per environment step.
        
        Args:
            actions: Raw actions from policy (num_envs, action_dim)
        """
        pass

    @abstractmethod
    def apply_actions(self):
        """Apply processed actions to the simulator.
        
        This is called once per simulation step (multiple times per env step if decimation > 1).
        """
        pass


class ActionManager(ManagerBase):
    """Manager for processing and applying actions.
    
    The action manager:
    1. Processes raw actions from the policy (once per environment step)
    2. Applies actions to the simulator (once per simulation step)
    
    Example:
        >>> # In task config
        >>> actions = ActionManagerCfg()
        >>> actions.joint_pos = JointPositionActionCfg(asset_name="robot", ...)
        >>> actions.gripper = GripperActionCfg(asset_name="robot", ...)
    """

    def __init__(self, cfg: ActionManagerCfg, env: ManagerBasedEnv):
        """Initialize action manager.
        
        Args:
            cfg: Configuration for action manager
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Parse configuration and create action terms
        self._prepare_terms()

    def _prepare_terms(self):
        """Parse configuration and create action terms."""
        # Get all attributes from config
        for attr_name in dir(self.cfg):
            if attr_name.startswith("_"):
                continue

            attr_value = getattr(self.cfg, attr_name)

            # Check if this is an action term config
            if isinstance(attr_value, ManagerTermCfg):
                # Create the term
                term_class = attr_value.func
                term = term_class(attr_value, self._env)
                self.active_terms[attr_name] = term

    @property
    def action_term_dim(self) -> List[int]:
        """Dimensions of each action term."""
        return [term.action_dim for term in self.active_terms.values()]

    def process_action(self, actions: torch.Tensor):
        """Process raw actions from policy.
        
        Args:
            actions: Raw actions (num_envs, total_action_dim)
        """
        # Split actions for each term
        action_dims = self.action_term_dim
        split_actions = torch.split(actions, action_dims, dim=-1)

        # Process each term
        for term, term_actions in zip(self.active_terms.values(), split_actions):
            term.process_actions(term_actions)

    def apply_action(self):
        """Apply processed actions to simulator."""
        for term in self.active_terms.values():
            term.apply_actions()

    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        """Reset action manager.
        
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
