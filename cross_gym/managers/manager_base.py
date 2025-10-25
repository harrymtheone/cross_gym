"""Base classes for managers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


class ManagerTermBase(ABC):
    """Base class for manager terms.
    
    A term is a specific component of a manager, such as:
    - An action term (processes raw actions)
    - An observation term (computes a specific observation)
    - A reward term (computes a specific reward component)
    """

    def __init__(self, cfg: Any, env: ManagerBasedEnv):
        """Initialize the term.
        
        Args:
            cfg: Configuration for this term
            env: The environment instance
        """
        self.cfg = cfg
        self._env = env

    def reset(self, env_ids: Any = None) -> dict:
        """Reset the term for specified environments.
        
        Args:
            env_ids: Environment IDs to reset
            
        Returns:
            Dictionary of reset information (optional)
        """
        return {}

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}"


class ManagerBase(ABC):
    """Base class for all managers.
    
    Managers orchestrate groups of terms (action terms, observation terms, etc.)
    and provide a unified interface for the environment.
    """

    def __init__(self, cfg: Any, env: ManagerBasedEnv):
        """Initialize the manager.
        
        Args:
            cfg: Configuration for this manager
            env: The environment instance
        """
        self.cfg = cfg
        self._env = env

        # Active terms (filled by subclasses)
        self.active_terms: Dict[str, Any] = {}

    @abstractmethod
    def reset(self, env_ids: Any = None) -> dict:
        """Reset the manager for specified environments.
        
        Args:
            env_ids: Environment IDs to reset
            
        Returns:
            Dictionary of reset information
        """
        pass

    def __str__(self) -> str:
        """String representation."""
        msg = f"<{self.__class__.__name__}>\n"
        msg += f"  Active terms: {list(self.active_terms.keys())}"
        return msg

    def __repr__(self) -> str:
        """Representation."""
        return str(self)
