"""Base class for all assets."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from cross_gym.sim import SimulationContext
from cross_gym.utils.configclass import configclass


@configclass
class AssetBaseCfg:
    """Base configuration for assets.
    
    This is the base configuration class for all assets in the scene.
    """

    # Prim path in the scene
    prim_path: str = "/World/envs/env_.*/Asset"

    # Initial state
    @configclass
    class InitStateCfg:
        """Initial state of the asset."""
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)

    init_state: InitStateCfg = InitStateCfg()

    # Collision group (-1 = global collision, 0+ = group ID)
    collision_group: int = 0


class AssetBase(ABC):
    """Base class for all assets.
    
    An asset represents any physical entity in the simulation scene, such as:
    - Articulated robots
    - Rigid objects
    - Deformable objects
    - Sensors
    
    This class provides a common interface for:
    - Resetting the asset state
    - Updating the asset (reading from simulation)
    - Writing data to simulation
    """

    def __init__(self, cfg: AssetBaseCfg):
        """Initialize the asset.
        
        Args:
            cfg: Configuration for the asset
        """
        self.cfg = cfg

        # Get simulator context
        self.sim = SimulationContext.instance()

        if self.sim is None:
            raise RuntimeError(
                "No SimulationContext found. Create a SimulationContext "
                "before initializing assets."
            )

        self.device = self.sim.device

        # Number of environments (will be set by scene)
        self.num_envs = 0

    @property
    @abstractmethod
    def num_instances(self) -> int:
        """Number of asset instances in the scene.
        
        Returns:
            Number of instances (usually same as num_envs for single robot per env)
        """
        pass

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the asset state for specified environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        pass

    @abstractmethod
    def update(self, dt: float):
        """Update the asset by reading the latest state from simulation.
        
        Args:
            dt: Time step in seconds
        """
        pass

    @abstractmethod
    def write_data_to_sim(self):
        """Write asset data to the simulation.
        
        This writes any buffered commands (e.g., joint torques) to the simulator.
        """
        pass
