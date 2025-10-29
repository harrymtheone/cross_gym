"""T1 DreamWAQ environment - humanoid locomotion with terrain curriculum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cross_tasks.direct_rl.base import HumanoidEnv

if TYPE_CHECKING:
    from . import T1DreamWaqCfg


class T1DreamWaqEnv(HumanoidEnv):
    """T1 robot environment for DreamWAQ locomotion.
    
    Inherits from HumanoidEnv and provides T1-specific configurations.
    """

    cfg: T1DreamWaqCfg

    def __init__(self, cfg: T1DreamWaqCfg):
        """Initialize T1 DreamWAQ environment.
        
        Args:
            cfg: T1 DreamWAQ configuration
        """
        super().__init__(cfg)
        
        # T1-specific joint indices
        self.yaw_roll_dof_indices = torch.tensor(
            self.robot.find_joints(['Waist', 'Roll', 'Yaw']),
            dtype=torch.long,
            device=self.device
        )

    def compute_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations for T1 DreamWAQ.
        
        Returns:
            Dictionary with 'policy' observations
        """
        # Get clock inputs for gait
        if self.cfg.gait is not None:
            clock = torch.stack(self._get_clock_input(), dim=1)  # (num_envs, 2)
            command_input = torch.cat((clock, self.commands[:, :3]), dim=1)  # (num_envs, 5)
        else:
            command_input = self.commands[:, :3]  # (num_envs, 3)

        # Proprio observation
        obs = torch.cat((
            self.base_ang_vel,  # 3
            self.projected_gravity,  # 3
            command_input,  # 5 (with clock) or 3 (without)
            (self.robot.data.dof_pos - self.robot.data.default_joint_pos),  # num_dof
            self.robot.data.dof_vel,  # num_dof
            self.actions,  # num_actions (last action)
        ), dim=-1)

        return {"policy": obs}


__all__ = ["T1DreamWaqEnv"]

