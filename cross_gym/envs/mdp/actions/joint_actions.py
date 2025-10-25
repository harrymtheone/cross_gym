"""Joint-level action terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cross_gym.managers import ActionTerm
from cross_gym.managers.manager_term_cfg import ManagerTermCfg
from cross_gym.utils.configclass import configclass

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedEnv


@configclass
class JointActionCfg(ManagerTermCfg):
    """Base configuration for joint actions."""

    asset_name: str = "robot"
    """Name of the articulation asset to control."""

    joint_names: list = []
    """List of joint names to control. If empty, control all joints."""

    scale: float = 1.0
    """Scale factor for actions."""

    offset: float = 0.0
    """Offset to add to actions."""


class JointPositionAction(ActionTerm):
    """Action term for joint position control.
    
    This action term converts policy actions to joint position targets.
    It can be used with PD controllers or position-controlled actuators.
    """

    cfg: JointActionCfg

    def __init__(self, cfg: JointActionCfg, env: ManagerBasedEnv):
        """Initialize joint position action.
        
        Args:
            cfg: Configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Get the articulation
        self._asset = env.scene[cfg.asset_name]

        # Determine which joints to control
        if cfg.joint_names:
            # Specific joints
            self._joint_ids = [
                i for i, name in enumerate(self._asset.dof_names)
                if name in cfg.joint_names
            ]
            self._num_joints = len(self._joint_ids)
        else:
            # All joints
            self._joint_ids = None
            self._num_joints = self._asset.num_dof

        # Action buffers
        self._raw_actions = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._asset.num_dof, device=env.device)

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions sent to this term."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions.
        
        Args:
            actions: Raw actions from policy (num_envs, num_joints)
        """
        # Store raw actions
        self._raw_actions[:] = actions

        # Apply scale and offset
        scaled_actions = self.cfg.scale * actions + self.cfg.offset

        # Map to full joint space
        if self._joint_ids is not None:
            # Only control specific joints
            self._processed_actions[:, self._joint_ids] = scaled_actions
        else:
            # Control all joints
            self._processed_actions[:] = scaled_actions

    def apply_actions(self):
        """Apply processed actions to the articulation.
        
        For position control, this sets the position targets.
        The actual PD control happens in the actuator/simulator.
        """
        # Set position targets (for PD control)
        self._asset.set_joint_position_target(self._processed_actions)


class JointEffortAction(ActionTerm):
    """Action term for direct joint torque/effort control.
    
    This action term directly sets joint torques without any intermediate
    controller. This is the most direct form of control.
    """

    cfg: JointActionCfg

    def __init__(self, cfg: JointActionCfg, env: ManagerBasedEnv):
        """Initialize joint effort action.
        
        Args:
            cfg: Configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Get the articulation
        self._asset = env.scene[cfg.asset_name]

        # Determine which joints to control
        if cfg.joint_names:
            # Specific joints
            self._joint_ids = [
                i for i, name in enumerate(self._asset.dof_names)
                if name in cfg.joint_names
            ]
            self._num_joints = len(self._joint_ids)
        else:
            # All joints
            self._joint_ids = None
            self._num_joints = self._asset.num_dof

        # Action buffers
        self._raw_actions = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._asset.num_dof, device=env.device)

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions sent to this term."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions (joint torques)."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions into joint torques.
        
        Args:
            actions: Raw actions from policy (num_envs, num_joints)
        """
        # Store raw actions
        self._raw_actions[:] = actions

        # Apply scale and offset
        scaled_actions = self.cfg.scale * actions + self.cfg.offset

        # Map to full joint space
        if self._joint_ids is not None:
            # Only control specific joints
            self._processed_actions.zero_()
            self._processed_actions[:, self._joint_ids] = scaled_actions
        else:
            # Control all joints
            self._processed_actions[:] = scaled_actions

    def apply_actions(self):
        """Apply torques directly to the articulation."""
        # Set torques in the articulation data
        self._asset.set_joint_effort_target(self._processed_actions)
