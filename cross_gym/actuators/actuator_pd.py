"""PD actuator models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import ActuatorCommand, ActuatorBase

if TYPE_CHECKING:
    from . import IdealPDActuatorCfg


class IdealPDActuator(ActuatorBase):
    """Ideal PD actuator model.
    
    Computes torques using a simple PD control law:
        torque = kp * (target_pos - current_pos) + kd * (target_vel - current_vel) + feedforward_torque
    
    Then clips to effort limits.
    """

    cfg: IdealPDActuatorCfg

    def compute(
            self,
            command: ActuatorCommand,
            joint_pos: torch.Tensor,
            joint_vel: torch.Tensor,
    ) -> ActuatorCommand:
        """Compute torques using PD control.
        
        Args:
            command: Desired joint commands
            joint_pos: Current joint positions
            joint_vel: Current joint velocities
            
        Returns:
            Modified command with computed joint_efforts
        """
        # Initialize torque
        self.computed_torque.zero_()

        # Position error (if target position provided)
        if command.joint_positions is not None:
            error_pos = command.joint_positions - joint_pos
            self.computed_torque += self.stiffness * error_pos

        # Velocity error (if target velocity provided)
        if command.joint_velocities is not None:
            error_vel = command.joint_velocities - joint_vel
            self.computed_torque += self.damping * error_vel

        # Feedforward torque (if provided)
        if command.joint_efforts is not None:
            self.computed_torque += command.joint_efforts

        # Clip torques
        self.applied_torque = self._clip_torque(self.computed_torque)

        # Set computed torques into command
        command.joint_efforts = self.applied_torque
        # Clear position/velocity targets (we've converted to torque)
        command.joint_positions = None
        command.joint_velocities = None

        return command

    def reset(self, env_ids: torch.Tensor):
        """Reset actuator (no state to reset for ideal PD).
        
        Args:
            env_ids: Environment IDs to reset
        """
        pass
