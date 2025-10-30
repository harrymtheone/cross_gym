"""Ideal PD actuator implementation."""

from __future__ import annotations

import torch

from cross_core.base import ActuatorBase, ActuatorCommand


class IdealPDActuator(ActuatorBase):
    """Ideal PD actuator that computes torques from position/velocity errors.
    
    The actuator computes torques as:
        tau = kp * (q_target - q) + kd * (dq_target - dq) + tau_ff
    """

    def compute(
            self,
            command: ActuatorCommand,
            joint_pos: torch.Tensor,
            joint_vel: torch.Tensor,
    ) -> ActuatorCommand:
        """Compute PD torques from command and state.
        
        Args:
            command: Target positions, velocities, efforts (feedforward)
            joint_pos: Current joint positions (num_envs, num_joints)
            joint_vel: Current joint velocities (num_envs, num_joints)
            
        Returns:
            Modified command with joint_efforts set to computed torques
        """
        # Initialize computed torque
        self.computed_torque.zero_()

        # PD control: tau = kp * (q_target - q) + kd * (dq_target - dq)
        if command.joint_positions is not None:
            pos_error = command.joint_positions - joint_pos
            self.computed_torque.add_(self.stiffness * pos_error)

        if command.joint_velocities is not None:
            vel_error = command.joint_velocities - joint_vel
            self.computed_torque.add_(self.damping * vel_error)

        # Add feedforward effort if provided
        if command.joint_efforts is not None:
            self.computed_torque.add_(command.joint_efforts)

        # Clip to effort limits
        torch.clamp(
            self.computed_torque,
            -self.effort_limit,
            self.effort_limit,
            out=self.applied_torque
        )

        # Return command with computed efforts
        command.joint_efforts.copy_(self.applied_torque)
        command.joint_positions = None  # Clear pos target
        command.joint_velocities = None  # Clear vel target
        
        return command

    def reset(self, env_ids: torch.Tensor):
        """Reset actuator state (no internal state for ideal PD)."""
        pass
