"""Rollout storage for on-policy algorithms."""

from __future__ import annotations

from typing import Dict, Generator

import torch


class RolloutStorage:
    """Storage for on-policy rollouts (PPO, A2C, etc.).
    
    Stores transitions during rollout collection, then provides batches
    for policy updates.
    """

    def __init__(
            self,
            num_steps: int,
            num_envs: int,
            obs_shape: Dict[str, tuple],
            action_dim: int,
            device: torch.device,
    ):
        """Initialize rollout storage.
        
        Args:
            num_steps: Number of steps to store
            num_envs: Number of parallel environments
            obs_shape: Dictionary of observation shapes
            action_dim: Action dimension
            device: Torch device
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        # Current step index
        self.step = 0

        # Observations storage
        self.observations = {}
        for name, shape in obs_shape.items():
            if isinstance(shape, int):
                shape = (shape,)
            self.observations[name] = torch.zeros(
                num_steps, num_envs, *shape,
                device=device
            )

        # Transition storage
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=device)
        self.dones = torch.zeros(num_steps, num_envs, 1, dtype=torch.bool, device=device)
        self.values = torch.zeros(num_steps, num_envs, 1, device=device)

        # For PPO
        self.actions_log_prob = torch.zeros(num_steps, num_envs, 1, device=device)
        self.action_mean = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.action_std = torch.zeros(num_steps, num_envs, action_dim, device=device)

        # Computed during compute_returns()
        self.returns = torch.zeros(num_steps, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, 1, device=device)

    def add_transitions(
            self,
            observations: Dict[str, torch.Tensor],
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            actions_log_prob: torch.Tensor,
            action_mean: torch.Tensor,
            action_std: torch.Tensor,
    ):
        """Add transitions to storage.
        
        Args:
            observations: Observations dictionary
            actions: Actions taken
            rewards: Rewards received
            dones: Done flags
            values: Value estimates
            actions_log_prob: Log probability of actions
            action_mean: Mean of action distribution
            action_std: Std of action distribution
        """
        if self.step >= self.num_steps:
            raise RuntimeError("Storage is full. Call clear() before adding more transitions.")

        # Store observations
        for name, obs in observations.items():
            if name in self.observations:
                self.observations[name][self.step] = obs

        # Store transitions
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards
        self.dones[self.step] = (dones.unsqueeze(-1) if dones.dim() == 1 else dones).bool()
        self.values[self.step] = values
        self.actions_log_prob[self.step] = actions_log_prob
        self.action_mean[self.step] = action_mean
        self.action_std[self.step] = action_std

        self.step += 1

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        """Compute returns and advantages using GAE.
        
        Args:
            last_values: Value estimates for the last observation
            gamma: Discount factor
            lam: GAE lambda
        """
        # Generalized Advantage Estimation (GAE)
        gae = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0
            else:
                next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step + 1].float()

            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            gae = delta + gamma * lam * next_non_terminal * gae

            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Generate random mini-batches from the storage.
        
        Args:
            num_mini_batches: Number of mini-batches to create
            
        Yields:
            Dictionary containing batch data
        """
        batch_size = self.num_envs * self.num_steps
        mini_batch_size = batch_size // num_mini_batches

        # Flatten time and env dimensions
        observations_flat = {
            name: obs.flatten(0, 1) for name, obs in self.observations.items()
        }
        actions_flat = self.actions.flatten(0, 1)
        values_flat = self.values.flatten(0, 1)
        returns_flat = self.returns.flatten(0, 1)
        advantages_flat = self.advantages.flatten(0, 1)
        actions_log_prob_flat = self.actions_log_prob.flatten(0, 1)
        action_mean_flat = self.action_mean.flatten(0, 1)
        action_std_flat = self.action_std.flatten(0, 1)

        # Generate random mini-batches
        indices = torch.randperm(batch_size, device=self.device)

        for i in range(num_mini_batches):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            batch_idx = indices[start:end]

            yield {
                'observations': {name: obs[batch_idx] for name, obs in observations_flat.items()},
                'actions': actions_flat[batch_idx],
                'values': values_flat[batch_idx],
                'returns': returns_flat[batch_idx],
                'advantages': advantages_flat[batch_idx],
                'actions_log_prob': actions_log_prob_flat[batch_idx],
                'action_mean': action_mean_flat[batch_idx],
                'action_std': action_std_flat[batch_idx],
            }

    def clear(self):
        """Clear storage and reset step counter."""
        self.step = 0


__all__ = ["RolloutStorage"]
