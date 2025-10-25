"""Proximal Policy Optimization (PPO) algorithm implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any

import torch
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

from learning.algorithms.algorithm_base import AlgorithmBase
from learning.storage import RolloutStorage
from .networks import ActorCritic

if TYPE_CHECKING:
    from cross_gym.envs import ManagerBasedRLEnv
    from .ppo_cfg import PPOCfg

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


class PPO(AlgorithmBase):
    """Proximal Policy Optimization algorithm.
    
    Implements the clipped surrogate objective PPO algorithm with GAE.
    """

    cfg: PPOCfg

    def __init__(self, cfg: PPOCfg, env: ManagerBasedRLEnv):
        """Initialize PPO algorithm.
        
        Args:
            cfg: PPO configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Get observation and action shapes
        obs_shape = env.observation_manager.group_obs_dim
        action_dim = sum(env.action_manager.action_term_dim)

        # Create actor-critic network
        self.actor_critic = ActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
            activation=cfg.activation,
            init_noise_std=cfg.init_noise_std,
        ).to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=cfg.learning_rate
        )

        # Create rollout storage
        self.storage = RolloutStorage(
            num_steps=cfg.num_steps_per_update,
            num_envs=self.num_envs,
            obs_shape=obs_shape,
            action_dim=action_dim,
            device=self.device,
        )

        # Mixed precision training
        self.scaler = GradScaler(enabled=cfg.use_amp)

        # Learning rate (for adaptive schedule)
        self.learning_rate = cfg.learning_rate

    def act(self, observations: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Generate actions from observations.
        
        Args:
            observations: Dictionary of observations
            **kwargs: Additional arguments
            
        Returns:
            Actions (num_envs, action_dim)
        """
        return self.actor_critic.act(observations, eval_mode=False)

    def process_env_step(
            self,
            rewards: torch.Tensor,
            terminated: torch.Tensor,
            truncated: torch.Tensor,
            infos: Dict[str, Any],
            observations: Dict[str, torch.Tensor],
            **kwargs
    ):
        """Process environment step and store transition.
        
        Args:
            rewards: Rewards (num_envs,)
            terminated: Termination flags (num_envs,)
            truncated: Truncation flags (num_envs,)
            infos: Info dictionary
            observations: Current observations (before step)
            **kwargs: Additional arguments
        """
        # Get values and action info from actor-critic
        values = self.actor_critic.evaluate(observations)
        actions_log_prob = self.actor_critic.get_actions_log_prob(self.storage.actions[self.storage.step - 1])
        action_mean = self.actor_critic.action_mean
        action_std = self.actor_critic.action_std

        # Get the actions that were just taken (stored in previous step)
        actions = self.storage.actions[self.storage.step - 1]

        # Store transition
        dones = terminated | truncated
        self.storage.add_transitions(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            values=values,
            actions_log_prob=actions_log_prob,
            action_mean=action_mean,
            action_std=action_std,
        )

    def compute_returns(self, last_observations: Dict[str, torch.Tensor]):
        """Compute returns and advantages.
        
        Args:
            last_observations: Observations from last step (for bootstrap)
        """
        with torch.no_grad():
            last_values = self.actor_critic.evaluate(last_observations)

        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, **kwargs) -> Dict[str, float]:
        """Update policy using PPO.
        
        Returns:
            Dictionary of losses and metrics
        """
        # Track losses
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0

        # Multiple epochs over the same data
        for epoch in range(self.cfg.num_learning_epochs):
            # Generate mini-batches
            for batch in self.storage.mini_batch_generator(self.cfg.num_mini_batches):
                # Compute PPO loss
                losses = self._compute_ppo_loss(batch)

                # Backward pass
                self.optimizer.zero_grad()
                self.sctoggle.scale(losses['total_loss']).backward()

                # Gradient clipping
                if self.cfg.max_grad_norm is not None:
                    self.sctoggle.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(),
                        self.cfg.max_grad_norm
                    )

                # Optimizer step
                self.sctoggle.step(self.optimizer)
                self.scaler.update()

                # Accumulate losses for logging
                mean_value_loss += losses['value_loss'].item()
                mean_surrogate_loss += losses['surrogate_loss'].item()
                mean_entropy_loss += losses['entropy_loss'].item()
                if 'kl' in losses:
                    mean_kl += losses['kl']

        # Average over all mini-batches and epochs
        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        if mean_kl > 0:
            mean_kl /= num_updates

        # Adaptive learning rate
        if self.cfg.learning_rate_schedule == 'adaptive' and self.cfg.desired_kl is not None:
            if mean_kl > self.cfg.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif mean_kl < self.cfg.desired_kl / 2.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        # Clip noise std
        self.actor_critic.clip_std(*self.cfg.noise_std_range)

        # Clear storage
        self.storage.clear()

        return {
            'Loss/value_loss': mean_value_loss,
            'Loss/surrogate_loss': mean_surrogate_loss,
            'Loss/entropy_loss': mean_entropy_loss,
            'Train/mean_kl': mean_kl,
            'Train/learning_rate': self.learning_rate,
            'Train/action_std': self.actor_critic.log_std.exp().mean().item(),
        }

    def _compute_ppo_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute PPO losses for a mini-batch.
        
        Args:
            batch: Mini-batch dictionary
            
        Returns:
            Dictionary of losses
        """
        # Extract batch data
        observations = batch['observations']
        actions = batch['actions']
        old_values = batch['values']
        returns = batch['returns']
        advantages = batch['advantages']
        old_actions_log_prob = batch['actions_log_prob']
        old_action_mean = batch['action_mean']
        old_action_std = batch['action_std']

        # Forward pass
        action_mean = self.actor_critic.actor(observations['policy'] if 'policy' in observations else observations[list(observations.keys())[0]])
        values = self.actor_critic.evaluate(observations)

        # Recompute action distribution
        std = torch.exp(self.actor_critic.log_std)
        self.actor_critic.distribution = Normal(action_mean, std)
        actions_log_prob = self.actor_critic.get_actions_log_prob(actions)
        entropy = self.actor_critic.entropy

        # Compute KL divergence for adaptive learning rate
        kl = 0
        if self.cfg.learning_rate_schedule == 'adaptive':
            with torch.no_grad():
                kl_dist = kl_divergence(
                    Normal(old_action_mean, old_action_std),
                    Normal(action_mean, std)
                )
                kl = kl_dist.sum(dim=-1, keepdim=True).mean().item()

        # PPO clipped surrogate loss
        ratio = torch.exp(actions_log_prob - old_actions_log_prob)
        surrogate = advantages * ratio
        surrogate_clipped = advantages * ratio.clamp(
            1.0 - self.cfg.clip_param,
            1.0 + self.cfg.clip_param
        )
        surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

        # Value loss
        if self.cfg.use_clipped_value_loss:
            value_clipped = old_values + (values - old_values).clamp(
                -self.cfg.clip_param,
                self.cfg.clip_param
            )
            value_loss = (returns - values).pow(2)
            value_loss_clipped = (returns - value_clipped).pow(2)
            value_loss = torch.max(value_loss, value_loss_clipped).mean()
        else:
            value_loss = (returns - values).pow(2).mean()

        # Entropy loss (bonus)
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
                surrogate_loss
                + self.cfg.value_loss_coef * value_loss
                + self.cfg.entropy_coef * entropy_loss
        )

        return {
            'total_loss': total_loss,
            'surrogate_loss': surrogate_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'kl': kl,
        }

    def reset(self, env_ids: torch.Tensor):
        """Reset algorithm state for specified environments.
        
        Args:
            env_ids: Environment IDs to reset
        """
        # For feedforward policies, no state to reset
        pass


__all__ = ["PPO"]
