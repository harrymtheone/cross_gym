from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any

import torch
import torch.optim as optim
from torch import GradScaler

from cross_gym.envs import VecEnv
from cross_gym_rl.algorithms.ppo import PPO, PPOCritic
from cross_gym_rl.storage import RolloutStorage
from .networks import Policy

if TYPE_CHECKING:
    from . import DreamWaQCfg


class DreamWaQ(PPO):
    cfg: DreamWaQCfg

    def __init__(self, cfg: DreamWaQCfg, env: VecEnv):
        super().__init__(cfg, env)

        # Loss coefficients
        self.kl_coef_vel = cfg.kl_coef_vel
        self.kl_coef_z = cfg.kl_coef_z

    def _init_components(self):
        # Initialize DreamWAQ actor
        proprio_shape = self.env.observation_manager.group_obs_dim['proprio']
        prop_next_shape = self.env.observation_manager.group_obs_dim['prop_next']
        action_size = self.env.action_manager.total_action_dim

        if len(proprio_shape) != 1:
            raise ValueError("Only proprio without history supported now")
        else:
            prop_size = proprio_shape[0]

        if len(prop_next_shape) != 1:
            raise ValueError("Only prop_next without history supported now")
        else:
            prop_next_size = prop_next_shape[0]

        self.actor = Policy(
            prop_size=prop_size,
            num_gru_layers=self.cfg.num_gru_layers,
            gru_hidden_size=self.cfg.gru_hidden_size,
            vae_latent_size=self.cfg.vae_latent_size,
            prop_next_size=prop_next_size,
            actor_hidden_dims=self.cfg.actor_hidden_dims,
            action_size=action_size,
        ).to(self.device)
        self.actor.init_hidden_states(self.env.num_envs, device=self.device)

        self.actor.reset_std(self.cfg.init_noise_std, self.device)

        # Initialize DreamWaQ critic
        critic_obs_shape = self.env.observation_manager.group_obs_dim['critic_obs']
        scan_shape = self.env.observation_manager.group_obs_dim['scan']

        self.critic = PPOCritic(
            critic_obs_shape=critic_obs_shape,
            scan_shape=scan_shape,
            critic_hidden_dims=self.cfg.critic_hidden_dims
        ).to(self.device)

        # Initialize DreamWaQ optimizer
        self.optimizer_ppo = optim.Adam(
            [*self.actor.actor_backbone.parameters(), *self.critic.parameters()],
            lr=self.learning_rate
        )
        self.optimizer_vae = optim.Adam(
            [*self.actor.gru.parameters(), *self.actor.vae.parameters()],
            lr=self.learning_rate
        )
        self.scaler_vae = GradScaler(enabled=self.cfg.use_amp)

        # Initialize DreamWaQ storage
        self.storage = RolloutStorage(
            obs_shape=self.env.observation_manager.group_obs_dim,
            num_actions=action_size,
            storage_length=self.cfg.num_steps_per_update,
            num_envs=self.num_envs,
            device=self.device
        )

        self.storage.register_hidden_state_buffer(
            'hidden_states', self.cfg.num_gru_layers, self.cfg.gru_hidden_size
        )
        self.storage.register_data_buffer(
            'vel_est', (3,), torch.float
        )
        self.storage.register_data_buffer(
            'z_est', (self.cfg.vae_latent_size,), torch.float
        )
        self.storage.register_data_buffer(
            'prop_next', prop_next_shape, torch.float
        )

    def act(self, observations, **kwargs) -> dict[str, torch.Tensor]:
        # Store observations and hidden states
        self.storage.add_transitions('observations', observations)
        self.storage.add_transitions('hidden_states', self.actor.get_hidden_states())

        # Generate actions
        actions, vel, z = self.actor.act(observations, **kwargs)

        # Store actor input
        self.storage.add_transitions('vel_est', vel)
        self.storage.add_transitions('z_est', z)

        # Evaluate using critic
        values = self.critic.evaluate(observations)

        # Store transition data
        self._store_transition_data(actions, values, **kwargs)

        return {'action': actions}  # TODO: after add support for single action

    def process_env_step(self, rewards, terminated, timeouts, infos, **kwargs):
        if self.actor.is_recurrent:
            self.actor.reset(terminated | timeouts)

        self.storage.add_transitions('prop_next', kwargs['obs_next']['prop_next'])

        super().process_env_step(rewards, terminated, timeouts, infos, **kwargs)

    def update(self, **kwargs) -> Dict[str, float]:
        """Main update loop for both PPO and VAE components."""
        # Clear statistics from previous update
        self.stats_tracker.clear()

        # Training loop
        for batch in self.storage.recurrent_mini_batch_generator(
                self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        ):
            # Extract batch data
            observations = batch['observations']

            # Forward pass through actor
            self.actor.actor_forward(observations['proprio'], batch['vel_est'], batch['z_est'])

            # Update PPO
            self._update_ppo(batch)

            # Update VAE
            self._update_vae(batch)

        self.storage.clear()

        # Get all mean statistics
        stats = self.stats_tracker.get_means()
        stats['Loss/learning_rate'] = self.learning_rate
        stats['Train/noise_std'] = self.actor.log_std.exp().mean().item()
        return stats

    def _update_vae(
            self,
            batch,
            target_snr_vel=2.0,  # Target SNR for velocity (mean/std ratio)
            target_snr_z=2.0,  # Target SNR for z (mean/std ratio)
    ):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            observations = batch['observations']
            prop_next = batch['prop_next']
            hidden_states = batch['hidden_states'] if self.actor.is_recurrent else None
            masks = batch['masks'] if self.actor.is_recurrent else slice(None)

            # Compute VAE forward pass using the vae_input (with recurrent wrapper)
            # VAE returns: z, vel, ot1, mu_z, logvar_z, mu_vel, logvar_vel
            vel, z, ot1, mu_vel, logvar_vel, mu_z, logvar_z = self.actor.vae_forward(observations, hidden_states)

            # Estimation loss
            vel_est_loss = masked_MSE(vel, observations['est_gt'], masks)
            ot1_pred_loss = masked_MSE(ot1, prop_next, masks)

            # KL loss
            vel_kl_loss = 1 + logvar_vel - mu_vel.pow(2) - logvar_vel.exp()
            vel_kl_loss = -0.5 * masked_mean(vel_kl_loss.sum(dim=2, keepdim=True), masks)
            z_kl_loss = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
            z_kl_loss = -0.5 * masked_mean(z_kl_loss.sum(dim=2, keepdim=True), masks)

            # Adaptive KL coefficient based on SNR (Signal-to-Noise Ratio)
            std_vel = logvar_vel.exp().sqrt().mean().item()
            std_z = logvar_z.exp().sqrt().mean().item()
            mean_abs_vel = mu_vel.abs().mean().item()  # Absolute mean for SNR calculation
            mean_abs_z = mu_z.abs().mean().item()  # Absolute mean for SNR calculation

            # Calculate SNR = |mean| / std (avoid division by zero)
            snr_vel = mean_abs_vel / (std_vel + 1e-8)
            snr_z = mean_abs_z / (std_z + 1e-8)

        # Adaptive KL coefficient based on SNR
        if snr_vel < target_snr_vel:
            self.kl_coef_vel = max(1e-7, self.kl_coef_vel / 1.1)
        else:
            self.kl_coef_vel = max(1e-7, self.kl_coef_vel * 1.1)

        if snr_z < target_snr_z:
            self.kl_coef_z = max(1e-7, self.kl_coef_z / 1.1)
        else:
            self.kl_coef_z = max(1e-7, self.kl_coef_z * 1.1)

        kl_loss = self.kl_coef_vel * vel_kl_loss + self.kl_coef_z * z_kl_loss

        # Total estimation loss
        total_loss = vel_est_loss + ot1_pred_loss + kl_loss

        # VAE gradient step
        self.optimizer_vae.zero_grad()
        self.scaler_vae.scale(total_loss).backward()
        self.scaler_vae.step(self.optimizer_vae)
        self.scaler_vae.update()

        # Track VAE statistics with proper labels
        self.stats_tracker.add("VAE/vel_est_loss", vel_est_loss)
        self.stats_tracker.add("VAE/ot1_pred_loss", ot1_pred_loss)

        # Track VAE mu and std statistics for both z and velocity
        self.stats_tracker.add("VAE_KL/vel_kl_loss", vel_kl_loss)
        self.stats_tracker.add("VAE_KL/z_kl_loss", z_kl_loss)
        self.stats_tracker.add("VAE_KL/abs_vel", mean_abs_vel)
        self.stats_tracker.add("VAE_KL/abs_z", mean_abs_z)
        self.stats_tracker.add("VAE_KL/std_vel", std_vel)
        self.stats_tracker.add("VAE_KL/std_z", std_z)
        self.stats_tracker.add("VAE_KL/SNR_vel", snr_vel)
        self.stats_tracker.add("VAE_KL/SNR_z", snr_z)
        self.stats_tracker.add("VAE_KL/kl_coef_vel", self.kl_coef_vel)
        self.stats_tracker.add("VAE_KL/kl_coef_z", self.kl_coef_z)

    def play_act(self, obs, **kwargs):
        """Generate actions for play/evaluation."""
        actions, vel, z = self.actor.act(obs, **kwargs)
        return {'actions': actions, 'est_vel': vel, 'est_z': z}

    def save(self) -> Dict[str, Any]:
        """Save model state."""
        save_dict = super().save()

        # Add DreamWAQ-specific state
        save_dict['optimizer_vae_state_dict'] = self.optimizer_vae.state_dict()

        return save_dict

    def load(self, loaded_dict: Dict[str, Any], load_optimizer: bool = True):
        """Load model state."""
        # Load base PPO state
        super().load(loaded_dict, load_optimizer)

        # Load VAE optimizer if available
        if load_optimizer and 'optimizer_vae_state_dict' in loaded_dict:
            self.optimizer_vae.load_state_dict(loaded_dict['optimizer_vae_state_dict'])
