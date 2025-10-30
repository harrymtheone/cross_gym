from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from bridge_rl.algorithms import make_linear_layers, BaseRecurrentActor


class VAE(nn.Module):
    def __init__(
            self,
            input_size: int,
            prop_size: int,
            vae_latent_size: int = 16,
            activation=nn.ELU(),
    ):
        super().__init__()

        self.encoder = make_linear_layers(input_size, 128, 64,
                                          activation_func=activation)

        self.mlp_vel = nn.Linear(64, 3)
        self.mlp_vel_logvar = nn.Linear(64, 3)

        self.mlp_z = nn.Linear(64, vae_latent_size)
        self.mlp_z_logvar = nn.Linear(64, vae_latent_size)

        self.decoder = make_linear_layers(vae_latent_size + 3, 64, prop_size,
                                          activation_func=activation,
                                          output_activation=False)

    def forward(self, x, sample=True):
        x = self.encoder(x)

        mu_vel, mu_z = self.mlp_vel(x), self.mlp_z(x)

        logvar_vel, logvar_z = self.mlp_vel_logvar(x), self.mlp_z_logvar(x)

        vel = self.reparameterize(mu_vel, logvar_vel) if sample else mu_vel
        z = self.reparameterize(mu_z, logvar_z) if sample else mu_z

        ot1 = self.decoder(torch.cat([vel, z], dim=-1))

        return vel, z, ot1, mu_vel, logvar_vel, mu_z, logvar_z

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class Policy(BaseRecurrentActor):
    def __init__(
            self,
            prop_size: int,
            gru_hidden_size: int,
            num_gru_layers: int,
            vae_latent_size: int,
            prop_next_size: int,
            actor_hidden_dims: Tuple[int, ...],
            action_size: int,
            activation=nn.ELU()
    ):
        super().__init__(action_size)

        self.gru = nn.GRU(
            input_size=prop_size,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers
        )
        self.hidden_states = None

        self.vae = VAE(
            input_size=gru_hidden_size * num_gru_layers,
            prop_size=prop_next_size,
            vae_latent_size=vae_latent_size,
            activation=activation
        )

        self.actor_backbone = make_linear_layers(
            vae_latent_size + 3 + prop_size, *actor_hidden_dims, action_size,
            activation_func=activation,
            output_activation=False
        )

    def init_hidden_states(self, num_envs: int, device: torch.device) -> None:
        self.hidden_states = torch.zeros(self.gru.num_layers, num_envs, self.gru.hidden_size, device=device)

    def get_hidden_states(self) -> torch.Tensor:
        return self.hidden_states

    def reset(self, dones: torch.Tensor) -> None:
        self.hidden_states[:, dones] = 0.

    def act(
            self,
            obs: dict[str, torch.Tensor],
            eval_: bool = False,
            **kwargs
    ):
        proprio = obs['proprio'].unsqueeze(0)

        # Process through GRU
        obs_enc, self.hidden_states[:] = self.gru(proprio, self.hidden_states)

        # Get VAE estimation
        vel, z = self.vae(obs_enc, sample=not eval_)[:2]

        # Concatenate proprioception with VAE output
        mean = self.actor_backbone(torch.cat([vel, z, proprio], dim=-1))

        mean, vel, z = mean.squeeze(0), vel.squeeze(0), z.squeeze(0)

        if eval_:
            return mean, vel, z

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample(), vel, z

    def actor_forward(self, proprio: torch.Tensor, vel: torch.Tensor, z: torch.Tensor):
        mean = self.actor_backbone(torch.cat([vel, z, proprio], dim=-1))
        self.distribution = Normal(mean, torch.exp(self.log_std))

    def vae_forward(
            self,
            obs: dict[str, torch.Tensor],
            hidden_states: torch.Tensor
    ):
        proprio = obs['proprio']
        obs_enc, _ = self.gru(proprio, hidden_states)
        return self.vae(obs_enc)
