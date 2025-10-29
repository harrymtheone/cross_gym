from cross_gym.utils import configclass
from bridge_rl.algorithms import PPOCfg

from . import DreamWaQ


@configclass
class DreamWaQCfg(PPOCfg):
    class_type: type = DreamWaQ

    observations = DreamWaQObservations

    # -- Actor Critic
    num_gru_layers: int = 1
    gru_hidden_size: int = 128
    actor_hidden_dims: tuple = (512, 256, 128)
    critic_hidden_dims: tuple = (512, 256, 128)

    # -- VAE
    vae_latent_size: int = 16
    vel_est_loss_coef: float = 1.0
    ot1_pred_loss_coef: float = 1.0
    kl_coef_vel: float = 1.0
    kl_coef_z: float = 1.0

    # other parameters
    symmetry_loss_coef: float = 1.0
