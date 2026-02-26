"""Part A: VAE experiments with different priors (Mean-field, MoG, Flow)."""

from .models import VAE, MixtureOfGaussians, NormalizingFlow
from .experiments import (
    train_vae_with_standard_prior,
    train_vae_with_mog_prior,
    train_vae_with_flow_prior,
    compute_elbo,
    compute_fid_scores,
)

__all__ = [
    "VAE",
    "MixtureOfGaussians",
    "NormalizingFlow",
    "train_vae_with_standard_prior",
    "train_vae_with_mog_prior",
    "train_vae_with_flow_prior",
    "compute_elbo",
    "compute_fid_scores",
]
