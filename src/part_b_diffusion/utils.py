"""
Utility functions for DDPM training and inference.
"""

import os
import yaml
import torch


def is_latent_mode(config: dict) -> bool:
    """Check if config specifies latent-space DDPM."""
    return config.get('mode', 'pixel') == 'latent'


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_path(config: dict) -> str:
    """Generate model path from config."""
    model_name = config['paths']['model_name']
    # Replace placeholders
    model_name = model_name.replace('{T}', str(config['model']['T']))
    if is_latent_mode(config):
        model_name = model_name.replace('{latent_dim}', str(config['vae']['latent_dim']))
        model_name = model_name.replace('{epochs}', str(config['training']['ddpm_epochs']))
    else:
        model_name = model_name.replace('{epochs}', str(config['training']['epochs']))
    return os.path.join(config['paths']['model_dir'], model_name)


def get_vae_path(config: dict) -> str:
    """Generate VAE model path from config."""
    vae_name = config['paths']['vae_name']
    vae_name = vae_name.replace('{beta}', str(config['vae']['beta']))
    vae_name = vae_name.replace('{latent_dim}', str(config['vae']['latent_dim']))
    return os.path.join(config['paths']['model_dir'], vae_name)


def preprocess_mnist(x: torch.Tensor) -> torch.Tensor:
    """Dequantize and scale MNIST images to [-1, 1]."""
    x = x.float()  # Already [0, 1] from ToTensor()
    x = x + torch.rand_like(x) / 256.0  # Dequantization noise
    x = (x - 0.5) * 2.0  # Scale to [-1, 1]
    return x
