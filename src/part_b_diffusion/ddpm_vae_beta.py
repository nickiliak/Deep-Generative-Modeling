"""
DDPM in β-VAE Latent Space

Trains a DDPM on latent representations from a β-VAE with Gaussian decoder.
The VAE encoder is frozen during DDPM training.
"""

import torch
import torch.nn as nn
import numpy as np

from part_b_diffusion.ddpm import DDPM, FcNetwork
from part_a_vaes.vae_priors_temp import GaussianPrior, Encoder, GaussianDecoder, VAE


def build_vae(latent_dim: int, beta: float, decoder_std: float = 0.5, device: str = 'cpu') -> VAE:
    """
    Build a β-VAE with Gaussian decoder for MNIST.
    
    Parameters:
    latent_dim: Dimension of latent space
    beta: KL weight for β-VAE
    decoder_std: Standard deviation for Gaussian decoder
    device: Device to place model on
    """
    # Encoder network
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, latent_dim * 2),
    )
    
    # Decoder network
    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )
    
    prior = GaussianPrior(latent_dim)
    encoder = Encoder(encoder_net)
    decoder = GaussianDecoder(decoder_net, std=decoder_std, learnable_std=False)
    
    return VAE(prior, decoder, encoder, beta=beta).to(device)


class DDPMLatent(nn.Module):
    """
    DDPM trained on latent representations from a β-VAE.
    
    Workflow:
    1. VAE encoder: x → q(z|x) → z
    2. DDPM: operates on z (denoise latent)
    3. VAE decoder: z → p(x|z) → x̂
    
    The VAE encoder is frozen during DDPM training.
    """
    
    def __init__(self, vae: VAE, ddpm: DDPM, freeze_encoder: bool = True):
        """
        Parameters:
        vae: Pre-trained β-VAE with Gaussian decoder
        ddpm: DDPM model for latent space
        freeze_encoder: Freeze VAE encoder (default: True)
        """
        super().__init__()
        self.vae = vae
        self.ddpm = ddpm
        
        if freeze_encoder:
            for param in self.vae.encoder.parameters():
                param.requires_grad = False
            for param in self.vae.decoder.parameters():
                param.requires_grad = False
            self.vae.eval()
    
    @property
    def latent_dim(self) -> int:
        """Get latent dimension from VAE prior."""
        return self.vae.prior.M
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent vectors.
        
        Parameters:
        x: Images, shape (batch, 28, 28) or (batch, 784)
        
        Returns:
        z: Latent vectors, shape (batch, latent_dim)
        """
        with torch.no_grad():
            q = self.vae.encoder(x)
            z = q.rsample()
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.
        
        Parameters:
        z: Latent vectors, shape (batch, latent_dim)
        
        Returns:
        x_recon: Reconstructed images, shape (batch, 28, 28)
        """
        with torch.no_grad():
            # Call decoder directly without clamping
            mean = self.vae.decoder.decoder_net(z)
            
            print(f"[DEBUG] Decoder output - min: {mean.min():.4f}, max: {mean.max():.4f}, mean: {mean.mean():.4f}")
            print(f"[DEBUG] Decoder output - has NaN: {torch.isnan(mean).any()}")
            
            # Only replace obvious errors (NaN/Inf)
            if torch.isnan(mean).any():
                print("[WARNING] NaN detected in decoder output, replacing with zeros")
                mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
            
            if torch.isinf(mean).any():
                print("[WARNING] Inf detected in decoder output, replacing with zeros")
                mean = torch.where(torch.isinf(mean), torch.zeros_like(mean), mean)
            
        return mean
    
    def negative_elbo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute DDPM loss on latent representations.
        
        Parameters:
        x: Batch of images
        
        Returns:
        loss: DDPM negative ELBO in latent space, shape (batch,)
        """
        z = self.encode(x)
        return self.ddpm.negative_elbo(z)
    
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Mean loss for a batch."""
        return self.negative_elbo(x).mean()
    
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """
        Generate images: sample from DDPM in latent space → decode.
        
        Parameters:
        n_samples: Number of samples to generate
        
        Returns:
        x_samples: Generated images, shape (n_samples, 28, 28)
        """
        with torch.no_grad():
            # Sample from DDPM in latent space
            z_samples = self.ddpm.sample((n_samples, self.latent_dim))
            
            print(f"[DEBUG] DDPM samples - min: {z_samples.min():.4f}, max: {z_samples.max():.4f}, mean: {z_samples.mean():.4f}")
            print(f"[DEBUG] DDPM samples - has NaN: {torch.isnan(z_samples).any()}, has Inf: {torch.isinf(z_samples).any()}")
            
            # Decode to images
            x_samples = self.decode(z_samples)
            
            print(f"[DEBUG] Decoded samples - min: {x_samples.min():.4f}, max: {x_samples.max():.4f}, mean: {x_samples.mean():.4f}")
            
        return x_samples
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute loss (for training)."""
        return self.loss(x)
