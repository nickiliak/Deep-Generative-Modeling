#!/usr/bin/env python3
"""
DDPM Runner - Load config from YAML and run training/sampling/evaluation.
Usage (from repo root):
    python -m src.part_b_diffusion.run_ddpm train --config configs/ddpm_unet.yaml
    python -m src.part_b_diffusion.run_ddpm sample --config configs/ddpm_unet.yaml
    python -m src.part_b_diffusion.run_ddpm fid --config configs/ddpm_unet.yaml
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import datetime
import yaml
import torch
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
import time

from part_b_diffusion.ddpm import DDPM, FcNetwork, train
from part_b_diffusion.unet import Unet
from part_b_diffusion.ddpm_vae_beta import DDPMLatent, build_vae
from part_a_vaes.vae_priors_temp import VAE
from shared.fid_utils import compute_fid
from shared.data_loader import get_mnist_raw


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

def get_data_loaders(config: dict):
    """Create train and test data loaders."""
    batch_size = config['data']['batch_size']
    
    train_dataset = get_mnist_raw(split='train')
    test_dataset = get_mnist_raw(split='test')
    
    def collate_fn(batch):
        images, _ = zip(*batch)
        images = torch.stack(images)
        images = preprocess_mnist(images).reshape(len(images), -1)
        return images
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader


def get_data_loaders_2d(config: dict):
    """Create train/test data loaders for VAE (keeps 28x28 shape)."""
    batch_size = config['data']['batch_size']
    
    train_dataset = get_mnist_raw(split='train')
    test_dataset = get_mnist_raw(split='test')
    
    def collate_fn(batch):
        images, _ = zip(*batch)
        images = torch.stack(images)
        # Keep 28x28 shape (squeeze channel dim from (b,1,28,28) to (b,28,28))
        return images.squeeze(1)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader


def build_model(config: dict, device: str) -> DDPM:
    """Build DDPM model from config."""
    network_type = config['model']['network']
    
    if network_type == 'unet':
        network = Unet()
    elif network_type == 'fc':
        network = FcNetwork(28 * 28, 64)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    model = DDPM(
        network=network,
        beta_1=config['model']['beta_1'],
        beta_T=config['model']['beta_T'],
        T=config['model']['T']
    )
    return model.to(device)


def run_train(config: dict):
    """Train the DDPM model."""
    device = config['training']['device']
    
    print("=" * 60)
    print("DDPM Training")
    print("=" * 60)
    for section, values in config.items():
        print(f"\n[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {values}")
    print("=" * 60)
    
    # Create data loaders
    train_loader, _ = get_data_loaders(config)
    
    # Build model
    model = build_model(config, device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # Train
    train(model, optimizer, train_loader, config['training']['epochs'], device)
    
    # Save model
    model_path = get_model_path(config)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n[OK] Model saved -> {model_path}")


def run_sample(config: dict):
    """Generate samples from trained model."""
    device = config['training']['device']
    model_path = get_model_path(config)
    
    # Build and load model
    model = build_model(config, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    num_samples = config['sampling']['num_samples']
    D = 28 * 28
    
    print(f"Generating {num_samples} samples...")
    # Time sampling (wall-clock). Make sure to sync CUDA if used.
    device_str = device if isinstance(device, str) else str(device)
    with torch.no_grad():
        start = time.perf_counter()
        samples = model.sample((num_samples, D)).cpu()
        if 'cuda' in device_str:
            torch.cuda.synchronize()
        end = time.perf_counter()
    elapsed = end - start
    s_per = num_samples / elapsed if elapsed > 0 else float('inf')
    print(f"[TIMING] DDPM sampling: {elapsed:.4f} s for {num_samples} samples ({s_per:.2f} samples/s)")
    
    print(f"[DEBUG] Raw samples - min: {samples.min():.4f}, max: {samples.max():.4f}, mean: {samples.mean():.4f}")
    
    # Transform back to [0, 1]
    samples = samples / 2 + 0.5
    print(f"[DEBUG] After scaling - min: {samples.min():.4f}, max: {samples.max():.4f}, mean: {samples.mean():.4f}")
    
    samples = samples.clamp(0, 1)
    print(f"[DEBUG] After clamping - min: {samples.min():.4f}, max: {samples.max():.4f}, mean: {samples.mean():.4f}")
    
    samples = samples.reshape(num_samples, 1, 28, 28)
     
    # Save samples
    samples_dir = config['paths']['samples_dir']
    os.makedirs(samples_dir, exist_ok=True)
    
    grid_size = config['sampling']['grid_size']
    sample_path = os.path.join(samples_dir, f"ddpm_{config['model']['network']}_samples.png")
    save_image(samples[:grid_size**2], sample_path, nrow=grid_size)
    print(f"[OK] Samples saved -> {sample_path}")


def run_fid(config: dict):
    """Compute FID score and save results."""
    device = config['training']['device']
    model_path = get_model_path(config)
    
    # Build and load model
    model = build_model(config, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get real test data
    _, test_loader = get_data_loaders(config)
    
    num_samples = config['fid']['num_samples']
    D = 28 * 28
    
    # Collect real images
    print(f"Collecting {num_samples} real images...")
    real_images = []
    for batch in test_loader:
        real_images.append(batch)
        if sum(len(b) for b in real_images) >= num_samples:
            break
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    real_images = real_images.reshape(num_samples, 1, 28, 28).to(device)
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    device_str = device if isinstance(device, str) else str(device)
    with torch.no_grad():
        start = time.perf_counter()
        gen_images = model.sample((num_samples, D))
        if 'cuda' in device_str:
            torch.cuda.synchronize()
        end = time.perf_counter()
    elapsed = end - start
    s_per = num_samples / elapsed if elapsed > 0 else float('inf')
    print(f"[TIMING] DDPM sampling: {elapsed:.4f} s for {num_samples} samples ({s_per:.2f} samples/s)")
    gen_images = gen_images.reshape(num_samples, 1, 28, 28).to(device)
    
    # Compute FID
    print("Computing FID score...")
    classifier_ckpt = config['fid']['classifier_ckpt']
    fid_score = compute_fid(real_images, gen_images, device=device, classifier_ckpt=classifier_ckpt)
    
    print(f"\n[FID Score] {fid_score:.4f}")
    
    # Save results to JSON
    output_dir = config['paths']['fid_output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Create run results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "fid_score": float(fid_score),
        "num_samples": num_samples,
        "sampling_time_sec": float(elapsed),
        "samples_per_sec": float(s_per),
        "model": {
            "network": config['model']['network'],
            "T": config['model']['T'],
            "beta_1": config['model']['beta_1'],
            "beta_T": config['model']['beta_T']
        },
        "training": {
            "epochs": config['training']['epochs'],
            "batch_size": config['data']['batch_size'],
            "lr": config['training']['lr']
        },
        "model_path": model_path
    }
    
    # Save JSON
    json_path = os.path.join(output_dir, f"run_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"[OK] Results saved -> {json_path}")
    return fid_score


# ============================================================================
# Latent-Space DDPM Functions
# ============================================================================

def train_vae(vae: VAE, train_loader, epochs: int, device: str):
    """Train the β-VAE model."""
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = vae(batch)  # Forward returns -elbo (negative ELBO)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  VAE Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def build_latent_model(config: dict, device: str) -> DDPMLatent:
    """Build DDPMLatent model from config."""
    vae_cfg = config['vae']
    model_cfg = config['model']
    
    # Build VAE
    vae = build_vae(
        latent_dim=vae_cfg['latent_dim'],
        beta=vae_cfg['beta'],
        decoder_std=vae_cfg.get('decoder_std', 0.5),
        device=device
    )
    
    # Build DDPM network for latent space
    latent_dim = vae_cfg['latent_dim']
    if model_cfg['network'] == 'fc':
        network = FcNetwork(latent_dim, 64)
    else:
        raise ValueError(f"Network '{model_cfg['network']}' not supported for latent mode (use 'fc')")
    
    ddpm = DDPM(
        network=network,
        beta_1=model_cfg['beta_1'],
        beta_T=model_cfg['beta_T'],
        T=model_cfg['T']
    ).to(device)
    
    return DDPMLatent(vae, ddpm, freeze_encoder=True)


def sample_from_latent_model(model, num_samples: int, device: str):
    """Generate samples from latent DDPM with timing."""
    device_str = device if isinstance(device, str) else str(device)
    
    with torch.no_grad():
        start = time.perf_counter()
        samples = model.sample(num_samples).cpu()
        if 'cuda' in device_str:
            torch.cuda.synchronize()
        end = time.perf_counter()
    
    elapsed = end - start
    s_per = num_samples / elapsed if elapsed > 0 else float('inf')
    print(f"[TIMING] Latent DDPM sampling: {elapsed:.4f} s for {num_samples} samples ({s_per:.2f} samples/s)")
    
    return samples, elapsed, s_per


def run_train_latent(config: dict):
    """Train VAE then DDPM in latent space."""
    device = config['training']['device']
    
    print("=" * 60)
    print("DDPM in β-VAE Latent Space - Training")
    print("=" * 60)
    print(f"\n[VAE] beta={config['vae']['beta']}, latent_dim={config['vae']['latent_dim']}")
    print(f"[DDPM] T={config['model']['T']}, network={config['model']['network']}")
    print("=" * 60)
    
    # Create data loaders
    train_loader, _ = get_data_loaders_2d(config)
    
    # Build model
    model = build_latent_model(config, device)
    
    # Step 1: Train or load VAE
    vae_path = get_vae_path(config)
    vae_checkpoint = config['vae'].get('checkpoint')
    
    if vae_checkpoint and os.path.exists(vae_checkpoint):
        print(f"\n[VAE] Loading pre-trained VAE from {vae_checkpoint}")
        model.vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
    else:
        print(f"\n[VAE] Training β-VAE for {config['training']['vae_epochs']} epochs...")
        # Temporarily unfreeze VAE for training
        for param in model.vae.parameters():
            param.requires_grad = True
        model.vae.train()
        
        train_vae(model.vae, train_loader, config['training']['vae_epochs'], device)
        
        # Save VAE
        os.makedirs(os.path.dirname(vae_path) if os.path.dirname(vae_path) else '.', exist_ok=True)
        torch.save(model.vae.state_dict(), vae_path)
        print(f"[OK] VAE saved -> {vae_path}")
        
        # Re-freeze VAE for DDPM training
        for param in model.vae.encoder.parameters():
            param.requires_grad = False
        for param in model.vae.decoder.parameters():
            param.requires_grad = False
        model.vae.eval()
    
    # Inspect latent distribution from VAE
    print(f"\n[INSPECT] Analyzing VAE latent distribution...")
    with torch.no_grad():
        latent_stats = []
        for batch in train_loader:
            batch = batch.to(device)
            q = model.vae.encoder(batch)
            z = q.rsample()
            latent_stats.append(z.cpu())
            if len(latent_stats) >= 5:  # Just a few batches
                break
        
        all_z = torch.cat(latent_stats, dim=0)
        print(f"  VAE latent - min: {all_z.min():.4f}, max: {all_z.max():.4f}")
        print(f"  VAE latent - mean: {all_z.mean():.4f}, std: {all_z.std():.4f}")
        print(f"  VAE latent - mean per dim: {all_z.mean(dim=0).abs().max():.4f}")
        print(f"  VAE latent - std per dim: {all_z.std(dim=0).mean():.4f}")
    
    # Step 2: Train DDPM in latent space
    print(f"\n[DDPM] Training in latent space for {config['training']['ddpm_epochs']} epochs...")
    optimizer = torch.optim.Adam(model.ddpm.parameters(), lr=config['training']['lr'])
    
    for epoch in range(config['training']['ddpm_epochs']):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  DDPM Epoch {epoch+1}/{config['training']['ddpm_epochs']}, Loss: {avg_loss:.4f}")
    
    # Save DDPM
    model_path = get_model_path(config)
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    torch.save(model.ddpm.state_dict(), model_path)
    print(f"\n[OK] DDPM saved -> {model_path}")


def run_sample_latent(config: dict):
    """Generate samples from latent DDPM."""
    device = config['training']['device']
    
    # Build and load model
    model = build_latent_model(config, device)
    
    # Load VAE
    vae_path = get_vae_path(config)
    vae_checkpoint = config['vae'].get('checkpoint') or vae_path
    model.vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
    
    # Load DDPM
    model_path = get_model_path(config)
    model.ddpm.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    num_samples = config['sampling']['num_samples']
    
    print(f"Generating {num_samples} samples from latent DDPM...")
    samples, elapsed, s_per = sample_from_latent_model(model, num_samples, device)

    # Additionally measure pure VAE prior sampling (z ~ p(z), x ~ p(x|z)) for comparison
    try:
        with torch.no_grad():
            start_v = time.perf_counter()
            vae_samples = model.vae.sample(num_samples).to(device)  # ensure device
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            end_v = time.perf_counter()
        elapsed_v = end_v - start_v
        s_per_v = num_samples / elapsed_v if elapsed_v > 0 else float('inf')
        print(f"[TIMING] VAE prior sampling: {elapsed_v:.4f} s for {num_samples} samples ({s_per_v:.2f} samples/s)")
    except Exception:
        # If VAE sampling fails for any reason, skip timing but continue
        print("[TIMING] VAE prior sampling: failed to run/sample (skipped)")
    
    # Clamp and reshape for saving
    samples = samples.clamp(0, 1)
    samples = samples.unsqueeze(1)  # Add channel dim: (n, 1, 28, 28)
    
    # Save samples
    samples_dir = config['paths']['samples_dir']
    os.makedirs(samples_dir, exist_ok=True)
    
    grid_size = config['sampling']['grid_size']
    sample_path = os.path.join(samples_dir, f"ddpm_latent_z{config['vae']['latent_dim']}_samples.png")
    save_image(samples[:grid_size**2], sample_path, nrow=grid_size)
    print(f"[OK] Samples saved -> {sample_path}")


def run_fid_latent(config: dict):
    """Compute FID for latent DDPM."""
    device = config['training']['device']
    
    # Build and load model
    model = build_latent_model(config, device)
    
    # Load VAE
    vae_path = get_vae_path(config)
    vae_checkpoint = config['vae'].get('checkpoint') or vae_path
    model.vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
    
    # Load DDPM
    model_path = get_model_path(config)
    model.ddpm.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get real test data
    _, test_loader = get_data_loaders_2d(config)
    
    num_samples = config['fid']['num_samples']
    
    # Collect real images
    print(f"Collecting {num_samples} real images...")
    real_images = []
    for batch in test_loader:
        real_images.append(batch)
        if sum(len(b) for b in real_images) >= num_samples:
            break
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    real_images = real_images.unsqueeze(1).to(device)  # (n, 1, 28, 28)
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    gen_images, elapsed, s_per = sample_from_latent_model(model, num_samples, device)
    gen_images = gen_images.unsqueeze(1).to(device)  # (n, 1, 28, 28)
    
    # Compute FID
    print("Computing FID score...")
    classifier_ckpt = config['fid']['classifier_ckpt']
    fid_score = compute_fid(real_images, gen_images, device=device, classifier_ckpt=classifier_ckpt)
    
    print(f"\n[FID Score] {fid_score:.4f}")
    
    # Save results to JSON
    output_dir = config['paths']['fid_output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "fid_score": float(fid_score),
        "num_samples": num_samples,
        "sampling_time_sec": float(elapsed),
        "samples_per_sec": float(s_per),
        "mode": "latent",
        "vae": {
            "beta": config['vae']['beta'],
            "latent_dim": config['vae']['latent_dim'],
            "decoder": config['vae']['decoder']
        },
        "ddpm": {
            "network": config['model']['network'],
            "T": config['model']['T'],
            "beta_1": config['model']['beta_1'],
            "beta_T": config['model']['beta_T']
        },
        "training": {
            "vae_epochs": config['training']['vae_epochs'],
            "ddpm_epochs": config['training']['ddpm_epochs'],
            "lr": config['training']['lr']
        }
    }
    
    json_path = os.path.join(output_dir, f"run_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"[OK] Results saved -> {json_path}")
    return fid_score


def main():
    parser = argparse.ArgumentParser(description="DDPM Runner")
    parser.add_argument('mode', choices=['train', 'sample', 'fid', 'all'],
                        help='Mode: train, sample, fid, or all (train+sample+fid)')
    parser.add_argument('--config', type=str, default='../../configs/ddpm_unet.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Determine if we're running in latent space mode
    latent_mode = is_latent_mode(config)
    
    if latent_mode:
        print("[Mode] Latent-space DDPM (β-VAE)")
        if args.mode == 'train':
            run_train_latent(config)
        elif args.mode == 'sample':
            run_sample_latent(config)
        elif args.mode == 'fid':
            run_fid_latent(config)
        elif args.mode == 'all':
            run_train_latent(config)
            run_sample_latent(config)
            run_fid_latent(config)
    else:
        print("[Mode] Pixel-space DDPM")
        if args.mode == 'train':
            run_train(config)
        elif args.mode == 'sample':
            run_sample(config)
        elif args.mode == 'fid':
            run_fid(config)
        elif args.mode == 'all':
            run_train(config)
            run_sample(config)
            run_fid(config)


if __name__ == "__main__":
    main()
