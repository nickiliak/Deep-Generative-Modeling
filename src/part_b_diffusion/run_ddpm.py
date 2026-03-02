#!/usr/bin/env python3
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import datetime
import torch
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
import time

from part_b_diffusion.ddpm import DDPM, FcNetwork, train, build_model
from part_b_diffusion.ddpm_vae_beta import DDPMLatent, build_vae, train_vae, build_latent_model
from part_b_diffusion.utils import is_latent_mode, load_config, get_model_path, get_vae_path, preprocess_mnist
from part_b_diffusion.plot_latent_distributions import compare_latent_distributions
from part_a_vaes.vae_priors_temp import VAE
from shared.fid_utils import compute_fid
from shared.data_loader import get_mnist_raw




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
    """Generate samples from trained model (pixel-space or latent-space)."""
    device = config['training']['device']
    latent = is_latent_mode(config)
    num_samples = config['sampling']['num_samples']
    
    print(f"Generating {num_samples} samples...")
    
    if latent:
        # Build and load latent model
        model = build_latent_model(config, device)
        vae_path = get_vae_path(config)
        vae_checkpoint = config['vae'].get('checkpoint') or vae_path
        model.vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
        model_path = get_model_path(config)
        model.ddpm.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Sample from latent DDPM
        samples, elapsed, s_per = sample_from_latent_model(model, num_samples, device)
        
        # Additionally measure pure VAE prior sampling for comparison
        try:
            with torch.no_grad():
                start_v = time.perf_counter()
                vae_samples = model.vae.sample(num_samples).to(device)
                if 'cuda' in str(device):
                    torch.cuda.synchronize()
                end_v = time.perf_counter()
            elapsed_v = end_v - start_v
            s_per_v = num_samples / elapsed_v if elapsed_v > 0 else float('inf')
            print(f"[TIMING] VAE prior sampling: {elapsed_v:.4f} s for {num_samples} samples ({s_per_v:.2f} samples/s)")
        except Exception:
            print("[TIMING] VAE prior sampling: failed to run/sample (skipped)")
        
        # Post-process: clamp to [0, 1]
        samples = samples.clamp(0, 1)
        samples = samples.unsqueeze(1)  # Add channel dim: (n, 1, 28, 28)
    else:
        # Build and load pixel-space model
        model = build_model(config, device)
        model_path = get_model_path(config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Sample from pixel-space DDPM
        D = 28 * 28
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
        
        # Post-process: transform from [-1, 1] to [0, 1]
        samples = samples / 2 + 0.5
        samples = samples.clamp(0, 1)
        samples = samples.reshape(num_samples, 1, 28, 28)
    
    # Save samples
    samples_dir = config['paths']['samples_dir']
    os.makedirs(samples_dir, exist_ok=True)
    
    grid_size = config['sampling']['grid_size']
    images_per_grid = grid_size ** 2
    num_grids = num_samples // images_per_grid
    
    # Save each grid as a separate image
    for grid_idx in range(num_grids):
        start_idx = grid_idx * images_per_grid
        end_idx = start_idx + images_per_grid
        grid_samples = samples[start_idx:end_idx]
        
        if latent:
            sample_path = os.path.join(samples_dir, f"ddpm_latent_z{config['vae']['latent_dim']}_samples_{grid_idx+1}.png")
        else:
            sample_path = os.path.join(samples_dir, f"ddpm_{config['model']['network']}_samples_{grid_idx+1}.png")
        
        save_image(grid_samples, sample_path, nrow=grid_size)
        print(f"[OK] Samples saved -> {sample_path}")


def run_fid(config: dict):
    """Compute FID score for both pixel-space and latent-space DDPM."""
    device = config['training']['device']
    latent = is_latent_mode(config)
    
    # Build and load model
    if latent:
        model = build_latent_model(config, device)
        vae_path = get_vae_path(config)
        vae_checkpoint = config['vae'].get('checkpoint') or vae_path
        model.vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
        test_loader = get_data_loaders_2d(config)[1]
    else:
        model = build_model(config, device)
        test_loader = get_data_loaders(config)[1]
    
    model_path = get_model_path(config)
    if latent:
        model.ddpm.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    num_samples = config['fid']['num_samples']
    
    # Collect real images
    print(f"Collecting {num_samples} real images...")
    real_images = []
    for batch in test_loader:
        real_images.append(batch)
        if sum(len(b) for b in real_images) >= num_samples:
            break
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # Reshape real images
    if latent:
        real_images = real_images.unsqueeze(1).to(device)  # (n, 1, 28, 28)
    else:
        real_images = real_images.reshape(num_samples, 1, 28, 28).to(device)
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    if latent:
        gen_images, elapsed, s_per = sample_from_latent_model(model, num_samples, device)
        gen_images = gen_images.unsqueeze(1).to(device)  # (n, 1, 28, 28)
    else:
        D = 28 * 28
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
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if latent:
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
    else:
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
    
    json_path = os.path.join(output_dir, f"run_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"[OK] Results saved -> {json_path}")
    return fid_score


# ============================================================================
# Latent-Space DDPM Functions
# ============================================================================

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
    
    # Step 2: Train DDPM in latent space
    print(f"\n[DDPM] Training in latent space for {config['training']['ddpm_epochs']} epochs...")
    model.train_ddpm(train_loader, config['training']['ddpm_epochs'], device)
    
    # Save DDPM
    model_path = get_model_path(config)
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    torch.save(model.ddpm.state_dict(), model_path)
    print(f"\n[OK] DDPM saved -> {model_path}")


def run_plot_latents(config: dict):
    """Plot and compare latent distributions (prior, posterior, DDPM) for latent-space models."""
    device = config['training']['device']
    latent = is_latent_mode(config)
    
    if not latent:
        print("[ERROR] plot_latents requires a latent-space config (with VAE section)")
        return
    
    print("Plotting latent distributions...")
    
    # Create data loader for computing aggregate posterior
    _, test_loader = get_data_loaders_2d(config)
    
    # Build and load model
    model = build_latent_model(config, device)
    
    # Load VAE checkpoint
    vae_path = get_vae_path(config)
    vae_checkpoint = config['vae'].get('checkpoint') or vae_path
    
    if not os.path.exists(vae_checkpoint):
        print(f"[ERROR] VAE checkpoint not found at {vae_checkpoint}")
        print(f"        Expected pattern: models/beta_vae_gaussian_beta{{beta}}_z{{latent_dim}}.pt")
        print(f"        You may need to train the VAE first using: uv run src/part_b_diffusion/run_ddpm.py train --config {config}")
        return
    
    model.vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
    print(f"[OK] Loaded VAE from {vae_checkpoint}")
    
    # Load DDPM checkpoint
    model_path = get_model_path(config)
    if not os.path.exists(model_path):
        print(f"[ERROR] DDPM checkpoint not found at {model_path}")
        print(f"        Expected pattern: models/ddpm_latent_T{{T}}_z{{latent_dim}}.pt")
        print(f"        You may need to train the DDPM first using: uv run src/part_b_diffusion/run_ddpm.py train --config {config}")
        return
    
    model.ddpm.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[OK] Loaded DDPM from {model_path}")
    model.eval()
    
    # Create output directory for latent distribution comparisons
    output_dir = 'reports/outputs/latent_distributions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and compare distributions
    compare_latent_distributions(
        model,
        test_loader,
        device=device,
        n_samples=config.get('plot_latents', {}).get('n_samples', 5000),
        save_dir=output_dir
    )
    print(f"[OK] Latent distribution plots saved to {samples_dir}")


def main():
    parser = argparse.ArgumentParser(description="DDPM Runner")
    parser.add_argument('mode', choices=['train', 'sample', 'fid', 'all', 'plot_latents'],
                        help='Mode: train, sample, fid, all, or plot_latents')
    parser.add_argument('--config', type=str, default='../../configs/ddpm_unet.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    latent_mode = is_latent_mode(config)
    
    mode_name = "Latent-space DDPM (β-VAE)" if latent_mode else "Pixel-space DDPM"
    print(f"[Mode] {mode_name}")
    
    # Map mode -> functions to run
    mode_map = {
        'train': [run_train_latent if latent_mode else run_train],
        'sample': [run_sample],
        'fid': [run_fid],
        'plot_latents': [run_plot_latents],
        'all': [
            run_train_latent if latent_mode else run_train,
            run_sample,
            run_fid
        ]
    }
    
    for func in mode_map[args.mode]:
        func(config)


if __name__ == "__main__":
    main()
