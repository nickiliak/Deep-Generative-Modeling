import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time
import json
import datetime
from tqdm import tqdm

# For FID computation
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.fid_utils import compute_fid


# Import modules (support both direct script execution and module import)
try:
    from .priors import GaussianPrior, MoGPrior, FlowPrior
    from .encoders import Encoder
    from .decoders import BernoulliDecoder, GaussianDecoder
    from .vae import VAE
except ImportError:
    from priors import GaussianPrior, MoGPrior, FlowPrior
    from encoders import Encoder
    from decoders import BernoulliDecoder, GaussianDecoder
    from vae import VAE


# Constants

PRIOR_LABELS = {
    "gaussian": "Gaussian",
    "mog":      "Mixture of Gaussians",
    "flow":     "Flow-based",
}


# Build model helpers

def build_encoder_net(M: int) -> nn.Module:
    """Build encoder network for MNIST (28x28 -> 2M latent dims)."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, M * 2),
    )


def build_decoder_net(M: int) -> nn.Module:
    """Build decoder network for MNIST (M latent dims -> 28x28)."""
    return nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )


def build_model(prior_name, M, device, flow_layers=4, flow_hidden=64, decoder_type='bernoulli', beta=1.0, decoder_std=0.5):
    """
    Build a VAE with specified prior and decoder type.
    
    Parameters:
    prior_name: 'gaussian', 'mog', or 'flow'
    M: Latent dimension
    device: Device to place model on
    flow_layers: Number of coupling layers for FlowPrior (default: 4)
    flow_hidden: Hidden dimension for FlowPrior (default: 64)
    decoder_type: 'bernoulli' for binary MNIST or 'gaussian' for continuous (default: 'bernoulli')
    beta: KL weight for β-VAE (default: 1.0)
    decoder_std: Standard deviation for Gaussian decoder (default: 0.5)
    """
    encoder_net = build_encoder_net(M)
    decoder_net = build_decoder_net(M)
    
    # Build prior
    if prior_name == "gaussian":
        prior = GaussianPrior(M)
    elif prior_name == "mog":
        prior = MoGPrior(M, K=10)
    elif prior_name == "flow":
        prior = FlowPrior(M, num_layers=flow_layers, hidden_dim=flow_hidden)
    else:
        raise ValueError(f"Unknown prior: {prior_name}")
    
    # Build decoder
    if decoder_type == 'bernoulli':
        decoder = BernoulliDecoder(decoder_net)
    elif decoder_type == 'gaussian':
        decoder = GaussianDecoder(decoder_net, std=decoder_std, learnable_std=False)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    encoder = Encoder(encoder_net)
    return VAE(prior, decoder, encoder, beta=beta).to(device)


def _sample_prior(model, N):
    """Helper: sample N points from any prior type."""
    with torch.no_grad():
        if isinstance(model.prior, GaussianPrior):
            return model.prior().sample(torch.Size([N])).cpu().numpy()
        else:
            return model.prior.sample(N).cpu().numpy()


# Training

def train(model, optimizer, data_loader, epochs, device):
    model.train()
    total_steps = len(data_loader) * epochs
    bar = tqdm(range(total_steps), desc="Training")
    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch+1}/{epochs}")
            bar.update()


# Evaluation: Test-set ELBO

def evaluate_elbo(model, data_loader, device):
    """Return mean ELBO over the full test loader."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in data_loader:
            total += model.elbo(x.to(device)).item()
            n     += 1
    return total / n


# Multi-run evaluation (mean +/- std)

def multirun_evaluation(prior_name, M, train_loader, test_loader, epochs, lr, device, n_runs,
                        flow_layers=4, flow_hidden=64):
    """
    Trains the model from scratch n_runs times and reports mean +/- std of
    the test-set ELBO across runs
    """
    elbos = []
    for run in range(1, n_runs + 1):
        print(f"\n-- Run {run}/{n_runs} --")
        model     = build_model(prior_name, M, device, flow_layers, flow_hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train(model, optimizer, train_loader, epochs, device)
        elbo = evaluate_elbo(model, test_loader, device)
        elbos.append(elbo)
        print(f"  Run {run} test ELBO: {elbo:.4f}")

    elbos      = np.array(elbos)
    mean, std  = elbos.mean(), elbos.std()
    label      = PRIOR_LABELS[prior_name]

    print(f"\n{'─'*50}")
    print(f"[Multi-run results -- {label} prior]")
    print(f"  Runs:    {n_runs}")
    print(f"  ELBOs:   {np.round(elbos, 4)}")
    print(f"  Mean:    {mean:.4f}")
    print(f"  Std:     {std:.4f}")
    print(f"  Summary: {mean:.4f} +/- {std:.4f}")
    print(f"{'─'*50}")
    return mean, std, elbos


# Plotting: prior vs aggregate posterior

def plot_prior_vs_posterior(model, data_loader, device, prior_name, save_path, n_batches=50):
    """
    Two-panel plot comparing prior and aggregate posterior.

    Following the project spec exactly:
      - Posterior points are SAMPLES from q(z|x)  [not the mean]
      - If M > 2: PCA projects samples onto the first 2 principal components
      - If M == 2: use the dimensions directly
      - Prior is also projected through the same PCA fit (if M > 2)
      - Points coloured by digit class (tab10)
      - KDE shown as smooth background density in each panel

    Left panel:  prior p(z) density  (KDE of prior samples, same PCA if M>2)
    Right panel: aggregate posterior q(z|X) density + class-coloured scatter
    """
    from scipy.stats import gaussian_kde
    from sklearn.decomposition import PCA

    model.eval()
    M = model.prior.M if hasattr(model.prior, 'M') else model.prior.base_mean.shape[0]

    # Collect SAMPLES from q(z|x) for each test point + labels
    posterior_z_list, label_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= n_batches:
                break
            q = model.encoder(x.to(device))
            z = q.rsample()           # sample from q(z|x), not the mean
            posterior_z_list.append(z.cpu())
            label_list.append(y)

    posterior_z = torch.cat(posterior_z_list, dim=0).numpy()   # (N, M)
    labels      = torch.cat(label_list,       dim=0).numpy()   # (N,)

    # Sample same number of points from prior
    prior_z = _sample_prior(model, posterior_z.shape[0])       # (N, M)

    # PCA projection when M > 2
    if M > 2:
        pca = PCA(n_components=2)
        # Fit PCA on posterior samples (captures most variance of the data)
        posterior_2d = pca.fit_transform(posterior_z)
        prior_2d     = pca.transform(prior_z)
        xlabel = "PC 1"
        ylabel = "PC 2"
        subtitle = f"(PCA projection from {M}D)"
    else:
        posterior_2d = posterior_z
        prior_2d     = prior_z
        xlabel = "z₁"
        ylabel = "z₂"
        subtitle = ""

    # Shared axis limits (union of both)
    all_2d = np.concatenate([prior_2d, posterior_2d], axis=0)
    pad    = 1.0
    xlim   = (all_2d[:, 0].min() - pad, all_2d[:, 0].max() + pad)
    ylim   = (all_2d[:, 1].min() - pad, all_2d[:, 1].max() + pad)

    # KDE on the 2D projections
    res      = 200
    xs       = np.linspace(xlim[0], xlim[1], res)
    ys       = np.linspace(ylim[0], ylim[1], res)
    gx, gy   = np.meshgrid(xs, ys)
    grid_pts = np.vstack([gx.ravel(), gy.ravel()])

    density_prior = gaussian_kde(prior_2d.T)(grid_pts).reshape(res, res)
    density_post  = gaussian_kde(posterior_2d.T)(grid_pts).reshape(res, res)

    # Plot
    prior_label = PRIOR_LABELS[prior_name]
    tab10       = plt.cm.get_cmap("tab10", 10)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    title = f"{prior_label} Prior: Prior vs Aggregate Posterior"
    if subtitle:
        title += f"  {subtitle}"
    fig.suptitle(title, fontsize=13)

    # Left: prior KDE
    ax = axes[0]
    cp = ax.contourf(gx, gy, density_prior, levels=30, cmap="Blues")
    ax.contour(gx, gy, density_prior, levels=6,  colors="white",
               linewidths=0.6, alpha=0.6)
    cbar = fig.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Density', rotation=270, labelpad=15)
    ax.set_title("Prior p(z)")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(xlim);     ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    # Right: posterior KDE + class-coloured scatter
    ax = axes[1]
    ax.contourf(gx, gy, density_post, levels=30, cmap="Reds", alpha=0.55)

    for digit in range(10):
        mask = labels == digit
        if mask.sum() == 0:
            continue
        ax.scatter(
            posterior_2d[mask, 0], posterior_2d[mask, 1],
            s=5, alpha=0.5, color=tab10(digit),
            label=str(digit), linewidths=0,
        )

    ax.set_title("Aggregate posterior q(z|x)")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xlim(xlim);     ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.legend(title="digit", fontsize=7, title_fontsize=8,
              markerscale=2.5, loc="upper right",
              framealpha=0.75, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[OK] Plot saved -> {save_path}")
    plt.show()


def plot_samples(model, device, prior_name, save_path, n_samples=64):
    """
    Draw n_samples from p(z) -> p(x|z) and display as a grid.
    Uses the decoder mean (soft probabilities) for cleaner visuals.
    Grid dimensions adjust based on n_samples, preferring square layouts.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model.prior, GaussianPrior):
            z = model.prior().sample(torch.Size([n_samples])).to(device)
        else:
            z = model.prior.sample(n_samples).to(device)
        samples = model.decoder(z).mean.cpu()   # (N, 28, 28)

    # Prefer square grids; max 8 columns
    import math
    sqrt_samples = int(math.sqrt(n_samples))
    if sqrt_samples * sqrt_samples == n_samples:
        # Perfect square: use square grid
        ncols = sqrt_samples
        nrows = sqrt_samples
    else:
        # Not a square: use max 8 columns
        ncols = min(8, n_samples)
        nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
    fig.suptitle(f"{PRIOR_LABELS[prior_name]} Prior -- Generated Samples", fontsize=12)

    # Flatten axes for iteration, handle case when there's only 1 subplot
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = axes.flat
    else:
        axes_flat = axes.flat
    
    for idx, ax in enumerate(axes_flat):
        if idx < n_samples:
            ax.imshow(samples[idx], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[OK] Samples saved -> {save_path}")
    plt.show()


# Main

if __name__ == "__main__":
    from torchvision import datasets, transforms

    parser = argparse.ArgumentParser(description="VAE -- Part A (Gaussian + MoG priors)")
    parser.add_argument("mode", choices=["train", "evaluate", "sample", "multirun", "fid"],
                        help="train: single run | evaluate: load & eval | sample: generate images | multirun: N runs -> mean+/-std | fid: compute FID score")
    parser.add_argument("--prior",      type=str,   default="gaussian", choices=["gaussian", "mog", "flow"])
    parser.add_argument("--model",      type=str,   default=None,
                        help="Path to save/load model weights (default: models/{prior}_z{latent_dim}.pt)")
    parser.add_argument("--plot",       type=str,   default=None,
                        help="Output path for prior/posterior plot (default: figures/{prior}_prior_posterior.png)")
    parser.add_argument("--samples",    type=str,   default=None,
                        help="Output path for samples image (default: figures/{prior}_samples.png)")
    parser.add_argument("--device",     type=str,   default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--latent-dim", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--runs",       type=int,   default=5,
                        help="Number of independent training runs for multirun mode (default: 5)")
    parser.add_argument("--flow-layers", type=int,  default=4,
                        help="Number of coupling layers in the Flow prior (default: 4)")
    parser.add_argument("--flow-hidden", type=int,  default=64,
                        help="Hidden dim of scale/translation nets in Flow prior (default: 64)")
    parser.add_argument("--num-samples", type=int,  default=64,
                        help="Number of samples to generate in sample mode (default: 64)")
    parser.add_argument("--fid-samples", type=int, default=10000,
                        help="Number of samples for FID computation (default: 10000)")
    parser.add_argument("--classifier-ckpt", type=str, default="../../models/mnist_classifier.pth",
                        help="Path to classifier checkpoint for FID computation")
    args = parser.parse_args()

    # Set default paths relative to reports/outputs/part_a_vaes
    if args.model is None:
        args.model = f"../../reports/outputs/part_a_vaes/models/{args.prior}_z{args.latent_dim}.pt"
    if args.plot is None:
        args.plot = f"../../reports/outputs/part_a_vaes/figures/{args.prior}_prior_posterior.png"
    if args.samples is None:
        args.samples = f"../../reports/outputs/part_a_vaes/figures/{args.prior}_samples.png"

    print("# Options")
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v}")

    device = torch.device(args.device)
    M      = args.latent_dim

    # Get the directory of this script and resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Helper to resolve paths: use CWD if exists there, else script-relative for defaults
    def resolve_path(path):
        if os.path.isabs(path):
            return path
        # If path exists relative to CWD, use that
        if os.path.exists(path) or not path.startswith("../../"):
            return os.path.abspath(path)
        # Otherwise resolve relative to script dir (for default paths)
        return os.path.join(script_dir, path)
    
    args.model = resolve_path(args.model)
    args.plot = resolve_path(args.plot)
    args.samples = resolve_path(args.samples)

    # Create necessary directories
    for path in [args.model, args.plot, args.samples]:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    # Binarized MNIST
    binarize  = transforms.Lambda(lambda x: (x > 0.5).float().squeeze())
    transform = transforms.Compose([transforms.ToTensor(), binarize])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/", train=True,  download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/", train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=False,
    )

    # Single training run
    if args.mode == "train":
        model     = build_model(args.prior, M, device, args.flow_layers, args.flow_hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, args.epochs, device)
        torch.save(model.state_dict(), args.model)
        print(f"[OK] Model saved -> {args.model}")

        elbo = evaluate_elbo(model, test_loader, device)
        print(f"\n[Test ELBO] {elbo:.4f}")

        plot_prior_vs_posterior(model, test_loader, device, args.prior, args.plot)
        plot_samples(model, device, args.prior, args.samples, n_samples=args.num_samples)

    # Sample from a saved model 
    elif args.mode == "sample":
        from torchvision.utils import save_image
        import math
        model = build_model(args.prior, M, device, args.flow_layers, args.flow_hidden)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        
        # Measure sampling time
        n_samples = args.num_samples
        start_time = time.perf_counter()
        with torch.no_grad():
            samples = model.sample(n_samples).cpu()
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        samples_per_sec = n_samples / elapsed
        
        # Calculate grid dimensions: prefer square layouts
        sqrt_samples = int(math.sqrt(n_samples))
        if sqrt_samples * sqrt_samples == n_samples:
            nrow = sqrt_samples
        else:
            nrow = min(8, n_samples)
        
        save_image(samples.view(n_samples, 1, 28, 28), args.samples, nrow=nrow)
        print(f"[OK] {n_samples} samples saved -> {args.samples}")
        print(f"[Timing] {elapsed:.4f}s total, {samples_per_sec:.2f} samples/sec")

    # Evaluate a saved model
    elif args.mode == "evaluate":
        model = build_model(args.prior, M, device, args.flow_layers, args.flow_hidden)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        elbo = evaluate_elbo(model, test_loader, device)
        print(f"\n[Test ELBO] {elbo:.4f}")

        plot_prior_vs_posterior(model, test_loader, device, args.prior, args.plot)
        plot_samples(model, device, args.prior, args.samples, n_samples=args.num_samples)

    # Multi-run: train N times, report mean +/- std
    elif args.mode == "multirun":
        multirun_evaluation(
            prior_name   = args.prior,
            M            = M,
            train_loader = train_loader,
            test_loader  = test_loader,
            epochs       = args.epochs,
            lr           = args.lr,
            device       = device,
            n_runs       = args.runs,
            flow_layers  = args.flow_layers,
            flow_hidden  = args.flow_hidden,
        )
    
    # FID computation
    elif args.mode == "fid":
        model = build_model(args.prior, M, device, args.flow_layers, args.flow_hidden)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        
        # Resolve classifier checkpoint path
        classifier_ckpt = args.classifier_ckpt
        if not os.path.isabs(classifier_ckpt):
            classifier_ckpt = os.path.join(script_dir, classifier_ckpt)
        
        # Collect real images from test set
        print(f"[FID] Collecting {args.fid_samples} real images from test set...")
        real_images = []
        for x, _ in test_loader:
            real_images.append(x)
            if sum(img.size(0) for img in real_images) >= args.fid_samples:
                break
        real_images = torch.cat(real_images, dim=0)[:args.fid_samples]
        # Reshape and normalize to [-1, 1] for FID computation
        real_images = real_images.view(-1, 1, 28, 28).to(device)
        real_images = real_images * 2 - 1  # [0,1] -> [-1,1]
        
        # Generate samples with timing
        print(f"[FID] Generating {args.fid_samples} samples...")
        start_time = time.perf_counter()
        with torch.no_grad():
            gen_images = model.sample(args.fid_samples)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        samples_per_sec = args.fid_samples / elapsed
        print(f"[Timing] {elapsed:.4f}s total, {samples_per_sec:.2f} samples/sec")
        
        # Reshape and normalize generated samples
        gen_images = gen_images.view(-1, 1, 28, 28).to(device)
        gen_images = gen_images * 2 - 1  # [0,1] -> [-1,1]
        
        # Compute FID
        print(f"[FID] Computing FID score...")
        fid_score = compute_fid(real_images, gen_images, device=str(device), classifier_ckpt=classifier_ckpt)
        print(f"\n[FID Score] {fid_score:.4f}")
        
        # Save results to JSON
        output_dir = os.path.join(script_dir, "../../reports/outputs/part_a_vaes/fid")
        output_dir = resolve_path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result = {
            "timestamp": timestamp,
            "fid_score": float(fid_score),
            "num_samples": args.fid_samples,
            "sampling_time_sec": float(elapsed),
            "samples_per_sec": float(samples_per_sec),
            "prior": args.prior,
            "vae": {
                "latent_dim": args.latent_dim,
                "decoder": "bernoulli",
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
            "model_path": args.model
        }
        
        json_path = os.path.join(output_dir, f"run_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"[OK] Results saved -> {json_path}")
