# Part A: Variational Autoencoders (VAEs)

All commands should be run from the **repo root** (`Deep-Generative-Modeling/`).

## Overview

This module trains VAEs with different prior distributions:
- **Gaussian**: Standard isotropic Gaussian prior N(0, I)
- **Mixture of Gaussians (MoG)**: K=10 mixture components with learnable selection
- **Normalising Flow**: Real-NVP style affine coupling layers

All models use a Bernoulli decoder for binary MNIST.

---

## Training

### Single Training Run

Train a VAE with a specified prior:

```bash
# Gaussian prior
uv run src/part_a_vaes/vae_priors.py train --prior gaussian --latent-dim 32 --epochs 20

# Mixture of Gaussians prior
uv run src/part_a_vaes/vae_priors.py train --prior mog --latent-dim 32 --epochs 20

# Flow-based prior
uv run src/part_a_vaes/vae_priors.py train --prior flow --latent-dim 32 --epochs 20 --flow-layers 4 --flow-hidden 64
```

**Output:**
- Model checkpoint saved to `reports/outputs/part_a_vaes/models/{prior}_z{latent_dim}.pt`
- Prior vs. aggregate posterior plot saved to `reports/outputs/part_a_vaes/figures/{prior}_prior_posterior.png`
- Generated sample images saved to `reports/outputs/part_a_vaes/figures/{prior}_samples.png`
- Test set ELBO printed to console

### Multi-Run Evaluation (Mean ± Std)

Train the same configuration N times and report mean ± std of test ELBO:

```bash
# Train 5 times with Gaussian prior
uv run src/part_a_vaes/vae_priors.py multirun --prior gaussian --latent-dim 32 --epochs 20 --runs 5

# Train 5 times with Flow prior (custom layers)
uv run src/part_a_vaes/vae_priors.py multirun --prior flow --latent-dim 32 --epochs 20 --runs 5 --flow-layers 6 --flow-hidden 128
```

Reports mean and standard deviation of test ELBO across runs.

---

## Evaluation & Visualization

### Evaluate a Saved Model

Load and evaluate an existing checkpoint:

```bash
uv run src/part_a_vaes/vae_priors.py evaluate \
  --prior gaussian \
  --latent-dim 32 \
  --model reports/outputs/part_a_vaes/models/gaussian_z32.pt
```

Produces:
- Test set ELBO
- Prior vs. aggregate posterior visualization
- Sample images

### Generate Samples Only

Sample from a trained VAE without evaluation:

```bash
uv run src/part_a_vaes/vae_priors.py sample \
  --prior gaussian \
  --latent-dim 32 \
  --model reports/outputs/part_a_vaes/models/gaussian_z32.pt
```

---

## Command-Line Arguments

### Required
- `mode`: One of `train`, `evaluate`, `sample`, `multirun`

### Optional
| Argument | Default | Description |
|----------|---------|-------------|
| `--prior` | `gaussian` | Prior type: `gaussian`, `mog`, `flow` |
| `--model` | Auto-generated | Path to save/load model checkpoint |
| `--plot` | Auto-generated | Path to save prior/posterior plot |
| `--samples` | Auto-generated | Path to save sample images |
| `--device` | `cpu` | Device: `cpu`, `cuda`, `mps` |
| `--batch-size` | `128` | Training batch size |
| `--epochs` | `20` | Training epochs |
| `--latent-dim` | `32` | Latent space dimension (z_dim) |
| `--lr` | `0.001` | Learning rate |
| `--runs` | `5` | Number of runs for multirun mode |
| `--flow-layers` | `4` | Number of coupling layers (Flow prior only) |
| `--flow-hidden` | `64` | Hidden dimension of scale/translation nets (Flow prior only) |

### Default Paths

If not specified, files are saved relative to `reports/outputs/part_a_vaes/`:

```
--model:    models/{prior}_z{latent_dim}.pt
--plot:     figures/{prior}_prior_posterior.png
--samples:  figures/{prior}_samples.png
```

---

## Output Locations

| Output | Default Path |
|--------|------|
| Model checkpoint | `reports/outputs/part_a_vaes/models/{prior}_z{latent_dim}.pt` |
| Prior vs. posterior plot | `reports/outputs/part_a_vaes/figures/{prior}_prior_posterior.png` |
| Generated samples | `reports/outputs/part_a_vaes/figures/{prior}_samples.png` |

All directories are created automatically if they don't exist.

---

## Examples

### Example 1: Quick Start (Train + Eval)
```bash
# Train a Gaussian VAE and auto-generate all outputs
uv run src/part_a_vaes/vae_priors.py train --prior gaussian --epochs 5 --latent-dim 16 --device cuda
```

### Example 2: Custom Paths
```bash
# Train with custom output locations
uv run src/part_a_vaes/vae_priors.py train \
  --prior mog \
  --model /tmp/my_vae.pt \
  --plot my_prior_post.png \
  --samples my_samples.png \
  --epochs 30
```

### Example 3: Flow Prior with More Layers
```bash
# Train Flow prior with 6 coupling layers
uv run src/part_a_vaes/vae_priors.py train \
  --prior flow \
  --flow-layers 6 \
  --flow-hidden 128 \
  --epochs 20
```

### Example 4: Multi-Run Comparison
```bash
# Compare three priors across 3 independent runs each
for prior in gaussian mog flow; do
  uv run src/part_a_vaes/vae_priors.py multirun --prior $prior --runs 3 --epochs 15
done
```

---

## Key Concepts

### Prior Distributions

**Gaussian**: Simple isotropic N(0, I). KL divergence computed in closed form.

**Mixture of Gaussians (MoG)**: K=10 fixed components with random initial means. KL estimated via Monte Carlo.

**Normalising Flow**: Real-NVP style with alternating coupling layers. Learned parameters. KL estimated via Monte Carlo using inverse flow + base density.

### Visualizations

**Prior vs. Aggregate Posterior Plot:**
- Left panel: Prior p(z) density (KDE heatmap)
- Right panel: Aggregate posterior q(z|X) with class-colored scatter
- If M > 2: PCA projection to 2D (captures 99%+ variance)
- Shared axis limits and contour levels for comparison

**Generated Samples:**
- 8x8 grid of images sampled from p(z) → decoder
- Uses decoder mean (soft probabilities) for clarity

### ELBO Details

ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

- **Gaussian prior**: Closed-form KL (stable, exact)
- **MoG/Flow priors**: Monte Carlo KL estimation (requires reparameterization)

---

## Notes

- **Binarized MNIST**: Data is binarized (pixel values > 0.5 → 1, else 0)
- **Encoder output**: 2M dims (M for mean, M for log-std)
- **Training**: Uses ELBO loss with Adam optimizer
- **Evaluation**: Test set ELBO reported (higher is better, less negative)
