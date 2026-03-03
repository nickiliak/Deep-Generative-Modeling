Deep Generative Modeling - MNIST
==============================

A team project implementing Variational Autoencoders (VAE) and Diffusion Probabilistic Models (DDPM) for MNIST generation.

**Project split into two sub-teams:**
- **Part A**: VAE with flexible priors (Gaussian, MoG, Normalizing Flows) - Best performer: Flow-based prior (ELBO: -72.78)
- **Part B**: Diffusion models (pixel-space UNet DDPM, latent DDPM with β-VAE) - Best performer: Latent DDPM at β=10⁻³ (FID: 4.67)

## Key Results

| Method | Metric | Value |
|--------|--------|-------|
| Flow-based VAE | Test ELBO | -72.78 ± 0.12 |
| Latent DDPM (β=10⁻³) | FID Score | 4.67 ± 0.01 |
| UNet DDPM (pixel-space) | FID Score | 10.17 ± 0.05 |
| Latent DDPM | Sampling Speed | 2431 samples/sec |
| Flow VAE | Sampling Speed | 20638 samples/sec |

## Project Organization

```
├── pyproject.toml                 <- Project configuration & dependencies (uv manager)
├── README.md                      <- This file
├── data/                          <- MNIST data (auto-downloaded on first run)
│
├── models/                        <- Trained models & checkpoints
│   └── mnist_classifier.pth       <- Pre-trained classifier for FID scoring
│
├── notebooks/                     <- Jupyter notebooks for exploration & visualization
│   ├── part_a_*.ipynb             <- Team A analysis
│   └── part_b_*.ipynb             <- Team B analysis
│
└── src/                           <- Source code (split by team)
    ├── __init__.py
    │
    ├── shared/                    <- Utilities used by both teams
    │   ├── data_loader.py         <- get_mnist() & get_binarized_mnist()
    │   ├── fid_utils.py           <- compute_fid() & Classifier network
    │   └── __init__.py
    │
    ├── part_a_vaes/               <- Team 1: VAE experiments
    │   └── __init__.py
    │
    └── part_b_diffusion/          <- Team 2: Diffusion experiments
        └── __init__.py
```

## Setup

See [reports/report.tex](reports/report.tex) for the full project report with architecture details, visualizations, and analysis.

Install dependencies using `uv`:

```bash
uv sync
```

## Data

**MNIST loading functions:**

- `get_mnist(split="train")` - Standard MNIST with normalization (for Part B)
  - Shape: (1, 28, 28)
  - Normalized to mean=0.1307, std=0.3081
  
- `get_binarized_mnist(split="train")` - Binary MNIST with 0.5 threshold (for Part A)
  - Shape: (1, 28, 28)
  - Values: 0 or 1 (thresholded at 0.5)

Data automatically downloads to `data/` on first use from torchvision.

```python
from src.shared import get_mnist, get_binarized_mnist

# Team A: binarized data
train_a = get_binarized_mnist(split="train")
test_a = get_binarized_mnist(split="test")

# Team B: standard data
train_b = get_mnist(split="train")
test_b = get_mnist(split="test")
```

## Requirements

All dependencies managed via `pyproject.toml` and `uv`:

- `torch`, `torchvision` - Deep learning & datasets
- `scipy` - FID calculations
- `numpy` - Numerical computations
- Other optional packages for notebooks & visualization