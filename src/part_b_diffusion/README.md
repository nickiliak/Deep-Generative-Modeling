# Part B: DDPM Commands

All commands should be run from the **repo root** (`Deep-Generative-Modeling/`).

## Pixel-Space DDPM (U-Net)

### Train
```bash
uv run src/part_b_diffusion/run_ddpm.py train --config configs/ddpm_unet.yaml
```

### Sample
```bash
uv run src/part_b_diffusion/run_ddpm.py sample --config configs/ddpm_unet.yaml
```

### FID Evaluation
```bash
uv run src/part_b_diffusion/run_ddpm.py fid --config configs/ddpm_unet.yaml
```

### Full Pipeline (Train + Sample + FID)
```bash
uv run src/part_b_diffusion/run_ddpm.py all --config configs/ddpm_unet.yaml
```

### Output Locations
| Output | Path |
|--------|------|
| Model checkpoint | `models/part_b_diffusion/ddpm_unet_T{T}_epochs{epochs}.pt` |
| Sample images | `reports/outputs/part_b_diffusion/samples/ddpm_unet_samples_*.png` |
| FID results | `reports/outputs/part_b_diffusion/fid/run_<timestamp>.json` |

---

## Latent-Space DDPM (β-VAE)

Train DDPM on latent representations from a β-VAE with Gaussian decoder.

### Train
```bash
uv run src/part_b_diffusion/run_ddpm.py train --config configs/ddpm_latent.yaml
```

### Sample
```bash
uv run src/part_b_diffusion/run_ddpm.py sample --config configs/ddpm_latent.yaml
```

### FID Evaluation
```bash
uv run src/part_b_diffusion/run_ddpm.py fid --config configs/ddpm_latent.yaml
```

### Full Pipeline (Train + Sample + FID)
```bash
uv run src/part_b_diffusion/run_ddpm.py all --config configs/ddpm_latent.yaml
```

### Plot Latent Distributions
Compares three distributions in latent space:
- **Prior**: p(z) from β-VAE (standard normal)
- **Aggregate posterior**: q(z|x) from encoding real MNIST data
- **Learned DDPM**: p_ddpm(z) from sampling the trained latent DDPM

```bash
uv run src/part_b_diffusion/run_ddpm.py plot_latents --config configs/ddpm_latent_beta1e-6.yaml
```

Replace `ddpm_latent_beta1e-6.yaml` with any other `ddpm_latent_*.yaml` config to compare distributions for different β values.
Note: The config file must have corresponding trained VAE and DDPM checkpoints.

Generates 2D PCA projections with KDE density heatmaps, scatter points, means, and 2-sigma covariance ellipses for visual comparison.

### Output Locations
| Output | Path |
|--------|------|
| VAE checkpoint | `models/part_b_diffusion/beta_vae_gaussian_beta{beta}_z{latent_dim}.pt` |
| DDPM checkpoint | `models/part_b_diffusion/ddpm_latent_T{T}_z{latent_dim}.pt` |
| Sample images | `reports/outputs/part_b_diffusion/samples/ddpm_latent_z{latent_dim}_samples_*.png` |
| FID results | `reports/outputs/part_b_diffusion/fid/run_<timestamp>.json` |
| Latent distribution comparison plot | `reports/outputs/part_b_diffusion/latent_distributions/comparison_<timestamp>.png` |
| Latent distribution statistics | `reports/outputs/part_b_diffusion/latent_distributions/stats_<timestamp>.json` |

---

## Custom Configuration

To use a different config file, replace the `--config` path:
```bash
uv run src/part_b_diffusion/run_ddpm.py <mode> --config configs/my_custom_config.yaml
```

Where `<mode>` is one of: `train`, `sample`, `fid`, `plot_latents`, `all`

---

## Notes

- **Sample images**: When `sampling.num_samples` is specified, multiple grids are saved (e.g., 256 samples with grid_size 8 produces 4 images)
- **VAE encoder**: Frozen during DDPM training in latent-space mode
- **Dependencies**: Latent-space DDPM uses VAE classes from `src/part_a_vaes/vae_priors.py`
