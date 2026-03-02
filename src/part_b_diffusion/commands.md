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
| Model checkpoint | `models/ddpm_unet_T{T}_epochs{epochs}.pt` |
| Sample images | `reports/samples/ddpm_unet_samples_*.png` |
| FID results | `reports/outputs/ddpm_unet/run_<timestamp>.json` |

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

### Output Locations
| Output | Path |
|--------|------|
| VAE checkpoint | `models/beta_vae_gaussian_beta{beta}_z{latent_dim}.pt` |
| DDPM checkpoint | `models/ddpm_latent_T{T}_z{latent_dim}.pt` |
| Sample images | `reports/samples/ddpm_latent_z{latent_dim}_samples_*.png` |
| FID results | `reports/outputs/ddpm_latent/run_<timestamp>.json` |

---

## Custom Configuration

To use a different config file, replace the `--config` path:
```bash
uv run src/part_b_diffusion/run_ddpm.py <mode> --config configs/my_custom_config.yaml
```

Where `<mode>` is one of: `train`, `sample`, `fid`, `all`

---

## Notes

- **Sample images**: When `sampling.num_samples` is specified, multiple grids are saved (e.g., 256 samples with grid_size 8 produces 4 images)
- **VAE encoder**: Frozen during DDPM training in latent-space mode
- **Dependencies**: Latent-space DDPM uses VAE classes from `src/part_a_vaes/vae_priors_temp.py`
