# Part B: DDPM Commands

All commands should be run from the **repo root** (`Deep-Generative-Modeling/`).

## Training

```bash
# Train DDPM with U-Net
uv run src/part_b_diffusion/run_ddpm.py train --config configs/ddpm_unet.yaml
```

## Sampling

```bash
# Generate samples from trained model
uv run src/part_b_diffusion/run_ddpm.py sample --config configs/ddpm_unet.yaml
```

## FID Evaluation

```bash
# Compute FID score and save results to reports/outputs/ddpm_unet/
uv run src/part_b_diffusion/run_ddpm.py fid --config configs/ddpm_unet.yaml
```

## All (Train + Sample + FID)

```bash
# Run full pipeline
uv run src/part_b_diffusion/run_ddpm.py all --config configs/ddpm_unet.yaml
```

## Custom Config

```bash
# Use a different config file
uv run src/part_b_diffusion/run_ddpm.py train --config configs/my_custom_config.yaml
```

## Output Locations

| Output | Path |
|--------|------|
| Model checkpoint | `models/ddpm_unet_T1000_epochs20.pt` |
| Sample images | `samples/ddpm_unet_samples.png` |
| FID results | `reports/outputs/ddpm_unet/run_<timestamp>.json` |

---

## Latent-Space DDPM (β-VAE)

Train DDPM on latent representations from a β-VAE with Gaussian decoder:

```bash
# Train VAE then DDPM in latent space
uv run src/part_b_diffusion/run_ddpm.py train --config configs/ddpm_latent.yaml

# Generate samples (VAE decode from latent DDPM samples)
uv run src/part_b_diffusion/run_ddpm.py sample --config configs/ddpm_latent.yaml

# Compute FID
uv run src/part_b_diffusion/run_ddpm.py fid --config configs/ddpm_latent.yaml

# Full pipeline
uv run src/part_b_diffusion/run_ddpm.py all --config configs/ddpm_latent.yaml
```

### Latent Mode Output Locations

| Output | Path |
|--------|------|
| VAE checkpoint | `models/beta_vae_gaussian_beta0.01_z32.pt` |
| DDPM checkpoint | `models/ddpm_latent_T1000_z32.pt` |
| Sample images | `reports/samples/ddpm_latent_z32_samples.png` |
| FID results | `reports/outputs/ddpm_latent/run_<timestamp>.json` |

### Dependencies

Latent-space DDPM uses components from Part A:
- VAE classes from `src/part_a_vaes/vae_priors_temp.py` (GaussianPrior, Encoder, GaussianDecoder, VAE)
