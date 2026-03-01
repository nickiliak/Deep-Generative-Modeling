#!/bin/bash
#BSUB -J vae
#BSUB -q gpuv100
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o logs_jobs/vae_%J.out
#BSUB -e logs_jobs/vae_%J.err
#BSUB -u alvarocuele@gmail.com
#BSUB -N

# Go to submission directory
cd "$LS_SUBCWD" || exit 1

# Create logs folder
mkdir -p logs_jobs

# Ensure local imports work
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "Running on: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"


# Run with uv for Gaussian Prior
# uv run python -u vae_priors.py train \
#   --prior gaussian \
#   --device cuda \
#   --latent-dim 20 \
#   --epochs 50 \
#   --batch-size 256 \
#   --model gaussian_model.pt \
#   --lr 1e-3

# Run with uv for MoG Prior
uv run python -u vae_priors.py train \
  --prior mog \
  --device cuda \
  --latent-dim 20 \
  --epochs 50 \
  --batch-size 256 \
  --model mog_model.pt \
  --lr 1e-3

  # Run with uv for Flow
#   uv run python -u vae_priors.py train \
#   --prior flow \
#   --device cuda \
#   --latent-dim 2 \
#   --epochs 100 \
#   --batch-size 256 \
#   --model flow_model_v2.pt \
#   --flow-layers 8 \
#   --flow-hidden 256 \
#   --lr 1e-4

# Run with uv
# uv run python -u vae_priors.py train \
#   --prior flow \
#   --device cuda \
#   --latent-dim 20 \
#   --epochs 50 \
#   --batch-size 256 \
#   --model flow_model.pt \
#   --flow-layers 8 \
#   --flow-hidden 256 \
#   --lr 5e-4