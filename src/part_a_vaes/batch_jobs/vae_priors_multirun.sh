#!/bin/bash
#BSUB -J vae
#BSUB -q gpuv100
#BSUB -W 06:00
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


# Gaussian (Baseline)
# uv run python vae_priors.py multirun --prior gaussian --latent-dim 20 --epochs 50 --lr 1e-3 --runs 5

# MoG (Structured)
# uv run python vae_priors.py multirun --prior mog --latent-dim 20 --epochs 50 --lr 1e-3 --runs 5

# Flow (Flexible)
uv run python vae_priors.py multirun --prior flow --latent-dim 20 --epochs 100 --lr 5e-4 --flow-layers 8 --flow-hidden 256 --runs 5