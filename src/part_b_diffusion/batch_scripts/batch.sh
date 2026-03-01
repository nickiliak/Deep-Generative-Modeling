#!/bin/sh
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J vdm_15_epochs
### -- ask for number of cores --
#BSUB -n 1
### -- Select the resources: 1 gpu --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: 30 minutes is plenty for 15 epochs --
#BSUB -W 0:30
### -- request 8GB of system-memory (safer for data loading) --
#BSUB -R "rusage[mem=8GB]"
### -- Specify the output and error file. %J is the job-id --
#BSUB -o vdm_%J.out
#BSUB -e vdm_%J.err

# 1. Activate your virtual environment
source /zhome/07/3/219372/pokemon_env/bin/activate

# 2. Run the training script
# We keep your specific path from the second script
python /zhome/07/3/219372/vision/vision/MLOps2026/mlops2026/src/vdm_pokemon/train.py