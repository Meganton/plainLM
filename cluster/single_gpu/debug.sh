#!/bin/bash

#SBATCH --account=p_deeplearning
#SBATCH --job-name=test
#SBATCH --error=/work/dlc2workfs2/gebureka-neps_bo/LLM_task/log/%x_%A_%a.err
#SBATCH --output=/work/dlc2workfs2/gebureka-neps_bo/LLM_task/log/%x_%A_%a.out
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=dev_capella
#SBATCH --array=1

# Activate environment
cd ~/plainLM
source .venvbin/activate

# Hyperparmeters are specified in a YAML configuration file
config=config/14M/14M.yaml

# SLURM job arrays range from 1 to n
job_idx=$((SLURM_ARRAY_TASK_ID - 1))

# Launch torch distributed run on 8 devices
python train.py --config=$config --job_idx=$job_idx
