#!/bin/bash
#SBATCH --job-name=short_cacher
#SBATCH --output=neps_runs/logs/%x/%A/%a.out
#SBATCH --error=neps_runs/logs/%x/%A/%a.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=testdlc2_gpu-l40s
#SBATCH --array=0-0


# Create log directory
mkdir -p neps_runs/logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/

nproc_per_node=2

# Activate environment and Load the TMPDIR wrapper functions
source ./.venv/bin/activate

# Job parameters
evals=2

start_time=$(date +%s)

python -u neps_nos/cacher.py \
    --evals $evals \


echo "Job completed in $(( ($(date +%s) - start_time) / 3600 ))h $(( (($(date +%s) - start_time) % 3600) / 60 ))m $(( ($(date +%s) - start_time) % 60 ))s"