#!/bin/bash
#SBATCH --job-name=short_8
#SBATCH --output=logs/%x/%A/%a.out
#SBATCH --error=logs/%x/%A/%a.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=testdlc2_gpu-l40s
#SBATCH --array=0-0


# Create log directory
mkdir -p logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/

# Auto-detect number of GPUs from SLURM allocation (default to 1 for this test)
echo "Detected $SLURM_GPUS_PER_NODE GPUs per Node"
nproc_per_node=${SLURM_GPUS_PER_NODE:-2}

# Activate environment and Load the TMPDIR wrapper functions
source ./.venv/bin/activate

# Job parameters
neps_optimizer="HB"                   # the NEPS algorithm to use
model_size="8M"
result_dir="neps_runs/tests/short_8"
runtime=7                                  # runtime in minutes (SLURM 10 min - 2 min overhead = 8 min available)
# evaluations=576                             # total number of evaluations to run, gets multiplied with max_fidelity
neps_space_config="AdamMore"          # the NOS space to search over
# warmstarter="SGDM_inter"
neps_mode="overwrite"                             # overwrite/continuation/normal -> decides wether to overwrite dir, warmstart again, etc.

echo "Running NEPS with optimizer: $neps_optimizer, model size: $model_size, seed: $SLURM_ARRAY_TASK_ID"

start_time=$(date +%s)

python -u neps_nos/neps_pipeline.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --neps_optimizer $neps_optimizer \
    --model_size $model_size \
    --result_dir $result_dir \
    --neps_space_config $neps_space_config \
    --nproc_per_node $nproc_per_node \
    --neps_mode $neps_mode \
    --runname $SLURM_JOB_NAME \
    --runtime $runtime


echo "Job completed in $(( ($(date +%s) - start_time) / 3600 ))h $(( (($(date +%s) - start_time) % 3600) / 60 ))m $(( ($(date +%s) - start_time) % 60 ))s"