#!/bin/bash
#SBATCH --job-name=short_46
#SBATCH --output=neps_runs/logs/%x/%A/%a.out
#SBATCH --error=neps_runs/logs/%x/%A/%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=testdlc2_gpu-l40s
#SBATCH --array=0-0

# Create log directory
mkdir -p neps_runs/logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/

# Auto-detect number of GPUs from SLURM allocation (default to 2)
nproc_per_node=${SLURM_GPUS_PER_NODE:-2}


# Activate environment
source ./.venv/bin/activate

# Job parameters
seed=$((SLURM_ARRAY_TASK_ID))
neps_optimizer="RS"                   # the NEPS algorithm to use
model_size="46M"
result_dir="neps_runs/tests/short_46"
runname="short_46"    # used as results_dir/neps/runname/... for neps files and as results_dir/results/runname_seed.json for the results file
runtime=10                                  # ca the runtime in minutes + some overhead
# evaluations=576                             # total number of evaluations to run, gets multiplied with max_fidelity
neps_space_config="NLinesU_nf_l_nw"          # the NOS space to search over
# warmstarter="SGDM_inter"
neps_mode="overwrite"                          # overwrite/continuation/normal -> decides wether to overwrite dir, warmstart again, etc.

echo "Running NEPS with optimizer: $neps_optimizer, model size: $model_size, seed: $seed"

start_time=$(date +%s)

# Export debug env vars for NCCL/CUDA/PyTorch to get more diagnostics
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export CUDA_LAUNCH_BLOCKING=1

python -u neps_nos/neps_pipeline.py \
    --seed $seed \
    --neps_optimizer $neps_optimizer \
    --model_size $model_size \
    --result_dir $result_dir \
    --neps_space_config $neps_space_config \
    --nproc_per_node $nproc_per_node \
    --neps_mode $neps_mode \
    --runname $runname \
    --runtime $runtime \
    # --warmstarter $warmstarter
    # --evaluations $evaluations \


echo "Job completed in $(( ($(date +%s) - start_time) / 3600 ))h $(( (($(date +%s) - start_time) % 3600) / 60 ))m $(( ($(date +%s) - start_time) % 60 ))s"