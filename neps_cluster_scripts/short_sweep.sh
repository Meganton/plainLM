#!/bin/bash
#SBATCH --account=hk-project-p0023364
#SBATCH --job-name=short_sweep
#SBATCH --error=/scratch/slurm_tmpdir/%x/job_%j/%x_%a.err
#SBATCH --output=/scratch/slurm_tmpdir/%x/job_%j/%x_%a.out
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=accelerated
#SBATCH --array=0-0

# Auto-detect number of GPUs from SLURM allocation (default to 1)
nproc_per_node=${SLURM_GPUS_PER_NODE:-1}

# Create _log directory
mkdir -p neps_runs/_log/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}

# Activate environment and Load the TMPDIR wrapper functions
source ./.venv/bin/activate
source neps_cluster_scripts/utils/neps_tmpdir_wrapper.sh

# Job parameters
seed=$((SLURM_ARRAY_TASK_ID))
neps_optimizer="LI1_r0.3"                   # the NEPS algorithm to use
model_size="8M"
result_dir="neps_runs/tests/short_lr_sweep"
runname="short_sweep"    # used as results_dir/neps/runname/... for neps files and as results_dir/results/runname_seed.json for the results file
runtime=5                                  # ca the runtime in minutes + some overhead
# evaluations=576                             # total number of evaluations to run, gets multiplied with max_fidelity
neps_space_config="NLinesU_f_nl_nw"          # the NOS space to search over
# warmstarter="SGDM_inter"
neps_mode="normal"                          # overwrite/continuation/normal -> decides wether to overwrite dir, warmstart again, etc.

echo "Running NEPS with optimizer: $neps_optimizer, model size: $model_size, seed: $seed"

start_time=$(date +%s)

NUM_FILES=20 run_neps_with_tmpdir \
    --seed $seed \
    --neps_optimizer $neps_optimizer \
    --model_size $model_size \
    --result_dir $result_dir \
    --neps_space_config $neps_space_config \
    --nproc_per_node $nproc_per_node \
    --neps_mode $neps_mode \
    --runname $runname \
    --runtime $runtime \
    --lr_mode "sweep" \
    # --warmstarter $warmstarter
    # --evaluations $evaluations \




echo "Job completed in $(( ($(date +%s) - start_time) / 3600 ))h $(( (($(date +%s) - start_time) % 3600) / 60 ))m $(( ($(date +%s) - start_time) % 60 ))s"