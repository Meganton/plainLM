#!/bin/bash
#SBATCH --account=hk-project-p0023364
#SBATCH --job-name=GridSearch_full
#SBATCH --error=/scratch/slurm_tmpdir/job_%J/grid_search_%a.err
#SBATCH --output=/scratch/slurm_tmpdir/job_%J/grid_search_%a.out
#SBATCH --time=26:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=accelerated
#SBATCH --array=0-4

nproc_per_node=4    # Has to match the number of gpus requested

# Create _log directory
mkdir -p neps_runs/_log/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}

# Activate environment and Load the TMPDIR wrapper functions
source ./.venv/bin/activate
source neps_cluster_scripts/utils/neps_tmpdir_wrapper.sh

# Job parameters
seed=$((SLURM_ARRAY_TASK_ID))
neps_optimizer="GridSearch"                   # the NEPS algorithm to use
model_size="8M"
result_dir="neps_runs/grid_search"
runname="grid_search_full"    # used as results_dir/neps/runname/... for neps files and as results_dir/results/runname_seed.json for the results file
# runtime=20                                  # ca the runtime in minutes + some overhead
evaluations=576                             # total number of evaluations to run, gets multiplied with max_fidelity
neps_space_config="SmallAdam_f_nl_nw"          # the NOS space to search over
# warmstarter="SGDM_inter"
neps_mode="overwrite"                          # overwrite/continuation/normal -> decides wether to overwrite dir, warmstart again, etc.

echo "Running NEPS with optimizer: $neps_optimizer, model size: $model_size, seed: $seed"

start_time=$(date +%s)

NUM_FILES=10 run_neps_with_tmpdir \
    --seed $seed \
    --neps_optimizer $neps_optimizer \
    --model_size $model_size \
    --result_dir $result_dir \
    --evaluations $evaluations \
    --neps_space_config $neps_space_config \
    --nproc_per_node $nproc_per_node \
    --neps_mode $neps_mode \
    --runname $runname \
    # --runtime $runtime \
    # --warmstarter $warmstarter



echo "Job completed in $(( ($(date +%s) - start_time) / 3600 ))h $(( (($(date +%s) - start_time) % 3600) / 60 ))m $(( ($(date +%s) - start_time) % 60 ))s"