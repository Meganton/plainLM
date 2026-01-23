#!/bin/bash
#SBATCH --account=hk-project-p0023364
#SBATCH --job-name=LI1_r0.3_simple
#SBATCH --error=neps_runs/_log/test_runs/wrapper_test/%x/%A/%a.err
#SBATCH --output=neps_runs/_log/test_runs/wrapper_test/%x/%A/%a.out
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=accelerated
#SBATCH --array=0-0

# Create _log directory
mkdir -p neps_runs/_log/test_runs/wrapper_test/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}

# Activate environment and Load the TMPDIR wrapper functions
source ./.venv/bin/activate
source neps_cluster_scripts/utils/neps_tmpdir_wrapper.sh

# Job parameters
seed=$((SLURM_ARRAY_TASK_ID))
neps_optimizer="LI1_r0.3"                   # the NEPS algorithm to use
model_size="8M"
result_dir="neps_runs/test_runs/wrapper_test"
runname="NLinesU_LI1_r03" # used as results_dir/neps/runname/... for neps files and as results_dir/results/runname_seed.json for the results file
runtime=20 # ca the runtime in minutes + some overhead
neps_space_config="NLinesU_f_l_nw"          # the NOS space to search over
warmstarter="SGDM_inter"
nproc_per_node=2        # Has to match the number of gpus requested
neps_mode="overwrite"   # overwrite/continuation/normal -> decides wether to overwrite dir, warmstart again, etc.

echo "Running NEPS with optimizer: $neps_optimizer, model size: $model_size, seed: $seed"

start_time=$(date +%s)

NUM_FILES=10 run_neps_with_tmpdir \
    --seed $seed \
    --neps_optimizer $neps_optimizer \
    --model_size $model_size \
    --result_dir $result_dir \
    --runtime $runtime \
    --neps_space_config $neps_space_config \
    --nproc_per_node $nproc_per_node \
    --neps_mode $neps_mode \
    --runname $runname \
    --warmstarter $warmstarter


end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))
echo "Job completed in ${hours}h ${minutes}m ${seconds}s"
