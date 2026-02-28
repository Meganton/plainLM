#!/bin/bash
#SBATCH --account=hk-project-p0023364
#SBATCH --job-name=short_8_v2
#SBATCH --output=/scratch/slurm_tmpdir/job_%j/logs/%a.out
#SBATCH --error=/scratch/slurm_tmpdir/job_%j/logs/%a.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=accelerated
#SBATCH --array=0-1

# Create log directory in TMPDIR first (SLURM writes logs here)
mkdir -p ${TMPDIR}/logs/

# Auto-detect number of GPUs from SLURM allocation (default to 2)
nproc_per_node=${SLURM_GPUS_PER_NODE:-2}

# Create log directory in HOME (periodic sync destination)
mkdir -p neps_runs/_log/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/

# Activate environment and Load the TMPDIR wrapper functions
source ./.venv/bin/activate
source neps_cluster_scripts/utils/neps_tmpdir_wrapper.sh

# Job parameters
neps_optimizer="RS"                   # the NEPS algorithm to use
model_size="8M"
result_dir="neps_runs/tests/short_8_v2"
runtime=2                                  # runtime in minutes (SLURM 15 min - 2 min overhead = 13 min available)
# evaluations=576                             # total number of evaluations to run, gets multiplied with max_fidelity
neps_space_config="NLinesU_nf_l_nw"          # the NOS space to search over
# warmstarter="SGDM_inter"
neps_mode="overwrite"                             # overwrite/continuation/normal -> decides wether to overwrite dir, warmstart again, etc.

echo "Running NEPS with optimizer: $neps_optimizer, model size: $model_size, seed: $SLURM_ARRAY_TASK_ID"

start_time=$(date +%s)

NUM_FILES=20 SYNC_INTERVAL=1 run_neps_with_tmpdir \
    --seed $SLURM_ARRAY_TASK_ID \
    --neps_optimizer $neps_optimizer \
    --model_size $model_size \
    --result_dir $result_dir \
    --neps_space_config $neps_space_config \
    --nproc_per_node $nproc_per_node \
    --neps_mode $neps_mode \
    --runname $SLURM_JOB_NAME \
    --runtime $runtime \
    # --warmstarter $warmstarter
    # --evaluations $evaluations \




echo "Job completed in $(( ($(date +%s) - start_time) / 3600 ))h $(( (($(date +%s) - start_time) % 3600) / 60 ))m $(( ($(date +%s) - start_time) % 60 ))s"