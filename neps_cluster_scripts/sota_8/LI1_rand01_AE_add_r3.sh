#!/bin/bash
#SBATCH --account=hk-project-p0023364
#SBATCH --job-name=sota_li1_rand_01_r3_ae_add
#SBATCH --output=/scratch/slurm_tmpdir/job_%j/logs/%a.out
#SBATCH --error=/scratch/slurm_tmpdir/job_%j/logs/%a.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=accelerated
#SBATCH --array=0-4

# Create log directory in TMPDIR first (SLURM writes logs here)
mkdir -p ${TMPDIR}/logs/

# Auto-detect number of GPUs from SLURM allocation (default to 4)
nproc_per_node=${SLURM_GPUS_PER_NODE:-4}

# Create log directory in HOME (periodic sync destination)
mkdir -p neps_runs/_log/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/

# Activate environment and Load the TMPDIR wrapper functions
source ./.venv/bin/activate
source neps_cluster_scripts/utils/neps_tmpdir_wrapper.sh

# Job parameters
neps_optimizer="LI1_rand0.1r3"                   # the NEPS algorithm to use
model_size="8M"
result_dir="neps_runs/sota_8"
runtime=2600                                  # ca the runtime in minutes + some overhead
# evaluations=576                             # total number of evaluations to run, gets multiplied with max_fidelity
neps_space_config="AdamExtend_f_l_nw"          # the NOS space to search over
# warmstarter="SGDM_inter"
neps_mode="normal"                             # overwrite/continuation/normal -> decides wether to overwrite dir, warmstart again, etc.

echo "Running NEPS with optimizer: $neps_optimizer, model size: $model_size, seed: $SLURM_ARRAY_TASK_ID"

start_time=$(date +%s)

NUM_FILES=20 SYNC_INTERVAL=2 run_neps_with_tmpdir \
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