#!/bin/bash
#SBATCH --account=p_deeplearning
#SBATCH --job-name=li1_fixed_1_results
#SBATCH --output=neps_runs/_log/result_extraction/li1_fixed_1_results/%a.out
#SBATCH --error=neps_runs/_log/result_extraction/li1_fixed_1_results/%a.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=capella
#SBATCH --array=0-4

# Create log directory in HOME (SLURM writes logs here)
mkdir -p neps_runs/_log/result_extraction/li1_fixed_1_results

# Auto-detect number of GPUs from SLURM allocation (default to 1)
nproc_per_node=${SLURM_GPUS_PER_NODE:-1}

# Create log directory in HOME (periodic sync destination)
mkdir -p neps_runs/_log/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/

# Activate environment
source ./.venv/bin/activate

# Job parameters
neps_optimizer="LI1_fixed1"                   # the NEPS algorithm to use
model_size="46M"
result_dir="neps_runs/46_LI_space"
runtime=1100                                  # ca the runtime in minutes + some overhead
# evaluations=576                             # total number of evaluations to run, gets multiplied with max_fidelity
neps_space_config="NLinesU_f_l_nw"          # the NOS space to search over
# warmstarter="SGDM_inter"
neps_mode="results"                         # results mode - extract results from existing run

echo "Extracting results for: $neps_optimizer, model size: $model_size, seed: $SLURM_ARRAY_TASK_ID"

start_time=$(date +%s)

python -u neps_nos/neps_pipeline.py \
    --seed $SLURM_ARRAY_TASK_ID \
    --neps_optimizer $neps_optimizer \
    --model_size $model_size \
    --result_dir $result_dir \
    --neps_space_config $neps_space_config \
    --nproc_per_node $nproc_per_node \
    --neps_mode $neps_mode \
    --runname li1_fixed_1 \
    --runtime $runtime




echo "Job completed in $(( ($(date +%s) - start_time) / 3600 ))h $(( (($(date +%s) - start_time) % 3600) / 60 ))m $(( ($(date +%s) - start_time) % 60 ))s"
