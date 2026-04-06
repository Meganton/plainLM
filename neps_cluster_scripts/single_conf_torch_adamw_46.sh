#!/bin/bash
#SBATCH --account=hk-project-p0023364
#SBATCH --job-name=single_torch_conf_adamw_46
#SBATCH --error=neps_runs/_log/46/single_torch_conf_adamw/%x/%A.err
#SBATCH --output=neps_runs/_log/46/single_torch_conf_adamw/%x/%A.out
#SBATCH --time=01:40:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=accelerated

# Create _log directory
mkdir -p neps_runs/_log/46/single_torch_conf_adamw_46/${SLURM_JOB_NAME}

# Activate environment
source ./.venv/bin/activate

model_size="46M"
runname="46_single_conf_torch_adamw"
neps_space_config="AdamExtend_f_l_nw"
config="torch_AdamW"
nproc_per_node=4

echo "Running NEPS with model size: $model_size, nproc_per_node: $nproc_per_node"

start_time=$(date +%s)
python -u neps_nos/utils/run_single_config.py --model_size $model_size --neps_space_config $neps_space_config --nproc_per_node $nproc_per_node --runname $runname --config $config

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))
echo "Job completed in ${hours}h ${minutes}m ${seconds}s"
