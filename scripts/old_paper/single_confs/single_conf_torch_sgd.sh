#!/bin/bash
#SBATCH --account=p_deeplearning
#SBATCH --job-name=single_torch_conf_sgd
#SBATCH --error=neps_runs/_log/8/single_torch_conf_sgd/%x/%A.err
#SBATCH --output=neps_runs/_log/8/single_torch_conf_sgd/%x/%A.out
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=capella

# Create _log directory
mkdir -p neps_runs/_log/8/single_torch_conf_sgd/${SLURM_JOB_NAME}

# Activate environment
source ./.venv/bin/activate

# SLURM job arrays range from 1 to n
model_size="8M"
runname="8_single_conf_torch_sgd"
neps_space_config="AdamExtend_f_l_nw" 
config="torch_SGD"
nproc_per_node=4

echo "Running NEPS with model size: $model_size, nproc_per_node: $nproc_per_node"

start_time=$(date +%s)
# Launch torch distributed run on 8 devices
python -u neps_nos/utils/run_single_config.py --model_size $model_size --neps_space_config $neps_space_config --nproc_per_node $nproc_per_node --runname $runname --config $config

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))
echo "Job completed in ${hours}h ${minutes}m ${seconds}s"
