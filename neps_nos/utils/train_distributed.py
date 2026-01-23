"""Wrapper script for distributed training called by NEPS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Profiler function.*will be ignored")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in cast.*")

import logging
import os
# Suppress PyTorch profiler warnings before any torch imports
os.environ['KINETO_LOG_LEVEL'] = '5'
logging.getLogger("torch._logging").setLevel(logging.CRITICAL)
logging.getLogger("torch._logging._internal").setLevel(logging.CRITICAL)

import argparse
import json
import os
import dill as pickle

import torch
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer_config_file", type=str, required=True, help="Path to pickled optimizer config")
    parser.add_argument("--pipeline_directory", type=str, required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--config_path", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--n_steps", type=int, default=None, help="Number of training steps (epochs multiplier). If not specified, uses config default.")
    # Dataset path overrides for $TMPDIR support
    parser.add_argument("--trainset_path", type=str, default=None, help="Override trainset path (for $TMPDIR usage)")
    parser.add_argument("--validset_path", type=str, default=None, help="Override validset path (for $TMPDIR usage)")
    args = parser.parse_args()
    
    # Get distributed training info
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"[Rank {rank}/{world_size}] Process started on device {local_rank}")
    
    # Import neps_nos_space before unpickling so the module is available
    try:
        import neps_nos.neps_nos_space
    except ImportError:
        pass  # Module might not be needed for all optimizer types
    
    # Load optimizer configuration from pickle file
    if rank == 0:
        print(f"[Rank 0] Loading optimizer config from {args.optimizer_config_file}")
    with open(args.optimizer_config_file, 'rb') as f:
        optimizer_config = pickle.load(f)
    
    optimizer_cls = optimizer_config['optimizer_cls']
    lr = optimizer_config['lr']
    weight_decay = optimizer_config['weight_decay']
    
    if rank == 0:
        print(f"[Rank 0] Optimizer: {optimizer_cls.__name__ if hasattr(optimizer_cls, '__name__') else str(optimizer_cls)}, LR: {lr}, Weight Decay: {weight_decay}")
    
    # Import train_model here to ensure DDP initialization happens correctly
    from neps_nos.utils.model_train_function import train_model
    
    if rank == 0:
        print(f"[Rank 0] Starting distributed training...")
    
    try:
        valid_loss = train_model(
            optimizer_cls=optimizer_cls,
            lr=lr,
            weight_decay=weight_decay,
            pipeline_directory=args.pipeline_directory,
            config_path=args.config_path,
            n_steps=args.n_steps if args.n_steps is not None else -1,
            trainset_path=args.trainset_path,  # Pass through for $TMPDIR support
            validset_path=args.validset_path,  # Pass through for $TMPDIR support
        )
        
        # Convert to float in case it's a tensor or numpy type
        valid_loss = float(valid_loss)
        
        if rank == 0:
            print(f"[Rank 0] Training completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        if "Train loss is nan" in error_msg or "exceeds" in error_msg:
            print(f"Training failed: {error_msg}. Setting loss to inf.")
            valid_loss = float('inf')
        else:
            # Re-raise for other errors
            raise
    
    # Save result to file so parent process can read it
    # Only the master process (rank 0) should write the result
    if rank == 0:
        result_file = Path(args.pipeline_directory) / "valid_loss.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Rank 0] Writing validation loss to {result_file}")
        with open(result_file, 'w') as f:
            json.dump({"valid_loss": valid_loss}, f)
        print(f"[Rank 0] Validation loss: {valid_loss}")
    
    # Clean up GPU memory before exiting the subprocess
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Clean up distributed process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    print(f"[Rank {rank}/{world_size}] Process finished")
