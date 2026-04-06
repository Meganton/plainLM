"""
Entry point script launched by torchrun for each NEPS training evaluation.

Torchrun sets LOCAL_RANK / RANK / WORLD_SIZE as env vars before calling this.
Rank 0 writes the validation loss to <pipeline_directory>/valid_loss.json.
"""
import sys
from pathlib import Path

# Add project root to path so we can import from plainLM root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import os
import json
import dill
import torch

from neps_nos.utils.model_train_function import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt_path", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--n_steps", type=int, required=True)
    parser.add_argument("--pipeline_directory", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--trainset_path", type=str, default=None)
    parser.add_argument("--validset_path", type=str, default=None)
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank != 0:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    with open(args.opt_path, 'rb') as f:
        optimizer_cls = dill.load(f)

    try:
        valid_loss = float(train_model(
            optimizer_cls=optimizer_cls,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pipeline_directory=args.pipeline_directory,
            config_path=args.config_path,
            n_steps=args.n_steps,
            trainset_path=args.trainset_path,
            validset_path=args.validset_path,
        ))
    except Exception as e:
        error_msg = str(e)
        # Log error even if not from rank 0
        import sys as real_sys
        real_sys.stderr.write(f"[Rank {rank}] Training error: {error_msg}\n")
        real_sys.stderr.flush()
        
        if "Train loss is nan" in error_msg or "exceeds" in error_msg or True:  # Handle all errors gracefully
            if rank == 0:
                print(f"Training failed: {error_msg}. Setting loss to inf.")
            valid_loss = float('inf')
        else:
            raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

    if rank == 0:
        result_file = Path(args.pipeline_directory) / "valid_loss.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump({"valid_loss": valid_loss}, f)
        print(f"[Rank 0] Validation loss: {valid_loss}")


if __name__ == "__main__":
    main()
