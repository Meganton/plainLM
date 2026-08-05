import sqlite3
import os
from neps_nos_space import NOSSpaceNLinesU
import argparse
import sys
from pathlib import Path

# Add parent directory to path so we can import from plainLM root
sys.path.insert(0, str(Path(__file__).parent.parent))

import neps
import argparse
import json
import socket
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in cast.*")
import subprocess
import cloudpickle
import torch
from functools import partial
from neps_nos.utils.model_train_function import CFG_PATH_46M, CFG_PATH_8M
from neps_nos.neps_config import get_space_basename_and_kwargs, get_space_base_callable, get_optimizer_name_and_kwargs, get_warmstarter_config, resolve_warmstarter_name
import logging
import time
import numpy as np
import pandas as pd

model_configs = {
    "8M": CFG_PATH_8M,
    "46M": CFG_PATH_46M,
}

DATABASE_SUFFIX = ".sqlite3"


def set_global_seeds(seed: int = 42):
    import random
    import numpy as np
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def _load_all_keys(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute("SELECT key FROM embeddings")
    }

def _open_database(database_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            key TEXT PRIMARY KEY,
            vec BLOB NOT NULL
        ) WITHOUT ROWID
        """
    )
    return connection

def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_training(
    optimizer_cls,
    learning_rate: float,
    weight_decay: float,
    fidelity: int,
    pipeline_directory: str,
    config_path: str,
    nproc_per_node: int,
    trainset_path=None,
    validset_path=None,
    ) -> float:
    """Launch distributed training via torchrun and return the validation loss."""
    pipeline_directory = Path(pipeline_directory)
    pipeline_directory.mkdir(parents=True, exist_ok=True)

    # Serialize optimizer to a file; torchrun_worker.py deserializes it.
    opt_file = pipeline_directory / "optimizer.cloudpickle"
    with open(opt_file, 'wb') as f:
        cloudpickle.dump(optimizer_cls, f)

    port = _find_free_port()
    worker_script = Path(__file__).parent / "utils" / "torchrun_worker.py"

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        "--tee=3",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={port}",
        str(worker_script),
        "--opt_path", str(opt_file),
        "--lr", str(learning_rate),
        "--weight_decay", str(weight_decay),
        "--n_steps", str(fidelity),
        "--pipeline_directory", str(pipeline_directory),
        "--config_path", str(config_path),
    ]
    if trainset_path is not None:
        cmd += ["--trainset_path", str(trainset_path)]
    if validset_path is not None:
        cmd += ["--validset_path", str(validset_path)]

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
        # Helps reduce allocator fragmentation across repeated trials.
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    try:
        print(f"[NEPS] Launching {nproc_per_node} training processes via torchrun (port {port})...")
        print("-" * 80)
        result = subprocess.run(cmd, env=env)
        print("-" * 80)

        result_file = pipeline_directory / "valid_loss.json"
        if result.returncode != 0 or not result_file.exists():
            print(f"[NEPS] Training failed (exit code {result.returncode}). Returning inf loss.")
            valid_loss = float('inf')
        else:
            with open(result_file) as f:
                valid_loss = json.load(f)["valid_loss"]
            print(f"[NEPS] Training complete. Validation loss: {valid_loss}")
    finally:
        opt_file.unlink(missing_ok=True)
        if torch.cuda.is_available():
            # Give elastic workers a moment to fully tear down before next trial.
            time.sleep(5)
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

    return valid_loss

def evaluate_pipeline_base(
    optimizer_cls,
    pipeline_directory: str,
    learning_rate: float = 0.001,
    weight_decay: float = 0,
    fidelity: int = -1,
    config_path: str = CFG_PATH_8M,
    nproc_per_node: int = 4,
    trainset_path=None,
    validset_path=None,
    ):
    """
    Evaluate a configuration by training a model.

    Returns:
        Dictionary with objective_to_minimize (validation loss) and cost (time in minutes)
    """
    print(f"Training using {pipeline_directory}\nand config {config_path}")
    print(f"Fidelity: {fidelity}, LR: {learning_rate}, WD: {weight_decay}")
    x = torch.zeros(1)
    print(f"Optimizer instance:\n{optimizer_cls([x], lr=learning_rate, weight_decay=weight_decay)}")

    start_time = time.time()
    valid_loss = run_training(
        optimizer_cls=optimizer_cls,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fidelity=fidelity,
        pipeline_directory=pipeline_directory,
        config_path=config_path,
        nproc_per_node=nproc_per_node,
        trainset_path=trainset_path,
        validset_path=validset_path,
    )
    cost = np.round((time.time() - start_time) / 60, 2)
    print(f"Training ended after {time.strftime('%H:%M:%S', time.gmtime(cost * 60))}.")
    return {"objective_to_minimize": valid_loss, "cost": cost}




spaces = {
    "NOSLineU": NOSSpaceNLinesU,
}

dataset_base_path = "/work/dlc2workfs2/gebureka-neps_bo/kernel_experiments/embeddings"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPS NOS Training Pipeline")
    # Example:  python neps_nos/neps_pipeline.py --result_dir neps_runs/testrun1 --neps_space_config NLinesU_f_l_nw --runtime 1
    parser.add_argument("--model_size", type=str, default="8M", choices=["8M", "46M"], help="Model size to use for training.")
    parser.add_argument("--result_dir", type=str, default="./neps_nos/cache", help="Directory to save checkpoints and logs.")
    parser.add_argument("--space", type=str, default="NOSLineU", help="The space to choose.")
    parser.add_argument("--seed", type=int, default=0, help="Seed.")
    parser.add_argument("--nproc_per_node", type=int, default=2, help="Number of GPU processes per node.")
    parser.add_argument("--evals", type=int, default=-1, help="Number of evaluations to run.")
    args = parser.parse_args()

    set_global_seeds(args.seed)

    database_path = os.path.join(dataset_base_path, f"{args.space}{DATABASE_SUFFIX}")
    connection = _open_database(database_path)
    all_keys = _load_all_keys(connection)
    all_keys = [key for key in all_keys if ": " in key] # Only keep full configs

    # Create csv file if it doesn't exist
    os.makedirs(args.result_dir, exist_ok=True)
    csv_path = os.path.join(args.result_dir, f"{args.space}_seed_{args.seed}_{args.model_size}_results.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("key,valid_loss,cost\n")
    n_keys_csv = sum(1 for _ in open(csv_path)) - 1  # Subtract header line

    # Go through all keys starting at n_keys_csv and compute valid_loss for each key, saving it
    for i, config_str in enumerate(all_keys[n_keys_csv:]):
        print(f"Processing key {i+1}/{len(all_keys)}:\n {config_str}")

        conf = {key.split(": ")[0][1:-1]: eval(key.split(": ")[1]) for key in config_str.split(", ")}
        conf_dict = neps.load_config(pipeline_space=spaces[args.space](), config=conf)

        result = evaluate_pipeline_base(
            optimizer_cls=conf_dict["optimizer_cls"],
            pipeline_directory=Path("cache") / "tmp",
            learning_rate = 0.001,
            weight_decay = 0,
            fidelity = -1,
            config_path = model_configs[args.model_size],
            nproc_per_node = args.nproc_per_node,
        )
        valid_loss = result["objective_to_minimize"]
        cost = result["cost"]

        with open(csv_path, "a") as f:
            f.write(f"{config_str},{valid_loss},{cost}\n")

        if args.evals != -1 and i + 1 > args.evals:
            print(f"Completed {i + 1} evaluations. Current results saved to {csv_path}.")
            break
