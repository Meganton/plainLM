import sys
from pathlib import Path

# Add parent directory to path so we can import from plainLM root
sys.path.insert(0, str(Path(__file__).parent.parent))

import neps
import argparse
import os
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

# Helper functions
def get_optimizer_dir_name(optimizer: str, warmstarter: str = None) -> str:
    """Generate directory name for optimizer with optional warmstarter suffix."""
    return optimizer + (f'_{warmstarter}' if warmstarter else "")

def process_neps_status(neps_dir):
    """Extract loss, cost, and fidelity histories from NEPS status."""
    full, _ = neps.status(neps_dir)
    if "objective_to_minimize" not in full or full.empty:
        return None
    
    print("Loss")
    print(full["objective_to_minimize"].head(5))
    print(full["objective_to_minimize"].tail(5))
    print(len(full["objective_to_minimize"]), "total entries in status dataframe.")
    loss_history = full["objective_to_minimize"].fillna(np.inf).tolist()
    print(len(loss_history), "loss entries collected.")
    print(loss_history[:5], "first 5 loss entries")
    print(loss_history[-5:], "last 5 loss entries")
    cost_history = full["cost"].fillna(float(0)).tolist()
    fidelity_history = []
    if "config.ENVIRONMENT__fidelity" in full:
        print("Fidelity found")
        print(full["config.ENVIRONMENT__fidelity"].head(5))
        print(full["config.ENVIRONMENT__fidelity"].tail(5))
        print(len(full["config.ENVIRONMENT__fidelity"]), "total fidelity entries in status dataframe.")
        fidelity_history = full["config.ENVIRONMENT__fidelity"].fillna(float(0)).tolist()
        print(len(fidelity_history), "fidelity entries collected.")
        print(fidelity_history[:5], "first 5 fidelity entries")
        print(fidelity_history[-5:], "last 5 fidelity entries")
    
    return loss_history, cost_history, fidelity_history

def compute_incumbent_history(loss_history, cost_history, fidelity_history):
    """Compute incumbent and cumulative histories from loss/cost data."""
    incumbent_history = []
    cumulated_cost_history = []
    cumulated_fidelity_history = []
    incumbent_loss = np.inf
    
    for i, loss in enumerate(loss_history):
        inc_isnan = pd.isna(incumbent_loss) or np.isnan(incumbent_loss)
        loss_is_nan = pd.isna(loss) or np.isnan(loss)
        if inc_isnan and loss_is_nan:
            incumbent_loss = np.inf
        elif inc_isnan:
            incumbent_loss = loss
        elif not loss_is_nan:
            incumbent_loss = min(incumbent_loss, loss)
        
        incumbent_history.append(incumbent_loss)
        cumulated_cost_history.append(sum(cost_history[:i+1]))
        if fidelity_history:
            cumulated_fidelity_history.append(sum(fidelity_history[:i+1]))
    
    return incumbent_history, cumulated_cost_history, cumulated_fidelity_history

def setup_result_directories(args):
    """Create result directories and handle overwrite mode."""
    run_directory = Path(args.result_dir)
    run_directory.mkdir(parents=True, exist_ok=True)
    
    # Setup directories based on whether runname is provided
    if args.runname is not None:
        # New structure: results_dir/neps/runname/seed_{seed}
        neps_dir = run_directory / "neps" / args.runname / f"seed_{args.seed}"
        results_cache_dir = run_directory / "neps" / args.runname / f"seed_{args.seed}_results_cache"
        results_dir = run_directory / "results"
        results_filename = f"{args.runname}_{args.seed}.json"
    else:
        # Original structure: results_dir/neps/optimizer/seed_{seed}
        neps_dir = run_directory / "neps" / get_optimizer_dir_name(args.neps_optimizer, args.warmstarter) / f"seed_{args.seed}"
        results_cache_dir = run_directory / "neps" / get_optimizer_dir_name(args.neps_optimizer, args.warmstarter) / "results_cache" / f"seed_{args.seed}"
        results_dir = run_directory / "results"
        results_filename = f"{get_optimizer_dir_name(args.neps_optimizer, args.warmstarter)}_{args.seed}.json"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.neps_mode == "overwrite":
        print(f"Overwriting previous results in {results_dir}")
        result_file = results_dir / results_filename
        if result_file.exists():
            print(f"Deleting file: {result_file}")
            result_file.unlink()
    
    print(f"NEPS results will be saved to {neps_dir}")
    # Handle existing neps directory for normal mode
    if neps_dir.exists() and args.neps_mode == "normal":
        warning_msg = f"NEPS directory exists at {neps_dir} but mode is '{args.neps_mode}'."
        warnings.warn(warning_msg, UserWarning)
    neps_dir.parent.mkdir(parents=True, exist_ok=True)
    
    return run_directory, results_dir, neps_dir, results_cache_dir, results_filename

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


def set_global_seeds(seed: int = 42):
    import random
    import numpy as np
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPS NOS Training Pipeline")
    # Example:  python neps_nos/neps_pipeline.py --result_dir neps_runs/testrun1 --neps_space_config NLinesU_f_l_nw --runtime 1
    parser.add_argument("--model_size", type=str, default="8M", choices=["8M", "46M"], help="Model size to use for training.")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--runname", type=str, default=None, help="Unique name to use for neps folder and results file.")
    parser.add_argument("--runtime", type=int, default=None, help="Time budget for the neps run in minutes.")
    parser.add_argument("--evaluations", type=int, default=None, help="Evaluations budget.")
    parser.add_argument("--neps_space_config", type=str, required=True, help="The space config to choose.")
    # Dataset path overrides for $TMPDIR usage
    parser.add_argument("--trainset_path", type=str, default=None, help="Override trainset path (for $TMPDIR usage)")
    parser.add_argument("--validset_path", type=str, default=None, help="Override validset path (for $TMPDIR usage)")
    parser.add_argument("--neps_optimizer", type=str, default="RE", help="The neps optimizer to use.")
    parser.add_argument("--seed", type=int, default=0, help="Seed.")
    parser.add_argument("--warmstarter", type=str, default=None, help="Warmstarter configuration to use.")
    parser.add_argument("--neps_mode", type=str, default="normal", choices=["normal", "continuation", "overwrite", "results"], help="NEPS run mode: 'normal' (default), 'continuation' (resume previous run, so no warmstarting), 'overwrite' (delete and restart), 'results' (skip NEPS and extract results from existing run).")
    parser.add_argument("--nproc_per_node", type=int, default=4, help="Number of GPU processes per node.")
    args = parser.parse_args()

    set_global_seeds(args.seed)

    # Setup directories
    run_directory, results_dir, neps_dir, results_cache_dir, results_filename = setup_result_directories(args)

    min_fidelity = 3 if "8M" in args.model_size else 10
    max_fidelity = 12 if "8M" in args.model_size else 50

    model_config = model_configs[args.model_size]
    space_base_name, space_kwargs = get_space_basename_and_kwargs(args.neps_space_config)
    if "fidelity" in space_kwargs and space_kwargs["fidelity"] is True:
        space_kwargs["fidelity"] = (min_fidelity, max_fidelity)# if args.fidelity_mode == "steps" else (1, 2)
    pipeline_space = get_space_base_callable(space_base_name)(**space_kwargs)
    evaluate_pipeline = partial(
        evaluate_pipeline_base,
        config_path=model_config,
        nproc_per_node=args.nproc_per_node,
        trainset_path=args.trainset_path,
        validset_path=args.validset_path,
    )
    optimizer_base_name, optimizer_kwargs = get_optimizer_name_and_kwargs(args.neps_optimizer)
    print("Pipeline space:\n", space_base_name, "\n", space_kwargs)
    print("\nOptimizer:\n", optimizer_base_name, "\n", optimizer_kwargs)

    logging.basicConfig(level=logging.INFO)
    
    if args.warmstarter is not None and args.neps_mode not in ["continuation", "results"]:
        print(f"\nUsing warmstart configuration: {args.warmstarter}\n")
        warmstart_name, warmstart_kwargs = resolve_warmstarter_name(args.warmstarter)
        # Resolve fidelity value: "min" -> min_fidelity, "max" -> max_fidelity, or use the number directly
        if warmstart_kwargs["fidelity"] == "min":
            warmstart_kwargs["fidelity"] = min_fidelity
        elif warmstart_kwargs["fidelity"] == "max":
            warmstart_kwargs["fidelity"] = max_fidelity
        warmstarter_config, warmstart_pipeline = get_warmstarter_config(space_base_name, space_kwargs, pipeline_space, warmstart_name, warmstart_kwargs)
        warmstart_result = evaluate_pipeline(**warmstart_pipeline, pipeline_directory=str(results_cache_dir))
        neps.import_trials(
            root_directory=neps_dir,
            overwrite_root_directory=(args.neps_mode == "overwrite"),
            evaluated_trials=[(
                warmstarter_config,
                neps.UserResultDict(
                    objective_to_minimize=warmstart_result["objective_to_minimize"],
                    cost=warmstart_result["cost"],
                ),
            )],
            pipeline_space=pipeline_space,
            optimizer=(optimizer_base_name, optimizer_kwargs),
        )

        
    if args.neps_mode == "continuation":
        print("\nContinuing previous NEPS run with\n")
        result = process_neps_status(neps_dir)
        if result is None:
            print("No results collected.")
            sys.exit(0)
        loss_history, cost_history, fidelity_history = result
        print("Previous best loss:", min(loss_history))
        print("Previous total cost:", sum(cost_history))
        if fidelity_history:
            print("Previous total fidelity:", sum(fidelity_history))
    
    if args.neps_mode == "results":
        print("\nRunning in 'results' mode: skipping NEPS execution and extracting results from existing run")
    else:
        try:
            print("\nStarting NEPS run\n")
            neps.run(
                evaluate_pipeline = evaluate_pipeline,
                pipeline_space = pipeline_space,
                overwrite_root_directory=(args.neps_mode == "overwrite" and args.warmstarter is None),
                root_directory=neps_dir,
                optimizer = (optimizer_base_name, optimizer_kwargs),
                cost_to_spend = args.runtime,
                fidelities_to_spend = args.evaluations*max_fidelity if args.evaluations is not None else None,
            )
            print("\nRun complete. Collecting results.")
        except Exception as e:
            print(f"NEPS run interrupted:\n{e}")
            print("Saving results before exiting.")

    # Collect and process results
    result = process_neps_status(neps_dir)
    if result is None:
        print("No results collected.")
        sys.exit(0)
    
    loss_history, cost_history, fidelity_history = result
    incumbent_history, cumulated_cost_history, cumulated_fidelity_history = compute_incumbent_history(
        loss_history, cost_history, fidelity_history
    )
    
    summary_file = results_dir / results_filename
    print(f"\nSaving results to {summary_file}.")
    with open(summary_file, "w") as f:
        import json
        json.dump({
            "loss_history": loss_history,
            "cost_history": cost_history,
            "incumbent_history": incumbent_history,
            "cumulated_cost_history": cumulated_cost_history,
            "fidelity_history": fidelity_history,
            "cumulated_fidelity_history": cumulated_fidelity_history,
        }, f, indent=2)
    print("Results saved.")