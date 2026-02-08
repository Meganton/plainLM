import sys
from pathlib import Path

# Add parent directory to path so we can import from plainLM root
sys.path.insert(0, str(Path(__file__).parent.parent))

import neps
import argparse
import os
import torch
import subprocess
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in cast.*")
from typing import Literal
try:
    import dill as pickle
except ImportError:
    import pickle
    print("Warning: dill not available, using pickle. This may fail with dynamic optimizer classes.")
import tempfile
from functools import partial
from neps_nos.utils.model_train_function import train_model, CFG_PATH_46M, CFG_PATH_8M, CFG_PATH_46M_1D, CFG_PATH_8M_1D
from neps_nos.neps_config import get_space_basename_and_kwargs, get_space_base_callable, get_optimizer_name_and_kwargs, get_warmstarter_config, resolve_warmstarter_name
import logging
from pprint import pprint
import time
import numpy as np
import pandas as pd

model_configs = {
    "8M_1D": CFG_PATH_8M_1D,
    "46M_1D": CFG_PATH_46M_1D,
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
    
    loss_history = full["objective_to_minimize"].fillna(np.inf).tolist()
    cost_history = full["cost"].fillna(float(0)).tolist()
    fidelity_history = []
    if "config.ENVIRONMENT__fidelity" in full:
        fidelity_history = full["config.ENVIRONMENT__fidelity"].fillna(float(0)).tolist()
    
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

def run_distributed_training(
    optimizer_cls,
    learning_rate: float,
    weight_decay: float,
    fidelity: int,
    pipeline_directory: str,
    config_path: str,
    nproc_per_node: int,
    trainset_path: str = None,  # For $TMPDIR support
    validset_path: str = None,  # For $TMPDIR support
) -> float:
    """
    Run distributed training via torchrun subprocess.
    
    Args:
        optimizer_cls: The optimizer class to use
        learning_rate: Learning rate for training
        weight_decay: Weight decay for training
        fidelity: Fidelity level (number of training steps)
        pipeline_directory: Directory to save checkpoints and logs
        config_path: Path to the model configuration YAML file
        nproc_per_node: Number of processes per node
    
    Returns:
        Validation loss value
    """
    # Serialize optimizer configuration to a temporary pickle file
    optimizer_config = {
        'optimizer_cls': optimizer_cls,
        'lr': learning_rate,
        'weight_decay': weight_decay,
    }
    
    print(f"[NEPS] Preparing distributed training (nproc={nproc_per_node})")
    if hasattr(optimizer_cls, '__name__'):
        optimizer_name = getattr(optimizer_cls, '__name__', str(optimizer_cls))
    else:
        optimizer_name = str(optimizer_cls)
    print(f"[NEPS] Optimizer class: {optimizer_name}")
    
    # Create temporary file for optimizer config
    config_fd, config_file = tempfile.mkstemp(suffix='.pkl', prefix='optimizer_config_')
    result = None  # Initialize to avoid unbound variable warning
    try:
        with os.fdopen(config_fd, 'wb') as f:
            pickle.dump(optimizer_config, f)
        
        print(f"[NEPS] Serialized optimizer config to: {config_file}")
        
        # Set environment variables
        env = os.environ.copy()
        env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Legacy name for older PyTorch
        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
        
        cmd = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={nproc_per_node}",
            "--redirects", "1:0,2:0,3:0",  # Redirect output from ranks 1,2,3 to rank 0
            "neps_nos/utils/train_distributed.py",
            "--optimizer_config_file", config_file,
            "--n_steps", str(fidelity),
            "--pipeline_directory", str(pipeline_directory),
            "--config_path", str(config_path),
        ]
        
        # Add dataset path overrides if provided (for $TMPDIR support)
        if trainset_path is not None:
            cmd.extend(["--trainset_path", str(trainset_path)])
        if validset_path is not None:
            cmd.extend(["--validset_path", str(validset_path)])
        
        print(f"[NEPS] Spawning subprocess: {' '.join(cmd)}")
        print(f"[NEPS] Launching {nproc_per_node} training processes...")
        print("-" * 80)
        # Remove capture_output to allow real-time streaming
        result = subprocess.run(cmd, check=False, text=True, env=env)
        print("-" * 80)
    finally:
        # Clean up temporary config file
        print(f"[NEPS] Cleaning up temporary config file: {config_file}")
        if os.path.exists(config_file):
            os.unlink(config_file)
    
        if result is None:
            raise ValueError("Training subprocess was not started properly")
        elif result.returncode != 0:
            print(f"[NEPS] Subprocess failed with return code {result.returncode}")
            # Note: stdout/stderr were streamed in real-time, not captured
            # Check for result file or assume failure
            result_file = Path(pipeline_directory) / "valid_loss.json"
            if result_file.exists():
                print("[NEPS] Found result file despite non-zero exit code, reading it...")
                with open(result_file) as f:
                    data = json.load(f)
                valid_loss = data["valid_loss"]
                if not np.isfinite(valid_loss):
                    print("[NEPS] Training failed due to NaN or inf loss. Returning inf loss.")
                    valid_loss = np.inf
            else:
                print("[NEPS] Training failed and no result file found. Returning inf loss.")
                valid_loss = np.inf
        else:
            # Read result from file written by rank 0
            result_file = Path(pipeline_directory) / "valid_loss.json"
            print(f"[NEPS] Reading result from {result_file}")
            if not result_file.exists():
                raise ValueError(f"Result file not found: {result_file}")
            with open(result_file) as f:
                data = json.load(f)
            valid_loss = data["valid_loss"]
            print(f"[NEPS] Distributed training completed, validation loss: {valid_loss}")
            
        # Clean up GPU memory between trials with delay to allow subprocess cleanup
        # Force GPU synchronization to ensure all operations complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            time.sleep(5)  # Increased delay to allow full GPU cleanup
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print("[NEPS] Cleaned up GPU memory in main process")
    
    return valid_loss


def evaluate_pipeline_base(
    optimizer_cls,
    pipeline_directory: str,
    learning_rate: float = 0.001,
    weight_decay: float = 0,
    fidelity: int = -1,
    fidelity_mode: Literal["steps", "model_size"] = "steps",
    lr_mode: Literal["normal", "sweep"] = "normal",
    config_path: str = CFG_PATH_8M_1D,
    nproc_per_node: int = 1,
    trainset_path: str = None,  # For $TMPDIR support
    validset_path: str = None,  # For $TMPDIR support
):
    """
    Evaluate a configuration by training a model.
    
    Args:
        optimizer_cls: The optimizer class to use
        pipeline_directory: Directory to save checkpoints and logs
        learning_rate: Learning rate for training
        weight_decay: Weight decay for training
        fidelity: Fidelity level
        config_path: Path to the model configuration YAML file
        nproc_per_node: Number of processes per node (1=single process, >1=distributed training)
    
    Returns:
        Dictionary with objective_to_minimize (validation loss) and cost (time in minutes)
    """
    
    print(f"Training using {pipeline_directory}\nand config {config_path}")
    print(f"Fidelity: {fidelity}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    x = torch.zeros(1)
    print(f"Optimizer instance:\n{optimizer_cls([x], lr=learning_rate, weight_decay=weight_decay)}")

    start_time = time.time()

    if fidelity_mode == "model_size":
        raise NotImplementedError("Fidelity mode 'model_size' is not implemented yet, as large model does not fit into GPUs")
    
    if lr_mode == "sweep":
        valid_loss = np.inf
        for lr in [0.002, 0.001, 0.0005, 0.00025]:
            print(f"Running learning rate sweep with lr={lr}")
            sweep_valid_loss = run_distributed_training(
                optimizer_cls=optimizer_cls,
                learning_rate=lr,
                weight_decay=weight_decay,
                fidelity=fidelity,
                pipeline_directory=pipeline_directory,
                config_path=config_path,
                nproc_per_node=nproc_per_node,
                trainset_path=trainset_path,  # Pass through for $TMPDIR support
                validset_path=validset_path,  # Pass through for $TMPDIR support
            )
            print(f"Learning rate {lr} resulted in validation loss: {sweep_valid_loss}")
            valid_loss = min(valid_loss, sweep_valid_loss)
        print(f"Best validation loss from learning rate sweep: {valid_loss}")
    else:
        valid_loss = run_distributed_training(
            optimizer_cls=optimizer_cls,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fidelity=fidelity,
            pipeline_directory=pipeline_directory,
            config_path=config_path,
            nproc_per_node=nproc_per_node,
            trainset_path=trainset_path,  # Pass through for $TMPDIR support
            validset_path=validset_path,  # Pass through for $TMPDIR support
        )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training ended after {time.strftime('%H:%M:%S', time.gmtime(total_time))}.")
    cost = np.round(total_time/60, 2)  # Cost in minutes rounded to 2 decimal places

    return {"objective_to_minimize": valid_loss, "cost": cost}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPS NOS Training Pipeline")
    # Example:  python neps_nos/neps_pipeline.py --result_dir neps_runs/testrun1 --neps_space_config NLinesU_f_l_nw --runtime 1
    parser.add_argument("--model_size", type=str, default="8M", choices=["8M_1D", "46M_1D", "8M", "46M"], help="Model size to use for training.")
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
    parser.add_argument("--lr_mode", type=str, default="normal", choices=["normal", "sweep"], help="Learning rate mode: 'normal' (default) or 'sweep'.")
    # parser.add_argument("--fidelity_mode", type=str, default="steps", choices=["steps", "model_size"], help="Fidelity mode to use: 'steps' or 'model_size'.")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node (1=single process, >1=distributed with torchrun).")
    args = parser.parse_args()

    # Setup directories
    run_directory, results_dir, neps_dir, results_cache_dir, results_filename = setup_result_directories(args)

    min_fidelity = 3 if "8M" in args.model_size else 10
    max_fidelity = 12 if "8M" in args.model_size else 50

    model_config = model_configs[args.model_size + ("_1D" if args.nproc_per_node == 1 else "")]
    space_base_name, space_kwargs = get_space_basename_and_kwargs(args.neps_space_config)
    if "fidelity" in space_kwargs and space_kwargs["fidelity"] is True:
        space_kwargs["fidelity"] = (min_fidelity, max_fidelity)# if args.fidelity_mode == "steps" else (1, 2)
    pipeline_space = get_space_base_callable(space_base_name)(**space_kwargs)
    evaluate_pipeline = partial(
        evaluate_pipeline_base, 
        config_path=model_config, 
        nproc_per_node=args.nproc_per_node,
        fidelity_mode="steps",
        lr_mode=args.lr_mode,
        trainset_path=args.trainset_path,  # Pass through for $TMPDIR support
        validset_path=args.validset_path,  # Pass through for $TMPDIR support
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