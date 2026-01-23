import sys
from pathlib import Path


# Add parent directory to path so we can import from plainLM root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import neps
import torch
from neps_nos.utils.model_train_function import train_model, CFG_PATH_46M, CFG_PATH_8M, CFG_PATH_46M_1D, CFG_PATH_8M_1D
from neps_nos.neps_config import get_space_basename_and_kwargs, get_space_base_callable, get_optimizer_name_and_kwargs, get_warmstarter_config, resolve_warmstarter_name
import logging
from pprint import pprint
import time
import numpy as np
import pandas as pd
from functools import partial
from neps_nos.neps_pipeline import evaluate_pipeline_base

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPS NOS Training Pipeline")
    # Example:  python neps_nos/neps_pipeline.py --result_dir neps_runs/testrun1 --neps_space_config NLinesU_f_l_nw --runtime 1
    parser.add_argument("--model_size", type=str, default="8M", choices=["8M_1D", "46M_1D", "8M", "46M"], help="Model size to use for training.")
    parser.add_argument("--runname", type=str, default=None, help="Unique name to use for neps folder and results file.")
    parser.add_argument("--neps_space_config", type=str, required=True, help="The space config to choose.")
    parser.add_argument("--config", type=str, default=None, help="Warmstarter/fixed configuration to use.")
    parser.add_argument("--lr_mode", type=str, default="normal", choices=["normal", "sweep"], help="Learning rate mode: 'normal' (default) or 'sweep'.")
    # parser.add_argument("--fidelity_mode", type=str, default="steps", choices=["steps", "model_size"], help="Fidelity mode to use: 'steps' or 'model_size'.")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node (1=single process, >1=distributed with torchrun).")
    args = parser.parse_args()


    model_configs = {
        "8M_1D": CFG_PATH_8M_1D,
        "46M_1D": CFG_PATH_46M_1D,
        "8M": CFG_PATH_8M,
        "46M": CFG_PATH_46M,
    }
    nos_space = args.neps_space_config
    warmstarter = args.config
    model_config = model_configs[args.model_size]
    
    min_fidelity = 3 if "8M" in args.model_size else 10
    max_fidelity = 12 if "8M" in args.model_size else 50


    space_base_name, space_kwargs = get_space_basename_and_kwargs(args.neps_space_config)
    space_class = get_space_base_callable(space_base_name)
    if "fidelity" in space_kwargs and space_kwargs["fidelity"] is True:
        space_kwargs["fidelity"] = (min_fidelity, max_fidelity)# if args.fidelity_mode == "steps" else (1, 2)
    pipeline_space = space_class(**space_kwargs)

    min_fidelity = 3 if "8M" in model_config else 10
    max_fidelity = 12 if "8M" in model_config else 50

    evaluate_pipeline = partial(
        evaluate_pipeline_base, 
        config_path=model_config, 
        nproc_per_node=args.nproc_per_node,
        fidelity_mode="steps",
        lr_mode=args.lr_mode,
    )
    print("Pipeline space:\n", space_base_name, "\n", space_kwargs)

    if warmstarter.startswith("torch_"):
        warmstart_name = warmstarter.replace("torch_", "")
        match warmstart_name:
            case "Adam":
                warmstart_pipeline = {
                    "optimizer_cls": partial(torch.optim.Adam, betas=(0.9, 0.95)),
                    "learning_rate": 1e-3,
                    "fidelity": max_fidelity,
                }
    else:
        warmstart_name, warmstart_kwargs = resolve_warmstarter_name(warmstarter)
        # Resolve fidelity value: "min" -> min_fidelity, "max" -> max_fidelity, or use the number directly
        if warmstart_kwargs["fidelity"] == "min":
            warmstart_kwargs["fidelity"] = min_fidelity
        elif warmstart_kwargs["fidelity"] == "max":
            warmstart_kwargs["fidelity"] = max_fidelity
        warmstarter_config, warmstart_pipeline = get_warmstarter_config(space_base_name, space_kwargs, pipeline_space, warmstart_name, warmstart_kwargs)
    best_result = float('inf')
    best_lr = 0
    for lr in [0.0001, 0.0009, 0.0003, 0.0027, 0.0243, 0.0081]:
        warmstart_pipeline["learning_rate"] = lr
        print(f"Evaluating with learning rate: {lr}")
        warmstart_result = evaluate_pipeline(**warmstart_pipeline, pipeline_directory=str("neps_runs/test_runs/single_config"))
        print("Result:")
        pprint(warmstart_result)
        if warmstart_result["objective_to_minimize"] < best_result:
            best_result = warmstart_result["objective_to_minimize"]
            best_lr = lr
    print(f"Best learning rate: {best_lr} with objective: {best_result}")