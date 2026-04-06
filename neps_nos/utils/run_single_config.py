import sys
from pathlib import Path


# Add parent directory to path so we can import from plainLM root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import neps
import torch
from neps_nos.utils.model_train_function import train_model, CFG_PATH_46M, CFG_PATH_8M
from neps_nos.neps_config import get_space_basename_and_kwargs, get_space_base_callable, get_optimizer_name_and_kwargs, get_warmstarter_config, resolve_warmstarter_name
import logging
from pprint import pprint
import time
import numpy as np
import pandas as pd
from functools import partial
from itertools import product as cartesian_product
from neps_nos.neps_pipeline import evaluate_pipeline_base

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPS NOS Training Pipeline")
    # Example:  python neps_nos/neps_pipeline.py --result_dir neps_runs/testrun1 --neps_space_config NLinesU_f_l_nw --runtime 1
    parser.add_argument("--model_size", type=str, default="8M", choices=["8M", "46M"], help="Model size to use for training.")
    parser.add_argument("--runname", type=str, default=None, help="Unique name to use for neps folder and results file.")
    parser.add_argument("--neps_space_config", type=str, required=True, help="The space config to choose.")
    parser.add_argument("--config", type=str, default=None, help="Warmstarter/fixed configuration to use.")
    parser.add_argument("--nproc_per_node", type=int, default=4, help="Number of GPU processes per node.")
    args = parser.parse_args()


    model_configs = {
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
            case "AdamW":
                warmstart_pipeline = {
                    "optimizer_cls": partial(torch.optim.AdamW, betas=(0.9, 0.95)),
                    "learning_rate": 1e-3,
                    "weight_decay": 0.01,
                    "fidelity": max_fidelity,
                }
            case "SGD":
                warmstart_pipeline = {
                    "optimizer_cls": torch.optim.SGD,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "fidelity": max_fidelity,
                }
            case "Adagrad":
                warmstart_pipeline = {
                    "optimizer_cls": torch.optim.Adagrad,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "fidelity": max_fidelity,
                }
            case "NAdam":
                warmstart_pipeline = {
                    "optimizer_cls": torch.optim.NAdam,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "fidelity": max_fidelity,
                }
            case "RMSprop":
                warmstart_pipeline = {
                    "optimizer_cls": torch.optim.RMSprop,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "fidelity": max_fidelity,
                }
            case "SGD_Momentum":
                warmstart_pipeline = {
                    "optimizer_cls": partial(torch.optim.SGD, momentum=0.9),
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "fidelity": max_fidelity,
                }
            case "SGD_Nesterov":
                warmstart_pipeline = {
                    "optimizer_cls": partial(torch.optim.SGD, momentum=0.9, nesterov=True),
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
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
    best_time = 0
    if hasattr(pipeline_space, "weight_decay"):
        wds = np.logspace(-3, -1, num=4)
    else:
        wds = [0.0]
    for wd,lr in cartesian_product(wds, [0.0001, 0.0009, 0.0003, 0.0027, 0.0243, 0.0081]):
        warmstart_pipeline["learning_rate"] = lr
        warmstart_pipeline["weight_decay"] = wd
        print(f"Evaluating with learning rate: {lr} and weight decay: {wd}")
        import time
        start_time = time.time()
        warmstart_result = evaluate_pipeline(**warmstart_pipeline, pipeline_directory=str(f"neps_runs/test_runs/single_config/lr{lr}_wd{wd}"))
        time_taken = time.time() - start_time
        print(f"Time taken for evaluation: {time_taken:.2f} seconds")
        print("Result:")
        pprint(warmstart_result)
        if warmstart_result["objective_to_minimize"] < best_result:
            best_result = warmstart_result["objective_to_minimize"]
            best_lr = lr
            best_time = round(time_taken/60, 2)
    print(f"Best learning rate: {best_lr} with objective: {best_result}")

    # Save results as json in neps_runs/single_configs/{runname}.json
    summary_file = Path(f"neps_runs/single_configs/{args.runname}.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to {summary_file}.")
    with open(summary_file, "w") as f:
        import json
        json.dump({
            "loss_history": [best_result, best_result],
            "cost_history": [best_time, best_time],
            "incumbent_history": [best_result, best_result],
            "cumulated_cost_history": [best_time, best_time*2],
            "fidelity_history": [max_fidelity, max_fidelity],
            "cumulated_fidelity_history": [max_fidelity, max_fidelity*2],
        }, f, indent=2)
    print("Results saved.")