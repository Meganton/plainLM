import sys
from pathlib import Path


# Add parent directory to path so we can import from plainLM root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import neps

from neps_nos.neps_config import get_space_basename_and_kwargs, get_space_base_callable, SPACES
from pprint import pprint
import torch


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--nos_space", type=str, required=True, choices=SPACES.keys(), help="Name of the NOS space configuration to create warmstarting config for.")
    args = arg_parser.parse_args()

    space_base_name, space_kwargs = get_space_basename_and_kwargs(args.nos_space)
    print(f"Creating warmstarting config for NOS space '{args.nos_space}' with base '{space_base_name}' and kwargs {space_kwargs}")
    space_class = get_space_base_callable(space_base_name)
    pipeline_space = space_class(**space_kwargs)
    config_dict, pipeline = neps.create_config(pipeline_space)
    print("Generated NEPS configuration:")
    pprint(config_dict)
    opt_class = pipeline["optimizer_cls"]
    x = torch.zeros(1)
    optimizer = opt_class([x])
    print(f"Created optimizer:\n{optimizer}")
