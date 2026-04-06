import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import neps
from neps_nos import neps_nos_space
import torch
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")
from pathlib import Path

dic = {}

folder = "46_LI_space"
run = "li1_rand_01"
# for r in [1,3,5,8]:
for run in [f"sota_li1_rand_01_ae",f"sota_li1_rand_03_ae",f"sota_li1_rand_01_ae_mul",f"sota_li1_rand_01_nl"]:
    for seed in range(5):
        directory = Path(f"neps_runs/{folder}/neps/{run}/seed_{seed}")
        print(f"Directory: {directory}")
        try:
            full, _ = neps.status(directory)
            best_idx = full['objective_to_minimize'].idxmin()
        except (FileNotFoundError, KeyError) as e:
            print(f"  [skipping: {e}]")
            continue
        best_run_id = str(best_idx)
        min_value = full.loc[best_idx, 'objective_to_minimize']
        print(f"Best run ID: {best_run_id}\nwith objective value: {min_value}")
        pipeline = neps.load_config(config_path=directory / "configs" / f"config_{best_run_id}")
        x = torch.zeros(1)
        optimizer = pipeline["optimizer_cls"]([x], pipeline["learning_rate"])
        print("Best optimizer sampled:\n", optimizer)
        dic[f"{run}_seed_{seed}"] = float(min_value)
from pprint import pprint
pprint(dic)