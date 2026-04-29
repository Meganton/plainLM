from pathlib import Path
import sys
import torch
import json

repo_root = Path.cwd()
if not (repo_root / "neps_nos").exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))
from neps_nos.neps_pipeline import process_neps_status

process_neps_status(Path("data/horse/ws/ange731i-anton/plainLM/neps_runs/paper_8/neps/nlines_re_8m/seed_0"))