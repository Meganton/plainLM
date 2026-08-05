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
import cloudpickle
import torch
import logging
import socket
import subprocess
from pathlib import Path

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

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

    # Setup per-rank, line-buffered log files so debug output appears in real-time
    pipeline_path = Path(args.pipeline_directory)
    debug_dir = pipeline_path / "debug_logs"
    debug_dir.mkdir(parents=True, exist_ok=True)
    log_path = debug_dir / f"rank_{rank}.log"

    # Open file in text mode with line buffering (buffering=1)
    try:
        logfile = open(log_path, 'a', buffering=1)

        # Create a tee stream that writes to both the original stream and the logfile
        class TeeStream:
            def __init__(self, primary, secondary):
                self.primary = primary
                self.secondary = secondary
            def write(self, data):
                try:
                    self.primary.write(data)
                except Exception:
                    pass
                try:
                    self.secondary.write(data)
                except Exception:
                    pass
            def flush(self):
                try:
                    self.primary.flush()
                except Exception:
                    pass
                try:
                    self.secondary.flush()
                except Exception:
                    pass

        # keep original underlying streams
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = TeeStream(orig_stdout, logfile)
        sys.stderr = TeeStream(orig_stderr, logfile)
    except Exception:
        # fallback: keep original streams
        pass

    # Minimal logger that writes to the same file/stdout
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

    # Emit initial debug information
    try:
        logging.info(f"[Rank {rank}] hostname={socket.gethostname()}")
        logging.info(f"[Rank {rank}] ENV RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
        logging.info(f"[Rank {rank}] torch.cuda.is_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available() and rank < torch.cuda.device_count():
            try:
                dev_name = torch.cuda.get_device_name(rank)
                logging.info(f"[Rank {rank}] CUDA device name: {dev_name}")
            except Exception:
                pass
        # Try to capture a short nvidia-smi summary for debugging (non-fatal)
        try:
            out = subprocess.check_output(["nvidia-smi", "-q", "-d", "MEMORY"] , stderr=subprocess.STDOUT, text=True)
            logging.info(f"[Rank {rank}] nvidia-smi MEMORY summary:\n{out.splitlines()[:20]}")
        except Exception:
            pass
    except Exception:
        pass

    with open(args.opt_path, 'rb') as f:
        # First try: ensure known alias exists for neps_nos_space
        try:
            import importlib
            if 'neps_nos_space' not in sys.modules:
                try:
                    mod = importlib.import_module('neps_nos.neps_nos_space')
                    sys.modules['neps_nos_space'] = mod
                except Exception:
                    # ignore; we'll attempt file-based import below if needed
                    pass

        except Exception:
            pass

        # Try to unpickle; if ModuleNotFoundError occurs for neps_nos_space,
        # attempt to load the module directly from the package file and retry.
        try:
            optimizer_cls = cloudpickle.load(f)
        except ModuleNotFoundError as e:
            missing = getattr(e, 'name', None) or str(e)
            if 'neps_nos_space' in missing:
                try:
                    import importlib.util
                    pkg_root = Path(__file__).parent.parent
                    candidate = pkg_root / 'neps_nos_space.py'
                    if not candidate.exists():
                        # fallback to package file
                        candidate = pkg_root / 'neps_nos' / 'neps_nos_space.py'
                    if candidate.exists():
                        spec = importlib.util.spec_from_file_location('neps_nos_space', str(candidate))
                        module = importlib.util.module_from_spec(spec)
                        sys.modules['neps_nos_space'] = module
                        spec.loader.exec_module(module)
                        # rewind and retry
                        f.seek(0)
                        optimizer_cls = cloudpickle.load(f)
                    else:
                        raise
                except Exception:
                    # log and reraise original
                    import traceback
                    tb = traceback.format_exc()
                    try:
                        sys.stderr.write(f"[Rank {rank}] Failed to load neps_nos_space from file:\n{tb}\n")
                        sys.stderr.flush()
                    except Exception:
                        pass
                    raise
            else:
                raise

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
    except Exception:
        import traceback
        tb = traceback.format_exc()
        # write full traceback to stderr (which is redirected to per-rank logfile)
        try:
            sys.stderr.write(f"[Rank {rank}] Training exception:\n{tb}\n")
            sys.stderr.flush()
        except Exception:
            pass

        # Also write a concise error file next to the per-rank log for easier discovery
        try:
            err_file = debug_dir / f"rank_{rank}_exception.txt"
            err_file.write_text(tb)
        except Exception:
            pass

        # Set loss to inf so NEPS can continue sampling other trials
        if rank == 0:
            try:
                print(f"Training failed (see debug_logs). Setting loss to inf.")
                sys.stdout.flush()
            except Exception:
                pass
        valid_loss = float('inf')
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
