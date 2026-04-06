import sys
from pathlib import Path

# Add parent directory to path so we can import from plainLM root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collections import defaultdict
from functools import partial
import inspect

import torch

import utils
from checkpoint_utils import save_checkpoint
from data import get_dataloaders
from engine import TorchEngine
from models import construct_model
from optim import initialize_scheduler
from torch_utils import destroy_ddp, pytorch_setup
from utils import print_master

CFG_PATH_8M = "config/optsearch/workloads/8M_200BT.yaml"
CFG_PATH_8M_1D = "config/optsearch/workloads/8M_200BT_1device.yaml"
CFG_PATH_46M = "config/optsearch/workloads/46M_1BT.yaml"
CFG_PATH_46M_1D = "config/optsearch/workloads/46M_1BT_1device.yaml"


def _infer_param_depth(model, param_name: str) -> float:
  """Infer parameter depth from standard Transformer naming with safe fallback."""
  if param_name.startswith("embed_tokens"):
    return 0.0
  if param_name.startswith("layers."):
    parts = param_name.split(".")
    if len(parts) > 1 and parts[1].isdigit():
      return float(int(parts[1]) + 1)
  if param_name.startswith("out_norm") or param_name.startswith("lm_head"):
    return float(getattr(model, "n_layers", 0) + 1)
  return 0.0


def _build_nos_param_groups(model):
  """Build param groups with optional metadata used by NOS custom optimizers."""
  param_groups = []
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    param_groups.append(
      {
        "params": [param],
        "depth": _infer_param_depth(model, name),
        "layer_type_attention": 1.0 if ".attn." in name else 0.0,
      }
    )
  return param_groups

def train_model(
  optimizer_cls,
  lr: float,
  pipeline_directory: str,
  weight_decay: float = 0,
  config_path: str = CFG_PATH_8M,
  n_steps: int = -1,
  trainset_path: str = None,  # For $TMPDIR support
  validset_path: str = None,  # For $TMPDIR support
) -> float:
  """
  Train a transformer on Causal Language Modeling.

  Args:
    otpimizer_cls: optimizer class to use for this training run.
    lr: learning rate to use for this training run.
    pipeline_directory: directory to save checkpoints and logs.
    trainset_path: Optional override for training dataset path (for $TMPDIR usage)
    validset_path: Optional override for validation dataset path (for $TMPDIR usage)
  Returns:
    valid_loss: validation loss after training.
  """
  
  # Load a default config as a namedtuple
  cfg, _ = utils.load_config(config_path)

  # Replace some arguments with user-specified ones
  cfg = cfg._replace(out_dir=pipeline_directory)
  cfg = cfg._replace(wandb_dir=pipeline_directory)
  
  # Override dataset paths if provided (for $TMPDIR support)
  if trainset_path is not None:
    cfg = cfg._replace(trainset_path=trainset_path)
  if validset_path is not None:
    cfg = cfg._replace(validset_path=validset_path)

  rank, world_size, device, master_process = pytorch_setup(cfg)

  if master_process:
    utils.maybe_make_dir(cfg)

  if cfg.use_wandb and master_process:
    utils.init_wandb(cfg)

  # Dataset
  trainloader, validloader = get_dataloaders(cfg)

  # Model
  model, _ = construct_model(cfg)

  # Engine
  engine = TorchEngine(model, cfg, device)

  # Optimizer is usually defined by engine, we define it here for ease of use with NOS
  engine.optimizers, engine.schedulers = {}, {}
  nos_param_groups = _build_nos_param_groups(model)
  
  # Build optimizer kwargs based on optimizer type
  # Handle both direct class references and functools.partial objects
  if isinstance(optimizer_cls, partial):
    actual_optimizer = optimizer_cls.func
  else:
    actual_optimizer = optimizer_cls
  
  # Inspect the __init__ signature to determine which parameters to include
  try:
    sig = inspect.signature(actual_optimizer.__init__)
    optimizer_params = set(sig.parameters.keys())
  except (TypeError, ValueError):
    optimizer_params = set()
  
  # Build optimizer kwargs only with parameters the optimizer accepts
  optimizer_kwargs = {"lr": lr}
  
  if "weight_decay" in optimizer_params:
    optimizer_kwargs["weight_decay"] = weight_decay
  
  if "eps" in optimizer_params:
    optimizer_kwargs["eps"] = getattr(cfg, "eps", 1e-8)
  
  if "betas" in optimizer_params:
    optimizer_kwargs["betas"] = (cfg.beta1, cfg.beta2)
  
  engine.optimizers['nos'] = optimizer_cls(nos_param_groups, **optimizer_kwargs)
  engine.schedulers['nos'] = initialize_scheduler(engine.optimizers['nos'], cfg)

  # If we are just cooling down, we set budget = resume + cooldown
  steps_budget = cfg.steps_budget if cfg.scheduler != "linear_cooldown" else cfg.resume_step + engine.scheduler.cooldown_steps

  if n_steps > 0:
    steps_budget = n_steps*100

  for optimizer in engine.optimizers.values():
    setattr(optimizer, "_nos_total_epochs", float(steps_budget))

  micro_step_budget = steps_budget * cfg.grad_accumulation_steps
  if micro_step_budget > len(trainloader):
    raise ValueError("trainloader too short!")

  # Start the dataloader from the correct micro-batch
  step_start = cfg.resume_step if cfg.resume else 0
  micro_step_start = step_start * cfg.grad_accumulation_steps
  print_master(
    f"=== Start Training from step: {step_start}/{steps_budget}, micro_step: {micro_step_start}/{micro_step_budget} ===",
  )

  # Bookkeeping
  metrics = defaultdict(list)

  # Training
  for micro_step, micro_batch in enumerate(trainloader, micro_step_start + 1):
    step = micro_step // cfg.grad_accumulation_steps
    is_step = micro_step % cfg.grad_accumulation_steps == 0
    if step > steps_budget and is_step:
      break

    # Train
    train_loss = engine.step(micro_batch)

    # Log
    if master_process and step % cfg.log_every_steps == 0 and is_step:
      metrics["step"].append(step)
      metrics["micro_step"].append(micro_step)
      metrics["tokens"].append(step * cfg.seq_len * cfg.micro_batch_size * world_size)
      metrics["train/loss"].append(train_loss.item())
      for n, optim in engine.optimizers.items():
        metrics[f"{n}_lr"].append(optim.param_groups[0]["lr"])
      utils.log(cfg, metrics)
    
    # Checkpoint
    if (
      cfg.save_intermediate_checkpoints
      and step % cfg.save_every_steps == 0
      and is_step
    ):
      save_checkpoint(step, model, engine, cfg, metrics, rank)

  # Eval
  if getattr(cfg, 'eval_when_finished', True):
    print_master("Evaluating on validation set")
    valid_loss = engine.eval(validloader)
    if master_process:
      metrics["valid/loss"] = valid_loss # no append here, eval at the end only
      utils.log(cfg, metrics)

  # End of training: log and save checkpoint
  print_master("=== Training Completed! ===")
  if cfg.save_last_checkpoint:
    save_checkpoint(step, model, engine, cfg, metrics, rank)

  # DDP slaughtering
  destroy_ddp()

  return valid_loss
