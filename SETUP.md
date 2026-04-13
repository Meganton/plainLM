# plainLM — Neural Optimizer Search (NOS) Setup Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dependencies](#3-dependencies)
4. [Model Configurations](#4-model-configurations)
5. [NOS Search Spaces](#5-nos-search-spaces)
6. [NePS Optimizer Strategies](#6-neps-optimizer-strategies)
7. [Training Pipeline (end-to-end)](#7-training-pipeline-end-to-end)
8. [Config YAML Reference](#8-config-yaml-reference)
9. [Warmstarting](#9-warmstarting)
10. [SLURM Cluster Scripts](#10-slurm-cluster-scripts)
11. [Utilities & Analysis Tools](#11-utilities--analysis-tools)
12. [Running the Pipeline](#12-running-the-pipeline)

---

## 1. Project Overview

**plainLM** is a research codebase for **Neural Optimizer Search (NOS)** on language models. The goal is to use **NePS** (Neural Pipeline Search) to automatically discover novel optimizer update rules that outperform hand-designed optimizers (AdamW, Muon, etc.) on transformer LM pretraining.

The search is conducted hierarchically using multiple fidelity levels (different numbers of training steps) across two model scales:

| Model | Parameters (non-embedding) | Approx. training time per trial |
|-------|---------------------------|----------------------------------|
| 8M    | ~1.57M                    | ~5 min                           |
| 46M   | ~20.5M                    | ~20 min                          |

The dataset is **FineWeb-Edu (100B tokens)** tokenized with the GPT-NeoX tokenizer.

---

## 2. Repository Structure

```
plainLM2/
├── train.py                        # Standard training entrypoint (non-NOS)
├── utils.py                        # Shared utilities (config loading, wandb, logging)
├── torch_utils.py                  # DDP setup, seed, device management
├── checkpoint_utils.py             # Saving and loading checkpoints
├── pyproject.toml                  # Package metadata
│
├── config/                         # YAML training configurations
│   ├── 70M_10BT.yaml               # Standard training config (non-NOS)
│   ├── dev/debug.yaml              # Quick local debug config
│   ├── oneB/adamw/                 # 1B-token standard training sweep
│   └── optsearch/
│       ├── extras/                 # Additional model-size configs (8M, 18M, 46M, 123M)
│       └── workloads/              # NOS-specific workload configs (see §4 & §8)
│           ├── 8M_200BT.yaml       # 8M model, 4-GPU, ~200BT
│           ├── 8M_200BT_1device.yaml   # 8M model, single GPU
│           ├── 46M_1BT.yaml        # 46M model, 4-GPU
│           └── 46M_1BT_1device.yaml    # 46M model, single GPU
│
├── models/                         # Transformer model definitions
│   ├── transformer.py              # Transformer++ (LLaMA-style, RMSNorm, RoPE, GLU)
│   ├── construct.py                # Model factory + param group splitting
│   ├── components.py               # GLU, MLP, RMSNorm
│   ├── embeddings.py               # RoPE (rotary positional embeddings)
│   └── param_count.py
│
├── data/                           # Data loading
│   ├── dataloaders.py              # DataLoader factory with multiple sampler options
│   ├── datasamplers.py             # Stateful samplers for deterministic resuming
│   └── datasets/                   # Dataset preparation scripts
│
├── engine/
│   └── engine.py                   # TorchEngine: wraps model + optimizer + scheduler
│
├── optim/                          # Standard optimizer implementations
│   ├── init_optim.py               # Optimizer factory (AdamW, SGD, ZeRO1AdamW, Muon)
│   ├── lr_schedule.py              # WSD, WarmupCosine, WarmupConstant, LinearCooldown
│   ├── zero1adamw.py               # ZeRO-1 sharded AdamW
│   ├── muon.py                     # Muon optimizer (MuonVanilla, MuonDP)
│   └── signSGD.py                  # signSGD / Signum (not used in NOS pipeline)
│
├── neps_nos/                       # NePS NOS integration (main search code)
│   ├── neps_pipeline.py            # Main NePS entrypoint (run as __main__)
│   ├── neps_nos_space.py           # NOS search space class definitions
│   ├── neps_config.py              # Registry of spaces, optimizers, warmstarters
│   ├── test.ipynb                  # Interactive dev/debugging notebook
│   ├── old/
│   │   └── fidelity_best_config.py # Legacy: superseded by neps_pipeline.py
│   ├── tests/                      # Test utilities and validation scripts
│   └── utils/
│       ├── model_train_function.py # train_model() — the NePS evaluation function
│       ├── torchrun_worker.py      # torchrun entry point: deserializes optimizer, calls train_model()
│       ├── run_single_config.py    # Run a single warmstarter/fixed config
│       ├── extract_best_function.py# Post-hoc: extract and print the best found optimizer
│       └── create_warmstarting_configs.py  # Helper to generate warmstart NEPS configs
│
├── neps_runs/                      # Output directory for all NePS runs
│   ├── _log/                       # SLURM logs synced from $TMPDIR
│   ├── 46_LI_space/                # Results for 46M NLinesU searches
│   ├── ...
│
├── neps_cluster_scripts/           # SLURM job submission scripts
│   ├── sota/                       # 46M SOTA optimizer experiments
│   ├── sota_8/                     # 8M SOTA optimizer experiments
│   ├── 46/                         # 46M model search configurations
│   ├── mutation_study/             # Ablation over mutation_mode
│   ├── single_conf_adam.sh         # Single-config Adam baseline runs
│   ├── single_conf_adam_wd.sh      # Single-config Adam with weight decay
│   ├── single_conf_torch_adam.sh   # PyTorch Adam (8M and 46M)
│   ├── single_conf_torch_adamw.sh  # PyTorch AdamW (8M and 46M)
│   ├── single_conf_torch_nadam.sh  # PyTorch NAdam (8M and 46M)
│   ├── single_conf_torch_rmsprop.sh# PyTorch RMSprop (8M and 46M)
│   ├── single_conf_torch_adagrad.sh# PyTorch Adagrad (8M and 46M)
│   ├── single_conf_torch_sgd.sh    # PyTorch SGD (8M and 46M)
│   ├── single_conf_torch_sgd_momentum.sh  # PyTorch SGD with momentum
│   ├── single_conf_torch_sgd_nesterov.sh # PyTorch SGD with Nesterov momentum
│   ├── short_8.sh                  # Quick 8M search test
│   ├── short_46.sh                 # Quick 46M search test
│   ├── short_sweep.sh              # Quick parameter sweep test
│   ├── tests/                      # Test job scripts
│   ├── old/                        # Legacy job scripts
│   └── utils/
│       ├── neps_tmpdir_wrapper.sh  # $TMPDIR optimization wrapper (sourced by job scripts)
│       └── copy_dataset_to_tmpdir.sh
│
├── neps_plots/                     # Plotting utilities for NePS results
│   ├── plotting.py                 # Core matplotlib plotting utilities
│   ├── plots/                      # Generated plot outputs
│   └── scripts/
│       ├── 8/                      # 8M model comparison plots
│       ├── 46/                     # 46M model comparison plots
│       └── extension/              # Extended plotting scripts (sota comparisons, ablations)
│
├── copy_results.sh                 # Utility script for copying results between runs
│
├── cluster/                        # Legacy SLURM scripts for standard train.py runs
│   ├── single_gpu/                 # Single-GPU job scripts
│   ├── multi_gpu/                  # Multi-GPU (torchrun) job scripts
│   └── multi_node/                 # Multi-node job scripts
│
└── synthetic/                      # Self-contained NOS experiments on synthetic benchmarks
    ├── nos_space.py                # NOS space definition (standalone, mirrors neps_nos_space.py)
    ├── nos_synthetic.py            # NOS on synthetic optimization problems (sphere, etc.)
    ├── nos_ml_benchmarks.py        # NOS on small ML benchmarks
    ├── plot_*.py                   # Plotting scripts for synthetic results
    └── README_ML_BENCHMARKS.md
```

---

## 3. Dependencies

| Package | Role |
|---------|------|
| `torch` | Model training, DDP, `torch.compile` |
| `neps` | Neural Pipeline Search — the HPO/NOS framework |
| `dill` | Serialization of dynamic optimizer classes for passing to spawned DDP worker processes |
| `datasets` (HuggingFace) | Loading tokenized arrow dataset from disk |
| `wandb` | Experiment tracking (optional, disabled in NOS runs) |
| `yaml` | Config file parsing |
| `absl` | CLI flags for `train.py` |
| `numpy`, `pandas` | Post-processing NePS results |

Python environment is managed in `.venv` and activated in every script via `source ./.venv/bin/activate`.

---

## 4. Model Configurations

### Workload Files (used by NOS)

Located in `config/optsearch/workloads/`. The pipeline uses the 4-GPU DDP configs (`8M_200BT.yaml`, `46M_1BT.yaml`). Single-device variants (`*_1device.yaml`) exist but are no longer used by the NOS pipeline.

#### 8M Model (`8M_200BT.yaml` / `8M_200BT_1device.yaml`)

| Parameter | 4-GPU | 1-GPU |
|-----------|-------|-------|
| `d_model` | 128 | 128 |
| `n_layers` | 6 | 6 |
| `n_heads` | 4 | 4 |
| `mlp_class` | `glu` | `glu` |
| `expand` | `8/3` | `8/3` |
| `steps_budget` | 1250 | 1250 |
| `micro_batch_size` | 48 | 48 |
| `grad_accumulation_steps` | 1 | 4 |
| `optim` (default) | `zero1adamw` | `adamw` |
| NOS fidelity range | 3–12 | — |

- Non-embedding parameters: **~1.57M**
- Training time per full run: **~5 min**

#### 46M Model (`46M_1BT.yaml` / `46M_1BT_1device.yaml`)

| Parameter | 4-GPU | 1-GPU |
|-----------|-------|-------|
| `d_model` | 512 | 512 |
| `n_layers` | 6 | 6 |
| `n_heads` | 8 | 8 |
| `mlp_class` | `glu` | `glu` |
| `expand` | `8/3` | `8/3` |
| `steps_budget` | 5000 | 5000 |
| `micro_batch_size` | 48 | 48 |
| `grad_accumulation_steps` | 1 | 4 |
| `optim` (default) | `zero1adamw` | `adamw` |
| NOS fidelity range | 10–50 | — |

- Non-embedding parameters: **~20.5M**
- Training time per full run: **~20 min**

### Model Architecture (Transformer++)

A LLaMA-style decoder-only transformer defined in `models/transformer.py`:

- **Attention**: Multi-head self-attention with RoPE embeddings and `scaled_dot_product_attention`
- **FFN**: Gated Linear Unit (GLU) by default
- **Normalization**: RMSNorm (pre-norm residual)
- **Embeddings**: Optional tied token embeddings (`tie_embeddings: True`)
- **Compilation**: `torch.compile` enabled by default

---

## 5. NOS Search Spaces

Defined in `neps_nos/neps_nos_space.py`. All spaces are subclasses of `neps.PipelineSpace`. The space registry is in `neps_nos/neps_config.py` under `SPACES` and `SPACE_BASES`.

### Space Registry (`SPACES` in `neps_config.py`)

Each entry is `"config_name": ("base_class_key", {kwargs})`:

| Config Name | Base Class | `fidelity` | `learning_rate` | `weight_decay` |
|---|---|---|---|---|
| `NLinesU_nf_nl_nw` | `NLinesU_1_10` | — | — | — |
| `NLinesU_nf_l_nw` | `NLinesU_1_10` | — | (1e-5, 1e-1) log | — |
| `NLinesU_f_l_nw` | `NLinesU_1_10` | (auto) | (1e-5, 1e-1) log | — |
| `NLinesU_f_nl_nw` | `NLinesU_1_10` | (auto) | — | — |
| `AdamExtend_nf_l_nw` | `AdamExtend_1_6` | — | (1e-5, 1e-1) log | — |
| `AdamExtend_f_l_nw` | `AdamExtend_1_6` | (auto) | (1e-5, 1e-1) log | — |
| `AdamExtend_f_l_w` | `AdamExtend_1_6` | (auto) | (1e-5, 1e-1) log | (0.0, 1e-1) log |
| `AdamExtendMul_f_l_nw` | `AdamExtend_1_6_mul` | (auto) | (1e-5, 1e-1) log | — |
| `PremadeModules_nf_l_nw` | `PremadeModules` | — | (1e-5, 1e-1) log | — |
| `PremadeModules_f_l_nw` | `PremadeModules` | (auto) | (1e-5, 1e-1) log | — |
| `SmallAdam_f_nl_nw` | `SmallAdam` | (auto) | — | — |

> **Naming convention**: `nf` = no fidelity, `f` = fidelity; `l` = learning rate searched, `nl` = no LR; `nw` = no weight decay searched, `w` = weight decay searched.

> **`fidelity: True`** in config kwargs gets resolved to the model-size-dependent range `(min_fidelity, max_fidelity)` at runtime:
> - 8M: `(3, 12)`
> - 46M: `(10, 50)`

The **fidelity value** passed to `train_model()` is multiplied by 100 to get `steps_budget` (e.g., fidelity=12 → 1200 steps).

---

### Space Class Descriptions

#### `NOSSpaceNLinesU` (key: `NLinesU_1_10`)

The **primary search space** for this project. It defines optimizers as a sequence of *N* update lines (variable length 1–10) plus one fixed last line that always computes the update direction `u`. The parameter update is `w ← w - lr * u`.

**Variables available inside the optimizer update rules:**

| Variable | Meaning |
|----------|---------|
| `g` | Current gradient (+ weight decay term) |
| `w` | Current parameter value |
| `u` | Update direction (written by last line) |
| `v1`, `v2` | Auxiliary state variables (initialized to 0.1) |
| `m`, `v` | Extra state (moment-like) variables |
| `b1`, `b2` | Beta hyperparameters (default 0.9, 0.999) |

**Operations available for intermediate lines:**
- `scale_by_constant(c)` — multiply by constant `c ∈ {10, 1, 0, 0.1, 0.01, 0.9, 0.99}`
- `clamp_by_constant(c)` — clamp to `[-c, c]`
- `torch.reciprocal`, `torch.sqrt`, `torch.sign`
- Binary ops: `torch.mul`, `torch.add`, `torch.div`, `torch.sub`
- `interpolate(c)` — `lerp` by constant

Each line has the form `(target_variable, expression)`.

#### `PremadeBlocks` (key: `PremadeModules`)

A higher-level space using pre-built moment building blocks: `first_moment`, `second_moment`, `bias_correction`, `clamp_gradient`. Lines are drawn from these blocks — NePS searches which blocks to include and their order.

#### `AdamWExtend` (keys: `AdamExtend_1_6`, `AdamExtend_1_6_mul`)

Extends AdamW by searching for an additional correction term. The `term_mode` arg controls whether the extra term is added (`"add"`) or multiplied (`"mul"`) to the Adam update.

#### `PremadeModules`

A modular space with high-level configuration choices:
- **Momentum**: `none`, `standard`, `heavy_ball`, `nesterov` (each with optional bias correction)
- **Second moment**: `none`, `ema_squared`, `accumulate`, `centered`
- Beta hyperparameters searched as continuous floats

#### `SmallAdamMul`

A narrower, Adam-centric space (1 line + last line) using multiplicative mode. Used for targeted searches around Adam.

---

## 6. NePS Optimizer Strategies

Defined in `neps_nos/neps_config.py` under `OPTIMIZERS`. The key is the string passed as `--neps_optimizer`.

| Key | NePS Algorithm | Notable kwargs |
|-----|---------------|----------------|
| `RE` | `neps_regularized_evolution` | `ignore_fidelity: "highest_fidelity"` |
| `RS` | `neps_random_search` | `ignore_fidelity: "highest_fidelity"` |
| `GridSearch` | `neps_exhaustive_search` | `sampling_density: 3` |
| `PB-like` | `neps_priorband` | `base: hyperband, eta: 2` |
| `LI1_r0.1` | `neps_local_and_incumbent` | `inc_takeover_mode: 1, random_ratio: 0.1` |
| `LI1_r0.3` | `neps_local_and_incumbent` | `inc_takeover_mode: 1, random_ratio: 0.3` |
| `LI1_r0.5` | `neps_local_and_incumbent` | `inc_takeover_mode: 1, random_ratio: 0.5` |
| `LI1_rand0.1` | `neps_local_and_incumbent` | `inc_takeover_mode: 1, mutation_mode: [random, 0.1]` |
| `LI1_rand0.3` | `neps_local_and_incumbent` | `inc_takeover_mode: 1, mutation_mode: [random, 0.3]` |
| `LI1_rand0.5` | `neps_local_and_incumbent` | `inc_takeover_mode: 1, mutation_mode: [random, 0.5]` |
| `LI1_rand0.8` | `neps_local_and_incumbent` | `inc_takeover_mode: 1, mutation_mode: [random, 0.8]` |
| `LI1_ratio0.1` … `LI1_ratio0.8` | `neps_local_and_incumbent` | `mutation_mode: [ratio, 0.X]` |
| `LI1_fixed1` … `LI1_fixed8` | `neps_local_and_incumbent` | `mutation_mode: [fixed, N]` (mutate exactly N positions) |
| `LI2_r0.3`, `LI3_r0.1`, etc. | `neps_local_and_incumbent` | different `inc_takeover_mode` values (0–3) |

**Key NePS parameters:**
- `inc_takeover_mode`: controls how the incumbent is exploited; `1` = local mutations around the best-so-far config
- `mutation_mode`: `[random, p]` = mutate each position with probability `p`; `[ratio, p]` = mutate `p*len` positions; `[fixed, n]` = always mutate exactly `n` positions
- `base: hyperband` + `eta: 2` — multi-fidelity using Hyperband with halving factor 2

---

## 7. Training Pipeline (end-to-end)

The overall flow from SLURM job → model training:

```
SLURM Job Script (*.sh)
  └─ source neps_tmpdir_wrapper.sh
       └─ run_neps_with_tmpdir [args]
            ├─ Copy dataset to $TMPDIR  (code + NePS files stay in HOME)
            ├─ cd $project_root (HOME)
            └─ python neps_nos/neps_pipeline.py [args]
                  │
                  ├─ setup_result_directories()
                  ├─ Resolve space and optimizer from registry
                  ├─ (optional) Warmstart via neps.import_trials()
                  │
                  └─ neps.run(evaluate_pipeline, ...)
                        │
                        └─ evaluate_pipeline_base(optimizer_cls, lr, wd, fidelity, ...)
                              │
                              └─ run_training(...)
                                    │
                                    ├─ Serialize optimizer_cls to pipeline_directory/optimizer.dill via dill
                                    ├─ _find_free_port() — find available TCP port
                                    └─ subprocess.run([torchrun, --standalone, --nproc_per_node=N,
                                          │             torchrun_worker.py, --opt_path, ...])
                                          │  (N workers launched by torchrun, one per GPU)
                                          └─ torchrun_worker.py (each worker)
                                                │
                                                ├─ Read LOCAL_RANK from env; suppress output for non-rank-0
                                                ├─ dill.load(opt_path) → optimizer_cls
                                                └─ train_model(optimizer_cls, lr, wd, ...)
                                                      │
                                                      ├─ Load YAML config
                                                      ├─ pytorch_setup() — init DDP
                                                      ├─ get_dataloaders()
                                                      ├─ construct_model()
                                                      ├─ TorchEngine(model, cfg, device)
                                                      ├─ Instantiate searched optimizer_cls
                                                      ├─ initialize_scheduler()
                                                      ├─ Training loop
                                                      └─ engine.eval() → valid_loss
                                                            └─ Rank 0 writes valid_loss.json
```

After each evaluation:
- Rank 0 writes `valid_loss.json` in `pipeline_directory`
- `run_training()` reads it and returns `{"objective_to_minimize": valid_loss, "cost": minutes}`
- NePS picks next config and the loop repeats

### Result Storage

Without `--runname`:
```
<result_dir>/neps/<neps_optimizer>[_<warmstarter>]/seed_<seed>/   # NePS trial data
<result_dir>/results/<neps_optimizer>_<seed>.json                   # Summary JSON
```

With `--runname` (SOTA scripts):
```
<result_dir>/neps/<runname>/seed_<seed>/    # NePS trial data
<result_dir>/results/<runname>_<seed>.json  # Summary JSON
```

The summary JSON contains:
- `loss_history` — validation loss of every trial
- `cost_history` — wall-clock cost (minutes) of every trial
- `incumbent_history` — running best validation loss
- `cumulated_cost_history` — cumulative cost
- `fidelity_history` / `cumulated_fidelity_history` — fidelity per trial

---

## 8. Config YAML Reference

All fields used across the workload YAML files:

### Dataset

| Field | Type | Description |
|-------|------|-------------|
| `trainset_path` | str | Path to HuggingFace arrow dataset (train split) |
| `validset_path` | str | Path to HuggingFace arrow dataset (valid split) |
| `vocab_size` | int | Vocabulary size (use 50280 for GPT-NeoX tokenizer) |
| `seq_len` | int | Sequence length (1024) |
| `sampler` | str | `stateful_random`, `random`, `sequential`, `stateful_sequential` |
| `sampler_seed` | int | Seed for the dataset sampler |
| `num_workers` | int | DataLoader workers |
| `valid_tokens` | int | Number of tokens for validation (~2M) |

### Evaluation

| Field | Type | Description |
|-------|------|-------------|
| `eval` | bool | Whether to run in-training evaluation |
| `eval_every_steps` | int | Steps between in-training evals (set very high to disable) |
| `eval_when_finished` | bool | Run final eval at end of training |

### Model

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | `transformer` or `pythia-*` |
| `d_model` | int | Hidden dimension |
| `mlp_class` | str | `glu`, `mlp`, `mlp_relu_sq` |
| `expand` | str | FFN expansion ratio as fraction string (e.g., `"8/3"`) |
| `n_layers` | int | Number of transformer blocks |
| `n_heads` | int | Number of attention heads |
| `rms_norm` | bool | Use RMSNorm |
| `tie_embeddings` | bool | Tie token embeddings with LM head |
| `torch_compile` | bool | Enable `torch.compile` |
| `allow_tf32` | bool | Enable TF32 matmul on Ampere+ GPUs |

### Training

| Field | Type | Description |
|-------|------|-------------|
| `steps_budget` | int | Total optimizer steps to train |
| `micro_batch_size` | int | Per-device batch size |
| `grad_accumulation_steps` | int | Gradient accumulation |
| `dtype` | str | `bfloat16`, `float32` |
| `grad_clip` | float | Gradient norm clipping threshold |
| `seed` | int | Global random seed |
| `deterministic` | bool | Enable deterministic CUDA ops |

### Optimizer (standard runs, overridden by NOS)

| Field | Type | Description |
|-------|------|-------------|
| `optim` | str | `adamw`, `zero1adamw`, `sgd`, `muon_vanilla`, `muon_dp` |
| `lr` | float | Learning rate |
| `weight_decay` | float | Weight decay |
| `beta1`, `beta2` | float | Adam β₁, β₂ |
| `eps` | float | Adam ε |
| `grad_clip` | float | Gradient clipping |
| `fused_optim` | bool | Use fused AdamW kernel |
| `adamc_wd` | bool | AdamC weight decay variant |

### LR Schedule

| Field | Type | Description |
|-------|------|-------------|
| `scheduler` | str | `wsd`, `cosine`, `constant`, `linear_cooldown` |
| `warmup_steps` | float/int | Steps or fraction of total steps for warmup |
| `cooldown_steps` | float/int | Steps or fraction for cooldown (WSD) |
| `lr_start` | float | Starting LR (pre-warmup) |
| `lr_end` | float | Final LR after cooldown |
| `lr_end_pct` | float/null | LR end as percentage of max LR |

**Schedule types:**
- `wsd` — Warmup + Stable + Decay (trapezoidal/WSD)
- `cosine` — Linear warmup then cosine decay
- `constant` — Linear warmup then constant
- `linear_cooldown` — Resumes from checkpoint for linear cooldown only

### Checkpointing

| Field | Type | Description |
|-------|------|-------------|
| `save_last_checkpoint` | bool | Save at training completion |
| `save_intermediate_checkpoints` | bool | Save periodically during training |
| `save_every_steps` | int | Checkpoint interval |
| `exp_name` | str | Experiment subdirectory name |
| `out_dir` | str | Output root directory (overridden by pipeline_directory in NOS) |
| `over_write` | bool | Delete existing experiment dir if it exists |

### Resuming

| Field | Type | Description |
|-------|------|-------------|
| `resume` | bool | Resume from checkpoint |
| `resume_step` | int/null | Specific step to resume from (null = latest) |
| `resume_exp_name` | str/null | Resume from different experiment name |

### Logging

| Field | Type | Description |
|-------|------|-------------|
| `log_every_steps` | int | Logging interval |
| `print_progress` | bool | Print metrics to stdout |
| `use_wandb` | bool | Enable W&B logging |
| `wandb_project` | str | W&B project name |
| `wandb_run_name` | str | W&B run name |
| `wandb_dir` | str | W&B local dir (overridden in NOS) |

---

## 9. Warmstarting

NePS can be initialized with a known good config to guide early search. This is implemented via `neps.import_trials()`.

### Warmstarter Registry (`WARMSTART_PARAMETERS` in `neps_config.py`)

| Key | Optimizer | Default fidelity | LR | WD |
|-----|-----------|-------------------|----|-----|
| `SGDM_inter` | SGD with momentum (interpolation update) | max_fidelity | 1e-3 | 0.0 |
| `SGDM_add` | SGD with momentum (additive update) | max_fidelity | 1e-3 | 0.0 |
| `Adam` | Adam-like optimizer | max_fidelity | 1e-3 | 0.0 |

### How It Works

1. The warmstarter name is resolved to a NEPS config dict via `WARMSTART_CONFIGS` (hand-crafted SAMPLING__ paths that pin discrete choices to specific categorical indices).
2. `get_warmstarter_config()` evaluates the config through the pipeline space's sampler to produce a `pipeline_dict` containing an instantiable `optimizer_cls`.
3. `evaluate_pipeline(**warmstart_pipeline)` trains the warmstart optimizer and records the result.
4. `neps.import_trials()` inserts this trial into the NePS directory before `neps.run()`, seeding the search with a known point.

### `run_single_config.py`

Used manually to evaluate a single config (e.g., sweep over LRs for a fixed warmstarter):

```bash
python neps_nos/utils/run_single_config.py \
  --neps_space_config NLinesU_f_l_nw \
  --config SGDM_inter \
  --model_size 46M \
  --nproc_per_node 4
```

It sweeps over LRs `[0.0001, 0.0009, 0.0003, 0.0027, 0.0081, 0.0243]`. If the space has a `weight_decay` parameter (e.g. `AdamExtend_f_l_w`), it also sweeps over WDs `np.logspace(-3, -1, num=4)` — all combinations are evaluated and the best is reported.

---

## 10. SLURM Cluster Scripts

### Resource Profile (typical SOTA runs)

```bash
#SBATCH --account=p_deeplearning
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=capella
#SBATCH --time=48:00:00
#SBATCH --array=0-4          # 5 independent seeds
```

Each array task = one independent NePS seed run.

### Job Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `neps_optimizer` | `LI1_rand0.1` | NePS search strategy key (from `OPTIMIZERS` registry) |
| `model_size` | `46M` | Model to train per trial (`8M`, `46M`) |
| `result_dir` | `neps_runs/46_LI_space` | Base directory for NEPS output |
| `runtime` | `2700` | Time budget in **minutes** passed to NePS |
| `neps_space_config` | `NLinesU_f_l_nw` | NOS search space name (from `SPACES` registry) |
| `neps_mode` | `normal` | `normal` / `continuation` / `overwrite` / `results` |
| `warmstarter` | `SGDM_inter` | (optional) Warmstarter to seed the search |
| `evaluations` | `576` | (optional) Total evaluation budget (mutually exclusive with `runtime`) |
| `nproc_per_node` | `4` | GPUs per evaluation |

### `$TMPDIR` Optimization (`neps_tmpdir_wrapper.sh`)

Running on HPC clusters, network filesystem I/O is a bottleneck for NePS (which reads/writes many small files). The wrapper:

1. **Copies N arrow dataset files** to `$TMPDIR/dataset/` (default 1 file ≈ 5 GB; set `NUM_FILES=2` for longer runs).
2. Runs the pipeline **from HOME** — code and NePS results stay on the shared filesystem for multi-node coordination.
3. **Periodic background sync** (default every 2 min, set `SYNC_INTERVAL=N`) copies SLURM logs from `$TMPDIR/logs/` back to `neps_runs/_log/` to prevent data loss.
4. **Final log sync** after the job ends.

Environment variables:
- `NUM_FILES` — number of arrow dataset files to copy (default 2)
- `SYNC_INTERVAL` — log sync period in minutes (default 2)

### `neps_mode` Values

| Mode | Behavior |
|------|----------|
| `normal` | Start new run; warn if directory already exists |
| `continuation` | Resume an existing NePS run without re-seeding warmstarter |
| `overwrite` | Delete result JSON and NePS directory, then restart fresh |
| `results` | Skip NePS execution entirely; only collect and save results from an existing run |

---

## 11. Utilities & Analysis Tools

### `neps_nos/utils/extract_best_function.py`

Post-hoc analysis: iterates over runs, finds the best configuration by `objective_to_minimize`, loads the optimizer class and prints it.

```bash
# Edit folder/run variables in the script, then:
python neps_nos/utils/extract_best_function.py
```

### `neps_nos/utils/create_warmstarting_configs.py`

Samples one random config from a given NOS space and prints the sampled NEPS config dict plus the instantiated optimizer. Useful for debugging or creating new warmstarter entries.

```bash
python neps_nos/utils/create_warmstarting_configs.py --nos_space NLinesU_f_l_nw
```

### `neps_plots/plotting.py`

Core plotting utilities that load summary JSONs from `neps_runs/` and generate performance curves (incumbent vs. cost). Handles:
- Multi-run aggregation with confidence bands (mean ± std across seeds)
- Automatic legend placement (left, bottom, or right-hand side)
- Boundary point injection for curves that reach axis limits
- Custom styling for different optimizer families

### `neps_plots/scripts/`

Pre-configured plotting scripts for common comparisons:
- `scripts/8/` — 8M model optimizer comparison plots
- `scripts/46/` — 46M model optimizer comparison plots
- `scripts/extension/` — Extended comparisons (SOTA baselines, ablations, mutation studies)

### `neps.status(neps_dir)`

The `process_neps_status()` function in `neps_pipeline.py` wraps `neps.status()` to extract loss, cost, and fidelity histories from a completed or in-progress NePS run.

### `copy_results.sh`

Utility script for batch-copying result JSONs from one run directory to another, handling seed variations and result aggregation.

### `synthetic/`

A self-contained set of scripts for testing NOS on synthetic optimization problems (sphere, ellipsoid, etc.) and small ML benchmarks, independent of the LM training pipeline. Has its own `nos_space.py` that mirrors the main `neps_nos_space.py`. Use `nos_synthetic.py` / `nos_ml_benchmarks.py` as entry points.

---

## 12. Running the Pipeline

### Quick local test (8M model, 4 GPUs)

```bash
source .venvbin/activate
python neps_nos/neps_pipeline.py \
  --result_dir neps_runs/testrun1 \
  --neps_space_config NLinesU_f_l_nw \
  --model_size 8M \
  --neps_optimizer LI1_rand0.1 \
  --runtime 60 \
  --seed 0
```

### Multi-GPU run (4 GPUs, 46M model)

```bash
python neps_nos/neps_pipeline.py \
  --result_dir neps_runs/46_test \
  --neps_space_config NLinesU_f_l_nw \
  --model_size 46M \
  --neps_optimizer LI1_rand0.1 \
  --nproc_per_node 4 \
  --runtime 2700 \
  --seed 0 \
  --runname my_experiment
```

### SLURM (production)

```bash
sbatch neps_cluster_scripts/sota/LI1_rand01_Nlines.sh
```

### Continue an existing run

```bash
python neps_nos/neps_pipeline.py \
  --result_dir neps_runs/46_LI_space \
  --neps_space_config NLinesU_f_l_nw \
  --model_size 46M \
  --neps_optimizer LI1_rand0.1 \
  --nproc_per_node 4 \
  --runtime 2700 \
  --seed 0 \
  --runname my_experiment \
  --neps_mode continuation
```

### Extract results only (no training)

```bash
python neps_nos/neps_pipeline.py \
  --result_dir neps_runs/46_LI_space \
  --neps_space_config NLinesU_f_l_nw \
  --neps_optimizer LI1_rand0.1 \
  --seed 0 \
  --runname my_experiment \
  --neps_mode results
```

### All `neps_pipeline.py` arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--result_dir` | **yes** | — | Root directory for all outputs |
| `--neps_space_config` | **yes** | — | NOS space key from `SPACES` registry |
| `--neps_optimizer` | no | `RE` | NePS search strategy key from `OPTIMIZERS` registry |
| `--model_size` | no | `8M` | `8M`, `8M_1D`, `46M`, `46M_1D` |
| `--nproc_per_node` | no | `1` | GPUs per trial (`1` = single process, `>1` = torchrun DDP) |
| `--runtime` | no | None | Time budget in **minutes** |
| `--evaluations` | no | None | Evaluation budget (multiplied by `max_fidelity` internally) |
| `--seed` | no | `0` | Random seed (also used as SLURM array task index) |
| `--runname` | no | None | Human-readable run name; changes directory structure |
| `--warmstarter` | no | None | Warmstarter key from `WARMSTART_PARAMETERS` |
| `--neps_mode` | no | `normal` | `normal` / `continuation` / `overwrite` / `results` |
| `--lr_mode` | no | `normal` | `normal` / `sweep` (try 4 LRs per config, take min) |
| `--trainset_path` | no | None | Override train dataset path (for `$TMPDIR`) |
| `--validset_path` | no | None | Override valid dataset path (for `$TMPDIR`) |
