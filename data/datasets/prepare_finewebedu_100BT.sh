#!/bin/bash

# This script will download and preprocess FineWebEdu-100BT.
# Expect some token loss by batched concat_chunk.

# Old setup (kept for reference):
# mkdir -p "~/tmp"
# cd ~/plainLM
# source .venv/bin/activate

mkdir -p "/work/dlc2workfs2/gebureka-neps_bo/LLM_task/data/tmp_cache"
mkdir -p "/work/dlc2workfs2/gebureka-neps_bo/LLM_task/data/fwedu"
cd /work/dlc2workfs2/gebureka-neps_bo/LLM_task
source .venv/bin/activate

# Tokenization is CPU-bound, so use more workers and a larger batch on the 128-CPU nodes.
TOKENIZE_NUM_PROC=16
TOKENIZE_BATCH_SIZE=2048

# Keep Hugging Face caches off the home directory quota.
export HF_HOME="/work/dlc2workfs2/gebureka-neps_bo/LLM_task/data/tmp_cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="/work/dlc2workfs2/gebureka-neps_bo/LLM_task/data/tmp_cache/xdg"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

# Old command (kept for reference):
# PYTHONPATH=. python data/datasets/prepare.py \
#   --out_path="~/data/lm/fwedu/fwedu_sample_100B_tokenizer_GPTNeoX" \
#   --cache_path="~/tmp" \
#   --download --tokenize --chunk \
#   --save_tokenized --save_tokenizer \
#   --dataset_path="HuggingFaceFW/fineweb-edu" \
#   --dataset_split="train" \
#   --dataset_name="sample-100BT" \
#   --tokenizer="EleutherAI/gpt-neox-20b" \
#   --seq_length=2048 \
#   --split_train_valid=True \
#   --n_tokens_valid=10000000


PYTHONPATH=. python data/datasets/prepare.py \
  --out_path="/work/dlc2workfs2/gebureka-neps_bo/LLM_task/data/fwedu/fwedu_sample_100B_tokenizer_GPTNeoX" \
  --cache_path="/work/dlc2workfs2/gebureka-neps_bo/LLM_task/data/tmp_cache" \
  --download --tokenize --chunk \
  --save_tokenizer \
  --dataset_path="HuggingFaceFW/fineweb-edu" \
  --dataset_split="train" \
  --dataset_name="sample-100BT" \
  --tokenizer="EleutherAI/gpt-neox-20b" \
  --seq_length=2048 \
  --map_num_proc="$TOKENIZE_NUM_PROC" \
  --map_batch_size="$TOKENIZE_BATCH_SIZE" \
  --split_train_valid=True \
  --n_tokens_valid=10000000
