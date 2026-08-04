#!/bin/bash
#
# Copy dataset subset to $TMPDIR for efficient I/O
# This script copies only the necessary arrow files for training
#

set -e  # Exit on error

# Default parameters (can be overridden)
NUM_FILES=${NUM_FILES:-32}  # Default: 32 files for up to 6K steps
SOURCE_DIR="/work/dlc2workfs2/gebureka-neps_bo/LLM_task/data/datasets/fwedu/fwedu_sample_100B_tokenizer_GPTNeoX/tokenized_EleutherAI_gpt-neox-20b/ctx_2048"

# Check if $TMPDIR is set
if [ -z "$TMPDIR" ]; then
    echo "ERROR: \$TMPDIR is not set. This script should be run within a SLURM job."
    exit 1
fi

echo "=============================================="
echo "Copying dataset subset to \$TMPDIR"
echo "=============================================="
echo "Source: $SOURCE_DIR"
echo "Target: $TMPDIR/dataset"
echo "Files to copy: $NUM_FILES arrow files per dataset"
echo "Estimated time: ~1 minute"
echo "Estimated size: ~$((NUM_FILES * 477 / 1000)) GB"
echo "=============================================="

# Create target directories
mkdir -p $TMPDIR/dataset/train
mkdir -p $TMPDIR/dataset/valid

# Copy training dataset metadata files
echo "[1/4] Copying train metadata files..."
cp -v $SOURCE_DIR/train/dataset_info.json $TMPDIR/dataset/train/
cp -v $SOURCE_DIR/train/state.json $TMPDIR/dataset/train/

# Copy first N arrow files from training set
echo "[2/4] Copying first $NUM_FILES training arrow files..."
for i in $(seq -f "%05g" 0 $((NUM_FILES - 1))); do
    if [ -f "$SOURCE_DIR/train/data-${i}-of-000801.arrow" ]; then
        cp -v $SOURCE_DIR/train/data-${i}-of-000801.arrow $TMPDIR/dataset/train/
    else
        echo "WARNING: File data-${i}-of-000801.arrow not found, skipping..."
    fi
done

# Update metadata to reflect only copied files
echo "[3/6] Updating train metadata to reflect copied files..."
python3 << EOF
import json
metadata_path = "$TMPDIR/dataset/train/state.json"
with open(metadata_path, 'r') as f:
    state = json.load(f)
# Update the number of shards to match copied files
if '_data_files' in state:
    # Keep only the first $NUM_FILES entries
    state['_data_files'] = state['_data_files'][:$NUM_FILES]
with open(metadata_path, 'w') as f:
    json.dump(state, f, indent=2)
print(f"Updated state.json to reference only $NUM_FILES files")
EOF

# Copy validation dataset (small, only 77MB total)
echo "[4/6] Copying validation dataset..."
cp -v $SOURCE_DIR/valid/* $TMPDIR/dataset/valid/

# Verify copy
echo "[5/6] Verifying copy..."
train_files=$(ls $TMPDIR/dataset/train/*.arrow 2>/dev/null | wc -l)
valid_files=$(ls $TMPDIR/dataset/valid/*.arrow 2>/dev/null | wc -l)
echo "Training files copied: $train_files"
echo "Validation files copied: $valid_files"

if [ "$train_files" -eq 0 ]; then
    echo "ERROR: No training files were copied!"
    exit 1
fi

echo "[6/6] Final verification of dataset loading..."
python3 << 'EOF'
import sys
try:
    from datasets import load_from_disk
    import os
    tmpdir_train = os.path.join(os.environ['TMPDIR'], 'dataset', 'train')
    dataset = load_from_disk(tmpdir_train)
    print(f"✓ Dataset loaded successfully: {len(dataset):,} samples available")
except Exception as e:
    print(f"✗ ERROR loading dataset: {e}")
    sys.exit(1)
EOF

echo "=============================================="
echo "Dataset copy completed successfully!"
echo "Train data available at: \$TMPDIR/dataset/train"
echo "Valid data available at: \$TMPDIR/dataset/valid"
echo "=============================================="

# Export path for use in training scripts
export TMPDIR_TRAIN_PATH="$TMPDIR/dataset/train"
export TMPDIR_VALID_PATH="$TMPDIR/dataset/valid"

echo "Environment variables set:"
echo "  TMPDIR_TRAIN_PATH=$TMPDIR_TRAIN_PATH"
echo "  TMPDIR_VALID_PATH=$TMPDIR_VALID_PATH"
