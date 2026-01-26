#!/bin/bash
#
# NEPS runner with full $TMPDIR optimization
# Source this file and use: run_neps_with_tmpdir <args>
#
# Copies ENTIRE PROJECT + DATASET to local SSD to eliminate metadata operations
# Default: Copies 10 arrow files (~5 GB) + code (~1.6 MB)
# Override: Set NUM_FILES=20 before calling for longer runs
#

# Function to copy code directories to $TMPDIR (only once per job)
copy_code_to_tmpdir() {
    if [ -d "$TMPDIR/plainLM" ]; then
        echo "Code already in $TMPDIR, skipping copy"
        return 0
    fi
    
    echo "=================================================="
    echo "Copying code directories to \$TMPDIR for full SSD optimization"
    echo "=================================================="
    
    # Get the project root (parent of the script's parent)
    local project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    
    # Create tmpdir project structure
    mkdir -p "$TMPDIR/plainLM"
    
    # Copy critical code directories (only 1.6 MB total)
    echo "Copying code directories..."
    for dir in neps_nos models engine data config optim; do
        if [ -d "$project_root/$dir" ]; then
            cp -r "$project_root/$dir" "$TMPDIR/plainLM/" || { echo "ERROR: Failed to copy $dir"; return 1; }
        fi
    done
    
    # Copy Python files
    echo "Copying Python modules..."
    for file in utils.py torch_utils.py checkpoint_utils.py nos_train.py train.py; do
        if [ -f "$project_root/$file" ]; then
            cp "$project_root/$file" "$TMPDIR/plainLM/" || { echo "ERROR: Failed to copy $file"; return 1; }
        fi
    done
    
    echo "Code directories copied to \$TMPDIR"
    echo "=================================================="
}

# Function to run NEPS with automatic $TMPDIR full optimization
run_neps_with_tmpdir() {
    # Parse optional NUM_FILES from environment or use default (10 files = ~5GB, up to 6K steps)
    local num_files=${NUM_FILES:-10}
    local skip_copy=${SKIP_TMPDIR_COPY:-false}
    
    # Get the project root (for comparison later)
    local project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    
    if [ "$skip_copy" = "false" ]; then
        echo "=============================================="
        echo "Setting up full \$TMPDIR optimization..."
        echo "=============================================="
        
        # Copy code to tmpdir first
        copy_code_to_tmpdir || return 1
        
        # Copy dataset to $TMPDIR
        echo "=============================================="
        echo "Copying dataset subset to \$TMPDIR"
        echo "=============================================="
        NUM_FILES=$num_files source "$(dirname "${BASH_SOURCE[0]}")/copy_dataset_to_tmpdir.sh"
        
        # Check if copy succeeded
        if [ ! -d "$TMPDIR/dataset/train" ]; then
            echo "ERROR: Dataset copy to \$TMPDIR failed!"
            return 1
        fi
        
        echo "Dataset ready on local SSD"
        echo "=============================================="
    else
        echo "Skipping tmpdir copy (SKIP_TMPDIR_COPY=true)"
    fi
    
    # Change to tmpdir and run NEPS from there
    cd "$TMPDIR/plainLM" || return 1
    
    echo "Running NEPS from \$TMPDIR to minimize HOME filesystem metadata operations..."
    
    # Run NEPS with dataset path overrides
    python -u neps_nos/neps_pipeline.py \
        "$@" \
        --trainset_path "${TMPDIR_TRAIN_PATH}" \
        --validset_path "${TMPDIR_VALID_PATH}"
    
    local neps_exit_code=$?
    
    # Sync results back to HOME (always, even if NEPS had partial failures)
    echo "=============================================="
    echo "Syncing results back to HOME..."
    echo "=============================================="
    if [ -d "$TMPDIR/plainLM/neps_runs" ]; then
        rsync -av "$TMPDIR/plainLM/neps_runs/" "$project_root/neps_runs/" || echo "WARNING: rsync of results failed"
        echo "Results synced to: $project_root/neps_runs/"
    else
        echo "No NEPS results found in tmpdir (job may have failed during initialization)"
    fi
    
    return $neps_exit_code
}

# Alternative: Simple wrapper that just adds the paths
run_neps_simple() {
    # Assumes dataset is already copied and env vars are set
    python -u neps_nos/neps_pipeline.py \
        "$@" \
        --trainset_path "${TMPDIR_TRAIN_PATH:-/home/hk-project-p0023364/hgf_omt7140/data/lm/fwedu/fwedu_sample_100B_tokenizer_GPTNeoX/tokenized_EleutherAI_gpt-neox-20b/ctx_2048/train}" \
        --validset_path "${TMPDIR_VALID_PATH:-/home/hk-project-p0023364/hgf_omt7140/data/lm/fwedu/fwedu_sample_100B_tokenizer_GPTNeoX/tokenized_EleutherAI_gpt-neox-20b/ctx_2048/valid}"
}

# Export functions so they're available in the script
export -f run_neps_with_tmpdir
export -f run_neps_simple
