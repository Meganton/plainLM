#!/bin/bash
#
# NEPS runner with automatic $TMPDIR optimization
# Source this file and use: run_neps_with_tmpdir <args>
#
# Default: Copies 10 arrow files (~5 GB, supports up to 6K training steps)
# Override: Set NUM_FILES=20 before calling for longer runs
#

# Function to run NEPS with automatic $TMPDIR dataset handling
run_neps_with_tmpdir() {
    # Parse optional NUM_FILES from environment or use default (10 files = ~5GB, up to 6K steps)
    local num_files=${NUM_FILES:-10}
    local skip_copy=${SKIP_TMPDIR_COPY:-false}
    
    # Copy dataset to $TMPDIR if not skipped
    if [ "$skip_copy" = "false" ]; then
        echo "=============================================="
        echo "Setting up $TMPDIR optimization..."
        echo "=============================================="
        
        # Run the copy script
        NUM_FILES=$num_files source "$(dirname "${BASH_SOURCE[0]}")/copy_dataset_to_tmpdir.sh"
        
        # Check if copy succeeded
        if [ ! -d "$TMPDIR/dataset/train" ]; then
            echo "ERROR: Dataset copy to \$TMPDIR failed!"
            return 1
        fi
        
        echo "Dataset ready on local SSD"
        echo "=============================================="
    else
        echo "Skipping dataset copy (SKIP_TMPDIR_COPY=true)"
    fi
    
    # Run NEPS with dataset path overrides
    python -u neps_nos/neps_pipeline.py \
        "$@" \
        --trainset_path "${TMPDIR_TRAIN_PATH}" \
        --validset_path "${TMPDIR_VALID_PATH}"
    
    return $?
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
