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
    local sync_interval=${SYNC_INTERVAL:-2}  # Default: 2 minutes
    local sync_interval_seconds=$((sync_interval * 60))  # Convert to seconds
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
    
    # Start periodic sync in background
    echo "Starting periodic sync (every ${sync_interval} min) for logs and NEPS results..."
    (
        while true; do
            sleep $sync_interval_seconds
            
            # Sync SLURM logs from TMPDIR to HOME
            if [ -d "$TMPDIR/logs" ]; then
                mkdir -p "$project_root/neps_runs/_log/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/"
                rsync -a --whole-file "$TMPDIR/logs/" "$project_root/neps_runs/_log/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/" 2>/dev/null || true
            fi
            
            # Sync NEPS results from TMPDIR to HOME
            if [ -d "$TMPDIR/plainLM/neps_runs" ]; then
                rsync -a --whole-file "$TMPDIR/plainLM/neps_runs/" "$project_root/neps_runs/" 2>/dev/null || true
            fi
            
            # Log sync timestamp (use task-specific log for array jobs)
            local sync_log_file="$TMPDIR/logs/sync.log"
            if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
                sync_log_file="$TMPDIR/logs/sync_${SLURM_ARRAY_TASK_ID}.log"
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Periodic sync completed" >> "$sync_log_file"
        done
    ) &
    SYNC_PID=$!
    echo "Periodic sync running (PID: $SYNC_PID)"
    
    # Run NEPS with dataset path overrides
    python -u neps_nos/neps_pipeline.py \
        "$@" \
        --trainset_path "${TMPDIR_TRAIN_PATH}" \
        --validset_path "${TMPDIR_VALID_PATH}"
    
    local neps_exit_code=$?
    
    # Kill the periodic sync process
    echo "Stopping periodic sync..."
    kill $SYNC_PID 2>/dev/null || true
    wait $SYNC_PID 2>/dev/null || true
    
    # Sync results back to HOME (always, even if NEPS had partial failures)
    echo "=============================================="
    echo "Final sync: copying results and logs back to HOME..."
    echo "=============================================="
    
    # Sync NEPS results
    if [ -d "$TMPDIR/plainLM/neps_runs" ]; then
        rsync -av --whole-file "$TMPDIR/plainLM/neps_runs/" "$project_root/neps_runs/" || echo "WARNING: rsync of NEPS results failed"
        echo "NEPS results synced to: $project_root/neps_runs/"
    else
        echo "No NEPS results found in tmpdir (job may have failed during initialization)"
    fi
    
    # Sync SLURM logs
    if [ -d "$TMPDIR/logs" ]; then
        mkdir -p "$project_root/neps_runs/_log/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/"
        rsync -av --whole-file "$TMPDIR/logs/" "$project_root/neps_runs/_log/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/" || echo "WARNING: rsync of logs failed"
        echo "Logs synced to: $project_root/neps_runs/_log/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}/"
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
