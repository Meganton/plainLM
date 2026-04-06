#!/bin/bash
#
# NEPS runner with $TMPDIR dataset optimization.
# Source this file and call: run_neps_with_tmpdir <args>
#
# What this does:
#   1. Copies the dataset to local SSD ($TMPDIR) to avoid network I/O during training.
#   2. Runs the NEPS pipeline from HOME (code + NePS files stay on the shared filesystem).
#   3. Periodically syncs SLURM logs from $TMPDIR back to HOME.
#   4. After Python exits, echoes a fence line (forces slurmstepd to drain the pipe)
#      then does a final rsync to capture the complete log.
#
# Environment variables:
#   NUM_FILES      Number of arrow dataset files to copy (default: 10, ~5 GB)
#   SYNC_INTERVAL  Log sync period in minutes (default: 2)
#

run_neps_with_tmpdir() {
    local num_files=${NUM_FILES:-10}
    local sync_interval=${SYNC_INTERVAL:-2}
    local sync_interval_seconds=$((sync_interval * 60))
    local project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    local log_dir="$project_root/neps_runs/_log/${SLURM_JOB_NAME:-neps}/${SLURM_ARRAY_JOB_ID:-0}"

    # ── 1. Copy dataset to local SSD ──────────────────────────────────────────
    echo "Copying dataset to \$TMPDIR ($num_files files)..."
    NUM_FILES=$num_files source "$(dirname "${BASH_SOURCE[0]}")/copy_dataset_to_tmpdir.sh"
    if [[ ! -d "$TMPDIR/dataset/train" ]]; then
        echo "ERROR: Dataset copy to \$TMPDIR failed!"
        return 1
    fi
    echo "Dataset ready at \$TMPDIR/dataset"
    
    # ── 2. Periodic log sync in background ───────────────────────────────────
    mkdir -p "$log_dir"
    (
        while true; do
            sleep $sync_interval_seconds
            echo "--- Syncing logs to $log_dir at $(date) ---"
            [[ -d "$TMPDIR/logs" ]] && rsync -a "$TMPDIR/logs/" "$log_dir/" 2>/dev/null || true
        done
    ) &
    local sync_pid=$!

    # ── 3. Run NEPS from HOME with dataset overrides ──────────────────────────
    echo "Starting NEPS from $project_root ..."
    cd "$project_root"
    python -u neps_nos/neps_pipeline.py "$@" \
        --trainset_path "$TMPDIR/dataset/train" \
        --validset_path "$TMPDIR/dataset/valid"
    local exit_code=$?


    echo "NEPS run finished (exit code: $exit_code)."
    echo "Syncing logs to $log_dir ..."
    
    [[ -d "$TMPDIR/logs" ]] && rsync -a "$TMPDIR/logs/" "$log_dir/" 2>/dev/null || true

    echo "Stopping log sync."

    # ── 4. Final log sync ─────────────────────────────────────────────────────
    kill $sync_pid 2>/dev/null; wait $sync_pid 2>/dev/null
    
    return $exit_code
}

export -f run_neps_with_tmpdir
