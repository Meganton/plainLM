#!/bin/bash

# Create target directories if they don't exist
mkdir -p neps_runs/46_LI_space/results
mkdir -p neps_runs/sota_8/results

# Copy 46M results to 46_LI_space/results with _0 to _4 suffixes
echo "Copying 46M result files..."
for file in neps_runs/single_configs/46_*.json; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .json)
        for i in {0..4}; do
            cp "$file" "neps_runs/46_LI_space/results/${filename}_${i}.json"
            echo "Copied: $file -> neps_runs/46_LI_space/results/${filename}_${i}.json"
        done
    fi
done

# Copy 8M results to sota_8/results with _0 to _4 suffixes
echo ""
echo "Copying 8M result files..."
for file in neps_runs/single_configs/8_*.json; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .json)
        for i in {0..4}; do
            cp "$file" "neps_runs/sota_8/results/${filename}_${i}.json"
            echo "Copied: $file -> neps_runs/sota_8/results/${filename}_${i}.json"
        done
    fi
done

echo ""
echo "Done!"
