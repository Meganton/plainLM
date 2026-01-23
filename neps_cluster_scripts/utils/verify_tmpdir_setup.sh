#!/bin/bash
# Verification script to check if all $TMPDIR optimization files are in place

echo "=========================================="
echo "$TMPDIR I/O Optimization - Setup Verification"
echo "=========================================="
echo ""

errors=0
warnings=0

# Check helper script
echo "[1/5] Checking helper script..."
if [ -f "neps_cluster_scripts/utils/copy_dataset_to_tmpdir.sh" ]; then
    if [ -x "neps_cluster_scripts/utils/copy_dataset_to_tmpdir.sh" ]; then
        echo "  ✅ neps_cluster_scripts/utils/copy_dataset_to_tmpdir.sh exists and is executable"
    else
        echo "  ⚠️  neps_cluster_scripts/utils/copy_dataset_to_tmpdir.sh exists but is not executable"
        echo "     Run: chmod +x neps_cluster_scripts/utils/copy_dataset_to_tmpdir.sh"
        warnings=$((warnings + 1))
    fi
else
    echo "  ❌ neps_cluster_scripts/utils/copy_dataset_to_tmpdir.sh NOT FOUND"
    errors=$((errors + 1))
fi

# Check example job script
echo "[2/5] Checking example job script..."
if [ -f "neps_cluster_scripts/LI_space/LI1_TMPDIR_example.sh" ]; then
    if [ -x "neps_cluster_scripts/LI_space/LI1_TMPDIR_example.sh" ]; then
        echo "  ✅ LI1_TMPDIR_example.sh exists and is executable"
    else
        echo "  ⚠️  LI1_TMPDIR_example.sh exists but is not executable"
        warnings=$((warnings + 1))
    fi
else
    echo "  ❌ LI1_TMPDIR_example.sh NOT FOUND"
    errors=$((errors + 1))
fi

# Check test script
echo "[3/5] Checking test script..."
if [ -f "neps_cluster_scripts/tests/test_tmpdir.sh" ]; then
    if [ -x "neps_cluster_scripts/tests/test_tmpdir.sh" ]; then
        echo "  ✅ test_tmpdir.sh exists and is executable"
    else
        echo "  ⚠️  test_tmpdir.sh exists but is not executable"
        warnings=$((warnings + 1))
    fi
else
    echo "  ❌ test_tmpdir.sh NOT FOUND"
    errors=$((errors + 1))
fi

# Check documentation
echo "[4/5] Checking documentation..."
doc_count=0
if [ -f "TMPDIR_MIGRATION.md" ]; then
    echo "  ✅ TMPDIR_MIGRATION.md exists"
    doc_count=$((doc_count + 1))
else
    echo "  ❌ TMPDIR_MIGRATION.md NOT FOUND"
    errors=$((errors + 1))
fi

if [ -f "TMPDIR_IMPLEMENTATION_SUMMARY.md" ]; then
    echo "  ✅ TMPDIR_IMPLEMENTATION_SUMMARY.md exists"
    doc_count=$((doc_count + 1))
else
    echo "  ⚠️  TMPDIR_IMPLEMENTATION_SUMMARY.md NOT FOUND"
    warnings=$((warnings + 1))
fi

# Check Python modifications
echo "[5/5] Checking Python code modifications..."
python_ok=0

if grep -q "trainset_path.*TMPDIR" neps_nos/neps_pipeline.py 2>/dev/null; then
    echo "  ✅ neps_pipeline.py has $TMPDIR support"
    python_ok=$((python_ok + 1))
else
    echo "  ❌ neps_pipeline.py missing $TMPDIR support"
    errors=$((errors + 1))
fi

if grep -q "trainset_path.*TMPDIR" neps_nos/train_distributed.py 2>/dev/null; then
    echo "  ✅ train_distributed.py has $TMPDIR support"
    python_ok=$((python_ok + 1))
else
    echo "  ❌ train_distributed.py missing $TMPDIR support"
    errors=$((errors + 1))
fi

if grep -q "trainset_path.*TMPDIR" neps_nos/model_train_function.py 2>/dev/null; then
    echo "  ✅ model_train_function.py has $TMPDIR support"
    python_ok=$((python_ok + 1))
else
    echo "  ❌ model_train_function.py missing $TMPDIR support"
    errors=$((errors + 1))
fi

# Summary
echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="

if [ $errors -eq 0 ] && [ $warnings -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED!"
    echo ""
    echo "Your $TMPDIR optimization is fully set up and ready to use."
    echo ""
    echo "Next steps:"
    echo "  1. Test with: sbatch neps_cluster_scripts/tests/test_tmpdir.sh"
    echo "  2. Read TMPDIR_IMPLEMENTATION_SUMMARY.md for usage instructions"
    echo "  3. Apply to your production jobs"
    exit 0
elif [ $errors -eq 0 ]; then
    echo "⚠️  Setup complete with $warnings warning(s)"
    echo ""
    echo "Everything essential is in place, but some optional issues were found."
    echo "Review the warnings above."
    exit 0
else
    echo "❌ SETUP INCOMPLETE - $errors error(s), $warnings warning(s)"
    echo ""
    echo "Some critical files are missing or incorrectly configured."
    echo "Please review the errors above and fix them."
    exit 1
fi
