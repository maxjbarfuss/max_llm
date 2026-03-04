#!/bin/bash
# Run optimized hyperparameter test: Lower LR + longer warmup + FAST training

set -e

source .venv/bin/activate

echo "=================================================="
echo "🚀 PHASE 3: OPTIMIZED HYPERPARAMETER TUNING (FAST)"
echo "=================================================="
echo ""
echo "📊 Configuration:"
echo "   Model:     256H × 4L (baseline)"
echo "   Data:      50M mixed (wiki+stories+webtext) ✓ VERIFIED"
echo "   Batch:     16 effective (BASELINE, not doubled)"
echo "   LR:        0.0001 (↓10x from 0.0003)"
echo "   Warmup:    1000 steps (↑5x from 200)"
echo "   Speed:     torch_compile ENABLED (~20% faster)"
echo ""
echo "🎯 Why this works:"
echo "   - Baseline batch proven stable"
echo "   - Lower LR prevents divergence on large data"
echo "   - Long warmup gradual learning"
echo "   - torch_compile + pre-cached data = fast"
echo ""
echo "Expected:"
echo "   Loss 4.1-4.3 (matching or better than baseline)"
echo "   ~30-35 GPU hours total"
echo ""
echo "=================================================="
echo ""

# Clean and launch
rm -rf outputs/p3-optimized-lowlr-fast

python -m src.training.train --config config/milestones/p3_optimized_lowlr_fast.toml

echo ""
echo "✓ Training complete!"
tail -5 outputs/p3-optimized-lowlr-fast/loss_curve.csv | tail -1 | awk -F',' '{print "Final loss: " $2 " (val: " $7 ")"}'
echo ""
