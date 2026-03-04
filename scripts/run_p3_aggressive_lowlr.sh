#!/bin/bash
# Run aggressive hyperparameter tuning test
# Lower LR (0.0001) + longer warmup (1000) to fix divergence on 50M data

set -e

cd /home/max/dev/max_llm
source .venv/bin/activate

echo "=================================================="
echo "🚀 PHASE 3: AGGRESSIVE HYPERPARAMETER TUNING"
echo "=================================================="
echo ""
echo "📊 Configuration:"
echo "   Model:     256H × 4L (same as baseline)"
echo "   Data:      50M mixed tokens (wiki+stories+webtext) ✓ VERIFIED"
echo "   Learning rate: 0.0001 (↓10x from 0.0003)"
echo "   Warmup:    1000 steps (↑5x from 200)"
echo "   Batch:     32 effective (8 × 4)"
echo "   Compute:   ~30 GPU hours"
echo ""
echo "🎯 Hypothesis:"
echo "   Previous divergence due to LR too high + warmup too short"
echo "   Gentler LR + longer warmup → smooth convergence"
echo ""
echo "📈 Expected result:"
echo "   If loss < 4.5: Aggressive tuning works (we're back on track)"
echo "   If loss < 4.0: Better than baseline despite larger data"
echo "   If loss < 3.8: Major breakthrough"
echo ""
echo "⏱️  Timeline:"
echo "   Warmup phase: steps 0-1000 (gentle learning)"
echo "   Training phase: steps 1000-10000 (full learning)"
echo "   Checkpoint every 500 steps"
echo "   Early stopping if val loss plateaus (patience=5)"
echo ""
echo "=================================================="
echo ""

# Clean previous output
rm -rf outputs/p3-aggressive-lowlr-hyperparams

# Launch training
python train.py \
    --config config/milestones/p3_aggressive_lowlr_hyperparams.toml \
    --output-dir outputs/p3-aggressive-lowlr-hyperparams \
    --use-ddp \
    --rank 0

echo ""
echo "=================================================="
echo "✓ Training complete!"
echo "=================================================="
echo ""
echo "📊 Results:"
tail -5 outputs/p3-aggressive-lowlr-hyperparams/loss_curve.csv | tail -1 | awk -F',' '{print "   Final loss: " $2 " (val: " $7 ")"}'
echo ""
echo "📈 Next steps:"
echo "   1. Check convergence pattern vs previous"
echo "   2. Compare final loss to baseline (4.31)"
echo "   3. If < 4.5: we've fixed the divergence"
echo "   4. If successful: iterate to even lower LR (0.00005)"
echo ""
