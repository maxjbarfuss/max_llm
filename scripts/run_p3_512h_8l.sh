#!/bin/bash
# Phase 3 Experiment: 512H × 8L Model Scaling Test
#
# Purpose: Determine if larger model (512H x 8L vs 256H x 4L baseline)
#          converges to lower loss on same data (10M interleaved tokens)
#
# Compute: ~25 GPU hours (H100)
# Goal:    Beat baseline loss of 4.31
#
# Monitor: Watch loss_curve.csv in real-time:
#   watch -n 5 'tail -20 outputs/p3-optimized-512h-8l/loss_curve.csv'

set -e

CONFIG="config/milestones/p3_optimized_512h_8l.toml"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 PHASE 3: MODEL SCALING TEST (512H × 8L)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Configuration:"
echo "   Model:     512H × 8L (vs 256H × 4L baseline)"
echo "   Data:      10M interleaved tokens (same as baseline)"
echo "   Steps:     5,000"
echo "   Batch:     16 effective (8 × 2)"
echo "   Compute:   ~25 GPU hours"
echo ""
echo "🎯 Hypothesis:"
echo "   If loss < 4.1: Model capacity was bottleneck → scale more"
echo "   If loss ≥ 4.2: Model size not limiting → focus on data/LR"
echo ""
echo "📝 Output:"
echo "   Directory: outputs/p3-optimized-512h-8l/"
echo "   Metrics:   loss_curve.csv (step, loss, ppl, lr, val_loss)"
echo "   Log:       training output to console + file"
echo ""
echo "⏱️  Starting: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo "❌ ERROR: Config not found: $CONFIG"
    exit 1
fi

# Verify torch
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || exit 1

# Run training
python -m src.training.train --config "$CONFIG"

echo ""
echo "✓ Training complete!"
echo "  Results: outputs/p3-optimized-512h-8l/loss_curve.csv"
echo "  Log:     $(ls -t outputs/p3-optimized-512h-8l/p3*.log 2>/dev/null | head -1)"
echo ""
echo "📊 Next steps:"
echo "  1. Review loss curve (should show convergence)"
echo "  2. Compare final loss to baseline (4.31)"
echo "  3. If loss < 4.1, proceed with 50m_data test"
echo "  4. If loss ≥ 4.2, data/LR are more critical"
