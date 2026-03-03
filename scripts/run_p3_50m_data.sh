#!/bin/bash
# Phase 3 Experiment: 50M Data Scaling Test
#
# Purpose: Determine if more data (50M vs 10M tokens) improves convergence
#          using same model (256H × 4L baseline)
#
# Compute: ~30 GPU hours (H100)
# Goal:    Beat baseline loss of 4.31
#
# Monitor: watch -n 5 'tail -20 outputs/p3-optimized-50m-data/loss_curve.csv'

set -e

# Activate venv
source .venv/bin/activate

CONFIG="config/milestones/p3_optimized_50m_data.toml"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 PHASE 3: DATA SCALING TEST (50M tokens)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Configuration:"
echo "   Model:     256H × 4L (same as baseline)"
echo "   Data:      50M interleaved tokens (5× baseline)"
echo "   Steps:     10,000 (with early stopping)"
echo "   Batch:     16 effective (8 × 2)"
echo "   Compute:   ~30 GPU hours"
echo ""
echo "🎯 Hypothesis:"
echo "   If loss < 3.9: Data was bottleneck → scale data is high-ROI"
echo "   If loss ≥ 4.1: Data quantity not limiting → focus on model"
echo ""
echo "📝 Output:"
echo "   Directory: outputs/p3-optimized-50m-data/"
echo "   Metrics:   loss_curve.csv + early stopping tracking"
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
echo "  Results: outputs/p3-optimized-50m-data/loss_curve.csv"
echo "  Log:     $(ls -t outputs/p3-optimized-50m-data/p3*.log 2>/dev/null | head -1)"
echo ""
echo "📊 Next steps:"
echo "  1. Review loss curve (should show smoother convergence)"
echo "  2. Compare final loss to baseline (4.31)"
echo "  3. If loss < 3.9, combine with 512h_8l for full scaling test"
echo "  4. If loss ≥ 4.1, model architecture is more critical"
