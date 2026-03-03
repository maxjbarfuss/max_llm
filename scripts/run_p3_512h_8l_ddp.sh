#!/bin/bash
# Phase 3 Experiment: 512H × 8L with DDP + Optimal Batch Sizes
#
# DDP Configuration for 2 GPUs (RTX 4090 + 3090 Ti):
#   - Per-GPU batch: 8
#   - Gradient accumulation: 4 steps
#   - Effective batch per GPU: 32
#   - DDP world size: 2
#   - Effective global batch: 64 tokens/step
#
# Compute: ~12-15 GPU hours total (2× speedup vs single GPU ~25h)
# Expected: Model scaling (512H×8L) should beat 256H×4L baseline (4.31)

set -e

CONFIG="config/milestones/p3_optimized_512h_8l_ddp.toml"
NPROC_PER_NODE=2

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 PHASE 3: MODEL SCALING TEST (512H × 8L) — DDP MODE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Configuration:"
echo "   Model:     512H × 8L (110M params)"
echo "   Data:      10M interleaved tokens"
echo "   Steps:     5,000"
echo "   GPUs:      2 (RTX 4090 + 3090 Ti)"
echo "   Batch:     32 global effective (8 per GPU × 2 accum × 2 GPUs)"
echo "   Compute:   ~12-15 GPU hours (2× speedup)"
echo ""
echo "🎯 Hypothesis:"
echo "   If loss < 4.1: Model capacity was bottleneck → scale more"
echo "   If loss ≥ 4.2: Model size not limiting → focus on data/LR"
echo ""
echo "📝 Output:"
echo "   Directory: outputs/p3-optimized-512h-8l-ddp/"
echo "   Metrics:   loss_curve.csv (aggregated from all ranks)"
echo "   Logs:      training output (main process only)"
echo ""
echo "⏱️  Starting: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo "❌ ERROR: Config not found: $CONFIG"
    exit 1
fi

# Activate environment
source .venv/bin/activate

# Verify environment
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ GPUs: {torch.cuda.device_count()}'); print(f'✓ Backend: nccl available={torch.distributed.is_available()}')" || exit 1

echo ""
echo "🔄 Launching DDP training on $NPROC_PER_NODE GPUs..."
echo ""

# Use torchrun for automatic DDP setup
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr="127.0.0.1" \
    --master_port=29501 \
    -m src.training.train \
    --config "$CONFIG"

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETE"
    echo ""
    echo "📊 Results:"
    echo "   Output dir: outputs/p3-optimized-512h-8l-ddp/"
    echo "   Loss curve: $(head -2 outputs/p3-optimized-512h-8l-ddp/loss_curve.csv | tail -1) ..."
    echo ""
    echo "📈 View results:"
    echo "   TensorBoard: tensorboard --logdir=outputs/p3-optimized-512h-8l-ddp"
    echo "   CSV tail:    tail -20 outputs/p3-optimized-512h-8l-ddp/loss_curve.csv"
else
    echo "❌ TRAINING FAILED (exit code: $EXIT_CODE)"
fi
echo "⏱️  Completed: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $EXIT_CODE
