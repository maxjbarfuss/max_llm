#!/bin/bash
# Launch p3-optimized-768h-8l with DDP on 2 GPUs
# 768H×8L model = ~200M params
# Optimized: batch_size=8 per GPU, grad_accum=4 → eff batch=64 across 2 GPUs
# Expected: ~35-40 GPU hours, target loss < 3.9

set -e
cd "$(dirname "$0")/.."

echo "=== Phase 3 Optimization: 768H×8L with DDP ==="
echo "GPUs: RTX 4090 + 3090 Ti"
echo "Config: config/milestones/p3_optimized_768h_8l.toml"
echo ""

# Activate venv
source .venv/bin/activate

# Verify config
if [ ! -f "config/milestones/p3_optimized_768h_8l.toml" ]; then
    echo "ERROR: Config not found"
    exit 1
fi

# Clear old output if exists
if [ -d "outputs/p3-optimized-768h-8l" ]; then
    echo "Removing previous run outputs..."
    rm -rf outputs/p3-optimized-768h-8l
fi

# Verify PyTorch + GPUs
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" || exit 1

# Launch with torchrun DDP
echo "Starting training with DDP (2 GPUs)..."
export PYTHONPATH="$(pwd):$PYTHONPATH"
torchrun --nproc_per_node=2 -m src.training.train \
    --config config/milestones/p3_optimized_768h_8l.toml

echo ""
echo "✓ Training complete"
echo "Outputs: outputs/p3-optimized-768h-8l/"
echo ""
echo "Monitor with (in separate terminal):"
echo "  tensorboard --logdir=outputs/p3-optimized-768h-8l"
