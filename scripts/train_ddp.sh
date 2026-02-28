#!/usr/bin/env bash
# Launch distributed training with torchrun (PyTorch Distributed Launcher)
#
# Usage:
#   ./scripts/train_ddp.sh <config_file> [num_gpus]
#
# Examples:
#   # Train on 2 GPUs
#   ./scripts/train_ddp.sh config/experiment_p3_optimized_2k.toml 2
#
#   # Train on all available GPUs
#   ./scripts/train_ddp.sh config/experiment_p3_optimized_2k.toml

set -e

# Parse arguments
CONFIG_FILE="${1:-config/experiment_p3_optimized_2k.toml}"
NUM_GPUS="${2:-2}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

echo "=== Multi-GPU Training with DDP ==="
echo "Config: $CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Launch with torchrun
# torchrun automatically sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    -m src.training.train \
    --config "$CONFIG_FILE" \
    --distributed

echo ""
echo "=== Training Complete ==="
