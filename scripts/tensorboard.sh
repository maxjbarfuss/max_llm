#!/bin/bash
# Launch TensorBoard to visualize training experiments
#
# Usage:
#   bash scripts/tensorboard.sh                    # All runs
#   bash scripts/tensorboard.sh 512h_8l            # Specific run
#   bash scripts/tensorboard.sh 512h_8l 50m_data   # Multiple runs

set -e

RUNS="${@:-p3-optimized-512h-8l p3-optimized-50m-data p3-optimized-512h-50m}"

echo ""
echo "🔍 TensorBoard Dashboard"
echo ""
echo "   Logdir: outputs/"
echo "   Runs:   $RUNS"
echo ""
echo "   Launch command:"
echo "   tensorboard --logdir=outputs/"
echo ""
echo "   Access at: http://localhost:6006"
echo ""

# Start tensorboard
tensorboard --logdir=outputs/ --port=6006
