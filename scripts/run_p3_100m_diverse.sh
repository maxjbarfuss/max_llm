#!/bin/bash
# Phase 3 Experiment: 100M Mixed Diverse Dataset Test
#
# Purpose: If 50M WikiText+TinyStories beats baseline, test if
#          ADDING complementary sources (ArXiv, StackExchange, Code)
#          provides further improvement via data diversity.
#
# Compute: ~50 GPU hours (H100)
# Goal:    Beat 50m_data final loss + aim for 3.3-3.5
#
# Prerequisites:
#   1. 50m_data experiment must complete successfully (loss < 3.9)
#   2. Run: python scripts/data/prepare_complementary.py --source all
#      (prepares ArXiv, Stack Exchange, and Code datasets)
#   3. Mix datasets: python scripts/data/mix_interleaved_pages.py ...
#      (creates 100m_diverse file from prepared sources)

set -e

# Activate venv
source .venv/bin/activate

CONFIG="config/milestones/p3_optimized_100m_diverse.toml"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 PHASE 3: DATA DIVERSITY TEST (100M mixed tokens)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Configuration:"
echo "   Model:     256H × 4L (same as baseline)"
echo "   Data:      100M diverse tokens (40% wiki, 20% stories, 40% other)"
echo "   Sources:   WikiText + TinyStories + ArXiv + StackExchange + Code"
echo "   Steps:     15,000 (more steps for more data)"
echo "   Batch:     32 effective (same as 50m_data)"
echo "   Compute:   ~50 GPU hours"
echo ""
echo "🎯 Hypothesis:"
echo "   If loss < 3.3: Data diversity provides improvement"
echo "   If loss 3.3-3.5: Marginal improvement, diminishing returns"
echo "   If loss ≥ 3.5: Architecture/model size may be limiting factor"
echo ""
echo "⏱️  Starting: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo "❌ ERROR: Config not found: $CONFIG"
    exit 1
fi

# Check for required data file
DATA_FILE="data/fast/interleaved_wikitext_tinystories_arxiv_stackexchange_code_100m_tokens_bpe_gpt2.npy"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    echo ""
    echo "To prepare the 100M diverse dataset:"
    echo "  1. python scripts/data/prepare_complementary.py --source all"
    echo "  2. python scripts/data/mix_interleaved_pages.py --ratio 0.4 0.2 0.2 0.2 ..."
    echo ""
    exit 1
fi

# Verify torch
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || exit 1

# Run training
python -m src.training.train --config "$CONFIG"

echo ""
echo "✓ Training complete!"
echo "  Results: outputs/p3-optimized-100m-mixed-diverse/loss_curve.csv"
echo ""
echo "📊 Next decision:"
echo "  If loss < 3.3: Data diversity works → proceed to model scaling with data"
echo "  If loss ≥ 3.5: Diminishing returns → investigate architecture changes"
