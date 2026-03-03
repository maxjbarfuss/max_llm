#!/bin/bash
# Re-run 50m_data experiment with proper 50M dataset

set -e

cd /home/max/dev/max_llm
source .venv/bin/activate

echo "=================================================="
echo "50m_data RE-RUN: Wait for data_prep → Mix → Train"
echo "=================================================="
echo ""

# Step 1: Wait for data_prep to finish
echo "[1/4] Waiting for data_prep to complete..."
while [ ! -f "data/fast/webtext_15m_bpe_gpt2.npy" ]; do
    echo "  Waiting for OpenWebText tokenization..."
    progress=$(tail -1 outputs/data_prep_github_webtext.log 2>/dev/null | grep -oP '\d+\.\d+(?=%)' || echo "0")
    echo "    Progress: ${progress}%"
    sleep 30
done
echo "✓ Data prep complete"
echo ""

# Step 2: Mix datasets
echo "[2/4] Mixing datasets (wiki + stories + webtext)..."
if python scripts/data/mix_for_50m_rerun.py; then
    echo "✓ Mixed successfully"
else
    echo "✗ Mixing failed"
    exit 1
fi
echo ""

# Step 3: Update config to use new mixed dataset
echo "[3/4] Updating config..."
sed -i 's|dataset_path = "data/fast/interleaved_wikitext_tinystories_50m_pages_bpe_gpt2.npy"|dataset_path = "data/fast/interleaved_mixed_50m_tokens_bpe_gpt2.npy"|' \
    config/milestones/p3_optimized_50m_data.toml
echo "✓ Config updated"
echo ""

# Step 4: Clean old output and launch
echo "[4/4] Launching training..."
rm -rf outputs/p3-optimized-50m-data
bash scripts/run_p3_50m_data.sh

echo ""
echo "=================================================="
echo "✓ 50m_data re-run launched with proper 50M data"
echo "=================================================="
