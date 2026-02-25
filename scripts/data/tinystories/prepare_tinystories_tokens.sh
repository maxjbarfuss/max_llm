#!/usr/bin/env bash
set -euo pipefail

# Prepare TinyStories tokens with explicit cutoff mode and naming convention.
#
# Usage:
#   scripts/data/tinystories/prepare_tinystories_tokens.sh \
#     --source /mnt/d/dev/data/tinystories/train.txt \
#     --output-dir data/fast \
#     --size 5M \
#     --mode utf8 \
#     --cutoff article \
#     --cutoff-size 10M
#
# Cutoff modes:
#   article   - story-aware (uses <|endoftext|> boundaries)
#   row       - first N rows (requires --rows)
#   delimiter - document-aware using --cutoff-pattern regex
#   none      - no text cutoff; tokenize full text

PYTHON="${PYTHON:-/home/max/dev/max_llm/.venv/bin/python}"
SOURCE="/mnt/d/dev/data/tinystories/train.txt"
OUTPUT_DIR="data/fast"
SIZE="5M"
MODE="utf8"
VOCAB_SIZE="128"
CUTOFF_MODE="article"
CUTOFF_SIZE="2M"
CUTOFF_PATTERN=""
ROWS=""
FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --size) SIZE="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --vocab-size) VOCAB_SIZE="$2"; shift 2;;
    --cutoff) CUTOFF_MODE="$2"; shift 2;;
    --cutoff-size) CUTOFF_SIZE="$2"; shift 2;;
    --cutoff-pattern) CUTOFF_PATTERN="$2"; shift 2;;
    --rows) ROWS="$2"; shift 2;;
    --force) FORCE="true"; shift 1;;
    -h|--help)
      echo "Usage: $0 [--source path] [--output-dir dir] [--size 5M]" >&2
      echo "          [--mode codepoint|utf8|utf16|utf32] [--vocab-size 128]" >&2
      echo "          [--cutoff article|row|delimiter|none]" >&2
      echo "          [--cutoff-size 2M] [--cutoff-pattern REGEX] [--rows N] [--force]" >&2
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

# Derive output paths based on input parameters
SOURCE_NAME=$(basename "$SOURCE" .txt)
OUTPUT_DIR="$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

if [[ "$MODE" == "codepoint" ]]; then
  TOKEN_SUFFIX="cp${VOCAB_SIZE}"
else
  TOKEN_SUFFIX="$MODE"
fi

SUBSET_TOKENS_PATH="$OUTPUT_DIR/${SOURCE_NAME}_${SIZE}_tokens__${TOKEN_SUFFIX}.npy"
METADATA_PATH="$OUTPUT_DIR/${SOURCE_NAME}_${SIZE}_tokens__${TOKEN_SUFFIX}.meta.json"

# Build run_data_prep.py invocation
echo "Preparing TinyStories tokens..."
echo "  Source: $SOURCE"
echo "  Output: $SUBSET_TOKENS_PATH"
echo "  Target size: $SIZE"
echo "  Mode: $MODE"
echo "  Vocab size: $VOCAB_SIZE"
echo "  Cutoff mode: $CUTOFF_MODE (size: $CUTOFF_SIZE)"

# Minimal YAML config for run_data_prep
TEMP_CONFIG=$(mktemp)
trap "rm -f '$TEMP_CONFIG'" EXIT

cat > "$TEMP_CONFIG" <<EOF
paths:
  source: $SOURCE
  output_dir: $OUTPUT_DIR
  subset_tokens_path: $SUBSET_TOKENS_PATH
  metadata_path: $METADATA_PATH

workflow:
  normalize: false
  tokenize: true
  force: $FORCE

cutoff:
  mode: $CUTOFF_MODE
  dataset: tinystories
  size: $CUTOFF_SIZE
  rows: $([ -n "$ROWS" ] && echo "$ROWS" || echo "null")
  pattern: $([ -n "$CUTOFF_PATTERN" ] && echo "\"$CUTOFF_PATTERN\"" || echo "null")

subset:
  size: $SIZE

tokenizer:
  name: char
  mode: $MODE
  vocab_size: $VOCAB_SIZE
EOF

"$PYTHON" scripts/data/run_data_prep.py --config "$TEMP_CONFIG"

echo ""
echo "✓ TinyStories tokens ready: $SUBSET_TOKENS_PATH"
echo "✓ Metadata: $METADATA_PATH"
