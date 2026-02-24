#!/usr/bin/env bash
set -euo pipefail

# Prepare WikiText tokens with explicit cutoff mode and naming convention.
#
# Usage:
#   scripts/data/wikitext-103/prepare_wikitext_tokens.sh \
#     --source /mnt/d/dev/data/wikitext-103-raw/train.txt \
#     --output-dir data/fast \
#     --size 5M \
#     --mode utf8 \
#     --cutoff article \
#     --cutoff-size 10M
#
# Cutoff modes:
#   article   - document-aware (WikiText headers)
#   row       - first N rows (requires --rows)
#   delimiter - document-aware using --cutoff-pattern regex
#   none      - no text cutoff; tokenize full normalized text

PYTHON="${PYTHON:-/home/max/dev/max_llm/.venv/bin/python}"
SOURCE="/mnt/d/dev/data/wikitext-103-raw/train.txt"
OUTPUT_DIR="data/fast"
SIZE="5M"
MODE="utf8"
VOCAB_SIZE="128"
CUTOFF_MODE="article"
CUTOFF_SIZE="10M"
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
      echo "          [--cutoff-size 10M] [--cutoff-pattern REGEX] [--rows N] [--force]" >&2
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "codepoint" && "$MODE" != "utf8" && "$MODE" != "utf16" && "$MODE" != "utf32" ]]; then
  echo "Invalid --mode: $MODE" >&2
  exit 1
fi

if [[ "$CUTOFF_MODE" != "article" && "$CUTOFF_MODE" != "row" && "$CUTOFF_MODE" != "delimiter" && "$CUTOFF_MODE" != "none" ]]; then
  echo "Invalid --cutoff: $CUTOFF_MODE" >&2
  exit 1
fi

if [[ "$CUTOFF_MODE" == "row" && -z "$ROWS" ]]; then
  echo "--rows is required when --cutoff row" >&2
  exit 1
fi

if [[ "$CUTOFF_MODE" == "delimiter" && -z "$CUTOFF_PATTERN" ]]; then
  echo "--cutoff-pattern is required when --cutoff delimiter" >&2
  exit 1
fi

SOURCE_PATH="$(realpath "$SOURCE")"
SOURCE_DIR="$(dirname "$SOURCE_PATH")"
SOURCE_BASE="$(basename "$SOURCE_PATH")"
SOURCE_STEM="${SOURCE_BASE%.*}"

NORMALIZED_PATH="$SOURCE_DIR/${SOURCE_STEM}_normalized.txt"

if [[ "$MODE" == "codepoint" ]]; then
  TOKEN_SUFFIX="cp${VOCAB_SIZE}"
else
  TOKEN_SUFFIX="$MODE"
fi

if [[ "$CUTOFF_MODE" == "none" ]]; then
  TOKEN_CACHE_PATH="$SOURCE_DIR/${SOURCE_STEM}_normalized_tokens__${TOKEN_SUFFIX}.npy"
else
  TOKEN_CACHE_PATH="$OUTPUT_DIR/${SOURCE_STEM}_subset__${CUTOFF_MODE}_${CUTOFF_SIZE}_tokens__${TOKEN_SUFFIX}.npy"
fi
mkdir -p "$OUTPUT_DIR"

TEXT_SUBSET_PATH=""
if [[ "$CUTOFF_MODE" != "none" ]]; then
  TEXT_SUBSET_PATH="$OUTPUT_DIR/${SOURCE_STEM}_subset__${CUTOFF_MODE}_${CUTOFF_SIZE}.txt"
fi

SUBSET_TOKENS_PATH="$OUTPUT_DIR/${SOURCE_STEM}_${SIZE}_tokens__${TOKEN_SUFFIX}.npy"
META_PATH="$OUTPUT_DIR/${SOURCE_STEM}_${SIZE}_tokens__${TOKEN_SUFFIX}.meta.json"

if [[ ! -f "$NORMALIZED_PATH" || "$FORCE" == "true" ]]; then
  echo "Normalizing: $SOURCE_PATH -> $NORMALIZED_PATH"
  "$PYTHON" src/data/datasets/wikitext/normalize.py "$SOURCE_PATH" "$NORMALIZED_PATH"
else
  echo "Using cached normalized text: $NORMALIZED_PATH"
fi

if [[ "$CUTOFF_MODE" == "none" ]]; then
  echo "No text cutoff requested"
elif [[ -f "$TEXT_SUBSET_PATH" && "$FORCE" == "false" ]]; then
  echo "Using cached text subset: $TEXT_SUBSET_PATH"
elif [[ "$CUTOFF_MODE" == "article" ]]; then
  echo "Extracting text subset (article boundaries): $CUTOFF_SIZE -> $TEXT_SUBSET_PATH"
  "$PYTHON" -m src.data.pipeline.extract_text \
    --input "$NORMALIZED_PATH" \
    --output "$TEXT_SUBSET_PATH" \
    --size "$CUTOFF_SIZE"
elif [[ "$CUTOFF_MODE" == "delimiter" ]]; then
  echo "Extracting text subset (delimiter): $CUTOFF_PATTERN"
  "$PYTHON" -m src.data.pipeline.extract_text \
    --input "$NORMALIZED_PATH" \
    --output "$TEXT_SUBSET_PATH" \
    --size "$CUTOFF_SIZE" \
    --boundary-pattern "$CUTOFF_PATTERN"
elif [[ "$CUTOFF_MODE" == "row" ]]; then
  echo "Extracting text subset (rows): $ROWS -> $TEXT_SUBSET_PATH"
  head -n "$ROWS" "$NORMALIZED_PATH" > "$TEXT_SUBSET_PATH"
fi

TOKEN_INPUT_PATH="$NORMALIZED_PATH"
if [[ -n "$TEXT_SUBSET_PATH" ]]; then
  TOKEN_INPUT_PATH="$TEXT_SUBSET_PATH"
fi

if [[ ! -f "$TOKEN_CACHE_PATH" || "$FORCE" == "true" ]]; then
  echo "Tokenizing: $TOKEN_INPUT_PATH -> $TOKEN_CACHE_PATH"
  TOKEN_ARGS=(
    -m src.data.pipeline.tokenize
    --input "$TOKEN_INPUT_PATH"
    --output "$TOKEN_CACHE_PATH"
    --tokenizer char
    --mode "$MODE"
  )
  if [[ "$MODE" == "codepoint" ]]; then
    TOKEN_ARGS+=(--vocab-size "$VOCAB_SIZE")
  fi
  "$PYTHON" "${TOKEN_ARGS[@]}"
else
  echo "Using cached token file: $TOKEN_CACHE_PATH"
fi

if [[ ! -f "$SUBSET_TOKENS_PATH" || "$FORCE" == "true" ]]; then
  echo "Extracting token subset: $SIZE -> $SUBSET_TOKENS_PATH"
  "$PYTHON" -m src.data.pipeline.extract_tokens \
    --input "$TOKEN_CACHE_PATH" \
    --output "$SUBSET_TOKENS_PATH" \
    --size "$SIZE" \
    -v
else
  echo "Using existing token subset: $SUBSET_TOKENS_PATH"
fi

cat > "$META_PATH" <<EOF
{
  "source": "$SOURCE_PATH",
  "normalized": "$NORMALIZED_PATH",
  "cutoff_mode": "$CUTOFF_MODE",
  "cutoff_size": "$CUTOFF_SIZE",
  "cutoff_rows": "${ROWS}",
  "cutoff_pattern": "${CUTOFF_PATTERN}",
  "subset_size": "$SIZE",
  "tokenizer": "char",
  "tokenizer_mode": "$MODE",
  "tokenizer_vocab_size": "${VOCAB_SIZE}",
  "token_input": "$TOKEN_INPUT_PATH",
  "token_cache": "$TOKEN_CACHE_PATH",
  "subset_tokens": "$SUBSET_TOKENS_PATH",
  "generated_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "Metadata written: $META_PATH"
