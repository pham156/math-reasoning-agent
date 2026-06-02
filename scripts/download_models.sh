#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/setup_env.sh"

echo "Download locations:"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  SCRATCH_ROOT=$SCRATCH_ROOT"
echo "  MODEL_DIR=$MODEL_DIR"
echo "  HF_HOME=$HF_HOME"

case "$SCRATCH_ROOT" in
  /scratch/gilbreth/*) ;;
  *)
    echo "ERROR: SCRATCH_ROOT is not under /scratch/gilbreth: $SCRATCH_ROOT" >&2
    exit 1
    ;;
esac

echo "Confirmed model downloads point to scratch."

if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli download)
else
  echo "ERROR: neither huggingface-cli nor hf was found. Install huggingface_hub in the active environment." >&2
  exit 1
fi

models=(
  "Qwen/Qwen2.5-Math-7B-Instruct"
  "Qwen/Qwen2.5-Math-7B"
  "deepseek-ai/deepseek-math-7b-instruct"
  "sentence-transformers/all-MiniLM-L6-v2"
)

for model_id in "${models[@]}"; do
  local_name=${model_id#*/}
  target="$MODEL_DIR/$local_name"
  echo "Downloading $model_id -> $target"
  mkdir -p "$target"
  "${HF_CLI[@]}" "$model_id" \
    --local-dir "$target"
done

echo "All model downloads completed under $MODEL_DIR"
