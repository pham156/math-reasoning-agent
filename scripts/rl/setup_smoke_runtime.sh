#!/usr/bin/env bash
set -euo pipefail
cd /home/pham156/math-reasoning-agent
source scripts/setup_env.sh
export APPTAINER_CACHEDIR="$CACHE_DIR/apptainer"
export APPTAINER_TMPDIR="$CACHE_DIR/apptainer_tmp"
mkdir -p "$SCRATCH_ROOT/containers" "$SCRATCH_ROOT/env" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$CACHE_DIR/pip"
IMAGE="$SCRATCH_ROOT/containers/rl-smoke-python311.sif"
if [[ ! -f "$IMAGE" ]]; then
  if [[ -f /scratch/gilbreth/pham156/pdf_tools/python311.sif ]]; then
    cp /scratch/gilbreth/pham156/pdf_tools/python311.sif "$IMAGE"
  else
    apptainer build "$IMAGE" containers/rl-smoke.def
  fi
fi
if [[ ! -x "$SCRATCH_ROOT/env/rl-smoke/bin/python" ]]; then
  apptainer exec --cleanenv --no-home --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" "$IMAGE" python -m venv --copies "$SCRATCH_ROOT/env/rl-smoke"
fi
bash scripts/rl/smoke_python.sh -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu118
bash scripts/rl/smoke_python.sh -m pip install -r containers/rl-smoke-requirements.txt
bash scripts/rl/smoke_python.sh -c 'import torch, datasets, transformers, math_verify; from trl import GRPOConfig, GRPOTrainer; print("Smoke runtime imports passed", torch.__version__, transformers.__version__)'
echo "Download target: $MODEL_DIR/Qwen2.5-Math-1.5B"
bash scripts/rl/smoke_python.sh -c 'from huggingface_hub import snapshot_download; from src.core.config import model_dir; snapshot_download("Qwen/Qwen2.5-Math-1.5B", local_dir=str(model_dir("Qwen2.5-Math-1.5B")), ignore_patterns=["*.bin", "*.onnx", "*.tflite", "*.msgpack"])'
