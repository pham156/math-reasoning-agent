#!/usr/bin/env bash

# Source this file before running project commands:
#   source scripts/setup_env.sh

export PROJECT_ROOT=/home/pham156/math-reasoning-agent
export SCRATCH_ROOT=/scratch/gilbreth/pham156/math-reasoning-agent

if [[ "$SCRATCH_ROOT" != /scratch/gilbreth/* ]]; then
  echo "ERROR: SCRATCH_ROOT must be under /scratch/gilbreth, got: $SCRATCH_ROOT" >&2
  return 1 2>/dev/null || exit 1
fi

export MODEL_DIR=$SCRATCH_ROOT/models
export DATA_DIR=$SCRATCH_ROOT/data
export OUTPUT_DIR=$SCRATCH_ROOT/outputs
export CACHE_DIR=$SCRATCH_ROOT/cache
export LOG_DIR=$PROJECT_ROOT/logs

export HF_HOME=$CACHE_DIR/huggingface
export TRANSFORMERS_CACHE=$CACHE_DIR/huggingface/transformers
export HF_DATASETS_CACHE=$CACHE_DIR/huggingface/datasets
export TORCH_HOME=$CACHE_DIR/torch
export WANDB_DIR=$LOG_DIR/wandb
export TMPDIR=$CACHE_DIR/tmp

export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}

mkdir -p \
  "$MODEL_DIR" \
  "$DATA_DIR" \
  "$OUTPUT_DIR" \
  "$CACHE_DIR/huggingface/transformers" \
  "$CACHE_DIR/huggingface/datasets" \
  "$CACHE_DIR/torch" \
  "$CACHE_DIR/tmp" \
  "$LOG_DIR/wandb" \
  "$PROJECT_ROOT/notebooks" \
  "$PROJECT_ROOT/env"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "SCRATCH_ROOT=$SCRATCH_ROOT"
echo "MODEL_DIR=$MODEL_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "TORCH_HOME=$TORCH_HOME"
echo "LOG_DIR=$LOG_DIR"
echo "Code/logs live in home; models, data, outputs, caches, and temp files live in scratch."
