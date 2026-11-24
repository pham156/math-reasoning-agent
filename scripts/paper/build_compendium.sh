#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source scripts/setup_env.sh >/dev/null
export APPTAINER_CACHEDIR="$CACHE_DIR/apptainer"
export APPTAINER_TMPDIR="$CACHE_DIR/apptainer_tmp"
bash scripts/rl/smoke_python.sh -m scripts.paper.prepare_compendium --snapshot_date "$(TZ=America/Indianapolis date +%F)"
TOOL_ROOT="$SCRATCH_ROOT/env/tectonic-0.15.0"
BUILD_ROOT="$CACHE_DIR/compendium-build"
mkdir -p "$BUILD_ROOT" "$CACHE_DIR/tectonic"
test -x "$TOOL_ROOT/tectonic"
apptainer exec --cleanenv --no-home \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" \
  --env "TOOL_ROOT=$TOOL_ROOT,BUILD_ROOT=$BUILD_ROOT,PROJECT_ROOT=$PROJECT_ROOT,XDG_CACHE_HOME=$CACHE_DIR/tectonic,TMPDIR=$TMPDIR,FONTCONFIG_FILE=$PROJECT_ROOT/scripts/paper/fonts.conf" \
  "$SCRATCH_ROOT/containers/lean-tools.sif" bash -c '
set -euo pipefail
"$TOOL_ROOT/tectonic" --keep-logs --outdir "$BUILD_ROOT" "$PROJECT_ROOT/docs/results_compendium/main.tex"
"$TOOL_ROOT/tectonic" --keep-logs --outdir "$BUILD_ROOT" "$PROJECT_ROOT/docs/results_compendium/all_results.tex"
cp "$BUILD_ROOT/all_results.pdf" "$PROJECT_ROOT/docs/results_compendium/completed_results.pdf"
'
