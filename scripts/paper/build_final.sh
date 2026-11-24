#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source scripts/setup_env.sh >/dev/null
export APPTAINER_CACHEDIR="$CACHE_DIR/apptainer"
export APPTAINER_TMPDIR="$CACHE_DIR/apptainer_tmp"
RL_SMOKE_PYTHON="$SCRATCH_ROOT/env/rl-smoke/bin/python" bash scripts/rl/smoke_python.sh -m scripts.paper.prepare_followup
RL_SMOKE_PYTHON="$SCRATCH_ROOT/env/rl-plots/bin/python" bash scripts/rl/smoke_python.sh -m scripts.paper.prepare_final
TOOL_ROOT="$SCRATCH_ROOT/env/tectonic-0.15.0"
BUILD_ROOT="$CACHE_DIR/final-build"
mkdir -p "$BUILD_ROOT" "$CACHE_DIR/tectonic"
test -x "$TOOL_ROOT/tectonic"
apptainer exec --cleanenv --no-home \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" \
  --env "TOOL_ROOT=$TOOL_ROOT,BUILD_ROOT=$BUILD_ROOT,PROJECT_ROOT=$PROJECT_ROOT,XDG_CACHE_HOME=$CACHE_DIR/tectonic,TMPDIR=$TMPDIR,FONTCONFIG_FILE=$PROJECT_ROOT/scripts/paper/fonts.conf" \
  "$SCRATCH_ROOT/containers/lean-tools.sif" bash -c '
set -euo pipefail
"$TOOL_ROOT/tectonic" --keep-logs --outdir "$BUILD_ROOT" "$PROJECT_ROOT/docs/final/main.tex"
cp "$BUILD_ROOT/main.pdf" "$PROJECT_ROOT/docs/final/final_report.pdf"
'
/usr/bin/python3 -m scripts.paper.package_final
