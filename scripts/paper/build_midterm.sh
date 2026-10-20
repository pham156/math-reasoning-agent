#!/usr/bin/env bash
set -euo pipefail
cd /home/pham156/math-reasoning-agent
source scripts/setup_env.sh >/dev/null
export APPTAINER_CACHEDIR="$CACHE_DIR/apptainer"
export APPTAINER_TMPDIR="$CACHE_DIR/apptainer_tmp"
TOOL_ROOT="$SCRATCH_ROOT/env/tectonic-0.15.0"
BUILD_ROOT="$CACHE_DIR/midterm-build"
mkdir -p "$TOOL_ROOT" "$BUILD_ROOT" "$CACHE_DIR/tectonic"
apptainer exec --cleanenv --no-home \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" \
  --env "TOOL_ROOT=$TOOL_ROOT,BUILD_ROOT=$BUILD_ROOT,PROJECT_ROOT=$PROJECT_ROOT,XDG_CACHE_HOME=$CACHE_DIR/tectonic,TMPDIR=$TMPDIR,FONTCONFIG_FILE=$PROJECT_ROOT/scripts/paper/fonts.conf" \
  "$SCRATCH_ROOT/containers/lean-tools.sif" bash -c '
set -euo pipefail
if [[ ! -x "$TOOL_ROOT/tectonic" ]]; then
  curl --fail --location --retry 3 https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz -o "$TOOL_ROOT/release.tar.gz"
  tar -xzf "$TOOL_ROOT/release.tar.gz" -C "$TOOL_ROOT" tectonic
fi
"$TOOL_ROOT/tectonic" --keep-logs --outdir "$BUILD_ROOT" "$PROJECT_ROOT/docs/midterm/main.tex"
cp "$BUILD_ROOT/main.pdf" "$PROJECT_ROOT/docs/midterm/midterm_update.pdf"
'
/usr/bin/python3 -c 'from zipfile import ZipFile; from pathlib import Path; root=Path("docs/midterm"); archive=ZipFile(root/"midterm_overleaf.zip", "w"); archive.write(root/"main.tex", "main.tex"); [archive.write(p, str(p.relative_to(root))) for p in sorted((root/"figures").glob("*.pdf"))]; archive.close(); print("Created PDF and Overleaf ZIP in", root.resolve())'
