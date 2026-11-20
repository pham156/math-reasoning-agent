#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${PROJECT_ROOT:-$SCRIPT_DIR}"
source scripts/setup_env.sh >/dev/null
LEAN_ROOT="$SCRATCH_ROOT/lean"
export APPTAINER_CACHEDIR="$CACHE_DIR/apptainer"
export APPTAINER_TMPDIR="$CACHE_DIR/apptainer_tmp"
MODE=ro
ROOT_ARGS=()
if [[ "${1:-}" == --setup ]]; then
  MODE=rw
  shift
elif [[ -f "$SCRATCH_ROOT/containers/lean-mathlib.squashfs" ]]; then
  ROOT_ARGS=(--bind "$SCRATCH_ROOT/containers/lean-mathlib.squashfs:$LEAN_ROOT:image-src=/" --bind "$LEAN_ROOT/checks:$LEAN_ROOT/checks:ro")
fi
if [[ ${#ROOT_ARGS[@]} == 0 ]]; then
  ROOT_ARGS=(--bind "$LEAN_ROOT:$LEAN_ROOT:$MODE")
fi
exec apptainer exec --cleanenv --containall --no-home --no-mount cwd \
  "${ROOT_ARGS[@]}" \
  --pwd "$LEAN_ROOT/mathlib4" \
  --env "PATH=$LEAN_ROOT/lean-4.24.0-linux/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "ELAN_HOME=$LEAN_ROOT/cache/elan,LAKE_HOME=$LEAN_ROOT/cache/lake,XDG_CACHE_HOME=$LEAN_ROOT/cache/xdg" \
  "$SCRATCH_ROOT/containers/lean-tools.sif" "$@"
