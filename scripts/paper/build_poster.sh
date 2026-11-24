#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
RL_SMOKE_PYTHON=/scratch/gilbreth/pham156/math-reasoning-agent/env/rl-smoke/bin/python bash scripts/rl/smoke_python.sh -m scripts.paper.prepare_followup
export RL_SMOKE_PYTHON=/scratch/gilbreth/pham156/math-reasoning-agent/env/rl-plots/bin/python
POSTER_DATE="$(TZ=America/Indianapolis date +%F)"
bash scripts/rl/smoke_python.sh -m scripts.paper.prepare_poster --snapshot_date "$POSTER_DATE"
