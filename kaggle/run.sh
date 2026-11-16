#!/usr/bin/env bash
# Kaggle entry point. In a notebook cell:
#   !cd /kaggle/working && (git -C repo pull -q 2>/dev/null || git clone -q --depth 1 https://github.com/b-kiani/mama-mia repo) && bash repo/kaggle/run.sh <stage>
set -euo pipefail
STAGE="${1:-smoke}"
REPO=/kaggle/working/repo
cd "$REPO"
# The notebook cell is hard to edit remotely; a committed kaggle/OVERRIDE_STAGE file selects the stage instead.
if [ -f "$REPO/kaggle/OVERRIDE_STAGE" ]; then
  echo "== stage argument '$STAGE' overridden by kaggle/OVERRIDE_STAGE"
  STAGE=$(tr -d '[:space:]' < "$REPO/kaggle/OVERRIDE_STAGE")
fi
echo "== commit $(git rev-parse --short HEAD) | stage $STAGE | $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
pip install -q -e . lifelines 2>&1 | grep -v -i "warning\|already" | tail -2 || true

case "$STAGE" in
  smoke)
    python -m pytest -q 2>&1 | tail -3
    python -m mmonc.pipeline --synthetic --out /kaggle/working/synthetic_smoke --n 240 2>&1 | tail -6
    ;;
  *)
    bash "$REPO/kaggle/stage_${STAGE}.sh"
    ;;
esac
echo "== done $STAGE $(date -u +%FT%TZ)"
