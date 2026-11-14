#!/usr/bin/env bash
# Print compact results from an attached finish-run output to the log.
set -euo pipefail
FINAL=$(dirname "$(find /kaggle/input -path '*final/primary.csv' | head -1)")
echo "final from $FINAL"
python -u -m mmonc.export --final "$FINAL"
