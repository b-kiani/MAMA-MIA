#!/usr/bin/env bash
# Run 2: reuse run-1 preprocessing and checkpoints (attached as input), score external collections ONCE.
set -euo pipefail
cd /kaggle/working/repo
SRC=$(dirname "$(find /kaggle/input -path '*run1/frozen.json' -o -path '*run1/table2_internal.csv' | head -1)")
PREP=$(dirname "$(find /kaggle/input -name table.csv -path '*prepared*' | head -1)")
echo "run1 from $SRC ; prepared from $PREP"
mkdir -p /kaggle/working/run1
cp -r "$SRC/checkpoints" /kaggle/working/run1/
python -u -m mmonc.pipeline --data "$PREP" --out /kaggle/working/run1 --config configs/kaggle_run.yaml --score-external
