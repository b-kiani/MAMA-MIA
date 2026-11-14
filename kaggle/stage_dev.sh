#!/usr/bin/env bash
# Run 1: preprocess all cases, train every model on ISPY2 (5-fold CV). External data NOT scored.
set -euo pipefail
cd /kaggle/working/repo
python -m mmonc.prep --out /kaggle/working/prepared --workers 4
python -u -m mmonc.pipeline --data /kaggle/working/prepared --out /kaggle/working/run1 --config configs/kaggle_run.yaml
python -u - <<'PY'
import pandas as pd, json
t = pd.read_csv("/kaggle/working/run1/table2_internal.csv")
pd.set_option("display.width", 60)
print(t[["model","auc","auc_lo","auc_hi","balanced_accuracy","slope","brier"]].round(3).to_string(index=False))
PY
