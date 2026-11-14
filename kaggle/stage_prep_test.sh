#!/usr/bin/env bash
# Preprocess a stratified sample of 20 cases to check speed and correctness.
set -euo pipefail
python -m mmonc.prep --out /kaggle/working/prep_test --limit 20 --workers 4
python -u - <<'PY'
import numpy as np, pandas as pd, glob
t = pd.read_csv("/kaggle/working/prep_test/table.csv"); r = pd.read_csv("/kaggle/working/prep_test/radiomics.csv")
print(t.groupby("collection").size().to_dict())
print(t[["age","hr","her2","menopausal","dt1","rfs_months","rfs_event"]].describe().loc[["count","mean"]].round(1).to_string())
z = np.load(sorted(glob.glob("/kaggle/working/prep_test/*.npz"))[0])
print("img", z["image"].shape, z["image"].dtype, "mask frac", round(float(z["mask"].mean()), 3))
print("sub in/out", round(float(z["image"][2][z["mask"]>0].mean()),2), round(float(z["image"][2][z["mask"]==0].mean()),2))
print("radiomics", r.shape)
PY
