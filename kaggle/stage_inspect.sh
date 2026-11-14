#!/usr/bin/env bash
# Inspect the attached MAMA-MIA copy: layout, table columns, counts vs the dataset paper.
set -euo pipefail
python - <<'EOF'
import glob, os, json
import pandas as pd
roots = sorted(glob.glob("/kaggle/input/**/clinical_and_imaging_info.xlsx", recursive=True))
print("tables found:", roots)
root = os.path.dirname(roots[0])
print("ROOT", root, sorted(os.listdir(root)))
tb = pd.read_excel(roots[0])
print("shape", tb.shape)
print("COLUMNS", list(tb.columns))
print(tb.head(3).T.to_string()[:4000])
dcol = [c for c in tb.columns if c.lower() in ("dataset", "collection")][0]
pcol = [c for c in tb.columns if c.lower() == "pcr"][0]
print("\nPER COLLECTION (paper: ISPY2 980 316/664, DUKE 291 64/216, ISPY1 171 49/118, NACT 64 11/53)")
print(tb.groupby(dcol)[pcol].agg(n="size", pcr1=lambda s: (s == 1).sum(), pcr0=lambda s: (s == 0).sum(), na=lambda s: s.isna().sum()))
for c in tb.columns:
    if tb[c].nunique() <= 8:
        print(f"{c}: {tb[c].value_counts(dropna=False).to_dict()}")
case = sorted(os.listdir(os.path.join(root, "images")))[0]
print("\nIMAGE FOLDER", case, sorted(os.listdir(os.path.join(root, "images", case))))
for d in sorted(os.listdir(os.path.join(root, "segmentations"))):
    p = os.path.join(root, "segmentations", d)
    print("SEG", d, len(os.listdir(p)), sorted(os.listdir(p))[:3])
pi = sorted(glob.glob(os.path.join(root, "patient_info_files", "*")))
print("\nPATIENT INFO", len(pi), pi[0])
print(open(pi[0]).read()[:2500])
for f in ["kfold_splits.csv", "train_test_splits.csv"]:
    s = pd.read_csv(os.path.join(root, f)); print("\n", f, s.shape, list(s.columns)); print(s.head(3).to_string())
import SimpleITK as sitk
for f in sorted(os.listdir(os.path.join(root, "images", case)))[:3]:
    im = sitk.ReadImage(os.path.join(root, "images", case, f))
    print(f, im.GetSize(), [round(x, 2) for x in im.GetSpacing()])
EOF
