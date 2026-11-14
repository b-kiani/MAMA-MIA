#!/usr/bin/env bash
# Compact inspection of the attached MAMA-MIA copy (narrow output for screen reading).
set -euo pipefail
python -u - <<'EOF'
import glob, os, json
import pandas as pd
pd.set_option("display.width", 52)
X = glob.glob("/kaggle/input/**/clinical_and_imaging_info.xlsx", recursive=True)[0]
R = os.path.dirname(X)
tb = pd.read_excel(X)
os.makedirs("/kaggle/working/inspect", exist_ok=True)
tb.to_csv("/kaggle/working/inspect/clinical.csv", index=False)
print("N", len(tb))
g = tb.groupby("dataset")["pcr"]
for k, s in g:
    print(f"{k:6s} n={len(s):4d} pcr1={(s==1).sum():4d} pcr0={(s==0).sum():4d} na={s.isna().sum()}")
print("paper: ISPY2 980 316/664 DUKE 291 64/216")
print("       ISPY1 171 49/118  NACT 64 11/53")
for c in ["age","hr","her2","menopause","multifocal_cancer","tumor_subtype","ethnicity","field_strength","manufacturer","view","site","nac_agent","days_to_follow_up","days_to_recurrence"]:
    s = tb[c]
    miss = s.isna().groupby(tb.dataset).mean().round(2).to_dict()
    u = s.dropna().unique()
    show = (list(map(str, u[:4])) if len(u) > 4 else list(map(str, u)))
    print(f"{c}: k={len(u)} miss={miss}")
    print("   ", ", ".join(x[:14] for x in show)[:48])
print("acq_times ex:", str(tb.acquisition_times.iloc[0])[:44])
print("acq_times ex:", str(tb[tb.dataset=='ISPY2'].acquisition_times.iloc[0])[:44])
print("acq_date ex:", str(tb.acquisition_date.dropna().iloc[0])[:30])
print("site per ISPY2:", tb[tb.dataset=="ISPY2"].site.nunique())
print("phases:", tb.num_phases.value_counts().to_dict())
import sys; sys.path.insert(0, "/kaggle/working/repo/src")
from mmonc.data import read_image
for d in ["DUKE", "ISPY1", "ISPY2", "NACT"]:
    cid = tb[tb.dataset == d].patient_id.iloc[0]
    fs = sorted(os.listdir(os.path.join(R, "images", cid)))
    print(cid, len(fs), fs[0])
    f0 = os.path.join(R, "images", cid, fs[0])
    im = read_image(f0)
    print("  magic", open(f0, "rb").read(2).hex(), im.GetSize(), [round(x, 2) for x in im.GetSpacing()])
for d in sorted(os.listdir(os.path.join(R, "segmentations"))):
    p = os.path.join(R, "segmentations", d)
    print("SEG", d, len(os.listdir(p)), sorted(os.listdir(p))[0])
pi = sorted(glob.glob(os.path.join(R, "patient_info_files", "*")))
print("PI", len(pi), os.path.basename(pi[0]))
j = json.load(open(pi[0]))
def keys(o, pre=""):
    for k, v in (o.items() if isinstance(o, dict) else []):
        print(" ", pre + k, ":", str(v)[:30] if not isinstance(v, dict) else "")
        if isinstance(v, dict): keys(v, pre + "  ")
keys(j)
for f in ["kfold_splits.csv", "train_test_splits.csv"]:
    s = pd.read_csv(os.path.join(R, f)); print(f, s.shape, list(s.columns)[:5])
EOF
