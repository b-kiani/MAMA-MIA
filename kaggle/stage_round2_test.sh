#!/usr/bin/env bash
# Round 2 shake-down on 160 cases: checks file layout, phases, kinetic features, GPU memory and speed.
set -uo pipefail
cd /kaggle/working/repo
P=/kaggle/working/prep2_test; O=/kaggle/working/round2_test
ROOT=$(dirname "$(find /kaggle/input -name clinical_and_imaging_info.xlsx | head -1)")
C=$(ls "$ROOT/images" | head -1); echo "example case $C:"; ls "$ROOT/images/$C" | head -12
python -m mmonc.prep2 --out $P --workers 4 --limit 160 || echo "PREP2 FAILED"
cat > /tmp/t.yaml <<'Y'
epochs: 2
folds: 2
control_folds: 2
patience: 2
controls: [hr]
Y
python -u -m mmonc.pipeline2 --data $P --out $O --exp devext --models tabular,image_only_k,carf_k,concat_k --tag g0 --config /tmp/t.yaml || echo "DEVEXT g0 FAILED"
python -u -m mmonc.pipeline2 --data $P --out $O --exp devext --models s25d,carf_25d --tag g1 --config /tmp/t.yaml || echo "DEVEXT g1 FAILED"
python -u -m mmonc.pipeline2 --data $P --out $O --exp controls --tag g1 --config /tmp/t.yaml || echo "CONTROLS FAILED"
python -u -m mmonc.pipeline2 --data $P --out $O --exp loco --models carf_k --tag g0 --config /tmp/t.yaml || echo "LOCO FAILED"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
python - <<'PY'
import pandas as pd, glob
for f in sorted(glob.glob('/kaggle/working/round2_test/*.csv')):
    d = pd.read_csv(f); print(f, d.shape, list(d.columns)[:12])
PY
rm -rf $P
