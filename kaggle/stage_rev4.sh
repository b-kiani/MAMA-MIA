#!/usr/bin/env bash
# Fourth review: fully nested late fusion (seeds 0-4, regimen-held-out), one queue per GPU.
set -uo pipefail
cd /kaggle/working/repo
export MMONC_DEADLINE=$(( $(date +%s) + ${REV_HOURS_X100:-1100} * 36 ))
export MMONC_REQUIRE_PRETRAINED=1   # no silent fallback to random initialisation
P=/kaggle/tmp/prep4; O=/kaggle/working/rev4
mkdir -p $P $O
echo "== commit $(git rev-parse HEAD)"
python -m mmonc.prep4 --out $P --workers 4 | tail -40 || echo "PREP4 FAILED"
df -h /kaggle/tmp /kaggle/working | tail -2
python -c "import torchvision; from mmonc.models import SliceEncoder; SliceEncoder(True)" || echo "WEIGHTS FAILED"
( export CUDA_VISIBLE_DEVICES=0
  python -u -m mmonc.pipeline5 --root $P --out $O --queue q4a --tag k0 || echo "k0 FAILED" ) &
( export CUDA_VISIBLE_DEVICES=1
  python -u -m mmonc.pipeline5 --root $P --out $O --queue q4b --tag k1 || echo "k1 FAILED" ) &
wait
tar -czf $O/preds.tar.gz -C $O preds
cp $P/failures.csv $O/ 2>/dev/null
python -m mmonc.export2 --dir $O --extra $O/preds.tar.gz
rm -rf $P
