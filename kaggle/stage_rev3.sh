#!/usr/bin/env bash
# Third-review experiments (paper/analysis_plan_revision3.md). Preprocessing once, then two GPU queues in parallel.
set -uo pipefail
cd /kaggle/working/repo
export MMONC_DEADLINE=$(( $(date +%s) + ${REV_HOURS_X100:-1050} * 36 ))
export MMONC_REQUIRE_PRETRAINED=1   # no silent fallback to random initialisation
P=/kaggle/tmp/prep4; O=/kaggle/working/rev3
mkdir -p $P $O
echo "== commit $(git rev-parse HEAD)"
python -m mmonc.prep4 --out $P --workers 4 | tail -40 || echo "PREP4 FAILED"
df -h /kaggle/tmp /kaggle/working | tail -2
python -c "import torchvision; from mmonc.models import SliceEncoder; SliceEncoder(True)" || echo "WEIGHTS FAILED"
( export CUDA_VISIBLE_DEVICES=0
  python -u -m mmonc.pipeline5 --root $P --out $O --queue q0 --tag g0 || echo "g0 FAILED" ) &
( export CUDA_VISIBLE_DEVICES=1
  python -u -m mmonc.pipeline5 --root $P --out $O --queue q1 --tag g1 || echo "g1 FAILED" ) &
wait
tar -czf $O/preds.tar.gz -C $O preds
cp $P/audit.csv $P/kinetics_variants.csv $P/failures.csv $O/ 2>/dev/null
python -m mmonc.export2 --dir $O --extra $O/preds.tar.gz
rm -rf $P
