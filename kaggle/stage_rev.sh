#!/usr/bin/env bash
# Revision experiments (reviewer response; secondary/exploratory). Two GPUs in parallel.
# New fits stop starting REV_HOURS after launch, so predictions are always exported.
set -uo pipefail
cd /kaggle/working/repo
export MMONC_DEADLINE=$(( $(date +%s) + ${REV_HOURS_X100:-1000} * 36 ))
P=/kaggle/working/prepared2; W=/kaggle/working/prepared3; O=/kaggle/working/rev
mkdir -p $O

# Round-1 per-patient predictions (per-collection paired differences, decision curves):
# copy them if a previous output is attached; otherwise re-score the frozen run-1 models (deterministic).
R1=$(find /kaggle/input -name predictions_external.csv -path '*run1*' 2>/dev/null | head -1)
if [ -n "$R1" ]; then
  echo "round-1 predictions found: $R1"
  cp "$R1" $O/r1_predictions_external.csv
  cp "$(dirname "$R1")/predictions_internal_oof.csv" $O/r1_predictions_internal_oof.csv 2>/dev/null || true
else
  echo "round-1 predictions not attached; re-scoring frozen run-1 models"
  ( bash kaggle/stage_external.sh > $O/_r1_rescore.log 2>&1 \
    && cp /kaggle/working/run1/predictions_external.csv $O/r1_predictions_external.csv \
    && cp /kaggle/working/run1/predictions_internal_oof.csv $O/r1_predictions_internal_oof.csv ) || echo "R1 RESCORE FAILED"
  tail -3 $O/_r1_rescore.log 2>/dev/null; rm -f $O/_r1_rescore.log
fi

python -m mmonc.prep2 --out $P --workers 4 | tail -4 || echo "PREP2 FAILED"
python -m mmonc.prep3 --out $W --workers 4 | tail -3 || echo "PREP3 FAILED"

( export CUDA_VISIBLE_DEVICES=0
  python -u -m mmonc.pipeline3 --data $P --out $O --exp devext --models carfm_k --tag g0a || echo "g0a FAILED"
  python -u -m mmonc.pipeline3 --data $P --whole $W --out $O --exp devext --models wb_fusion,wb_img --tag g0b || echo "g0b FAILED"
  python -u -m mmonc.pipeline3 --data $P --out $O --exp lc --models image_only_k --tag g0c || echo "g0c FAILED" ) &
( export CUDA_VISIBLE_DEVICES=1
  python -u -m mmonc.pipeline3 --data $P --out $O --exp dino --tag g1a || echo "g1a FAILED"
  python -u -m mmonc.pipeline3 --data $P --out $O --exp devext --models carfm_25d --tag g1b || echo "g1b FAILED"
  python -u -m mmonc.pipeline3 --data $P --out $O --exp posctl --models s25d --tag g1c || echo "g1c FAILED"
  python -u -m mmonc.pipeline3 --data $P --out $O --exp overfit --models image_only_k,s25d --tag g1d || echo "g1d FAILED"
  python -u -m mmonc.pipeline3 --data $P --out $O --exp posctl --models image_only_k --tag g1e || echo "g1e FAILED"
  python -u -m mmonc.pipeline3 --data $P --out $O --exp lc --models s25d --tag g1f || echo "g1f FAILED" ) &
wait
python -m mmonc.export2 --dir $O --extra $O/dino_g1a.json
rm -rf $P/*.npz $W/wb.npy $O/ck
