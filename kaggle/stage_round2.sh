#!/usr/bin/env bash
# Round 2 (secondary analyses; plan in paper/analysis_plan_round2.md). Two GPUs in parallel.
# New fits stop starting 10 h 15 min after launch, so predictions are always exported.
set -uo pipefail
cd /kaggle/working/repo
export MMONC_DEADLINE=$(( $(date +%s) + ${R2_HOURS_X100:-1025} * 36 ))
P=/kaggle/working/prepared2; O=/kaggle/working/round2
python -m mmonc.prep2 --out $P --workers 4 || echo "PREP2 FAILED"
python -u -m mmonc.pipeline2 --data $P --out $O --exp devext --models tabular --tag tab || echo "TABULAR FAILED"
( export CUDA_VISIBLE_DEVICES=0
  python -u -m mmonc.pipeline2 --data $P --out $O --exp devext --models image_only_k,carf_k --tag g0 || echo "g0 devext FAILED"
  python -u -m mmonc.pipeline2 --data $P --out $O --exp loco --models image_only_k,carf_k --tag g0 || echo "g0 loco FAILED"
  python -u -m mmonc.pipeline2 --data $P --out $O --exp devext --models concat_k --tag g0b || echo "g0b FAILED" ) &
( export CUDA_VISIBLE_DEVICES=1
  python -u -m mmonc.pipeline2 --data $P --out $O --exp devext --models s25d --tag g1 || echo "g1 devext FAILED"
  python -u -m mmonc.pipeline2 --data $P --out $O --exp controls --tag g1 || echo "g1 controls FAILED"
  python -u -m mmonc.pipeline2 --data $P --out $O --exp devext --models carf_25d --tag g1b || echo "g1b FAILED" ) &
wait
python -m mmonc.export2 --dir $O --samples-from $P --extra $P/table.csv $P/kinetics.csv $P/failures.csv
rm -rf $P/*.npz $O/ck
