#!/usr/bin/env bash
# Second-review experiments (paper/analysis_plan_revision2.md). Two GPUs in parallel; GPU-resident training.
set -uo pipefail
cd /kaggle/working/repo
export MMONC_DEADLINE=$(( $(date +%s) + ${REV_HOURS_X100:-1050} * 36 ))
P=/kaggle/working/prepared2; O=/kaggle/working/rev2
mkdir -p $O
python -m mmonc.prep2 --out $P --workers 4 | tail -4 || echo "PREP2 FAILED"
( export CUDA_VISIBLE_DEVICES=0
  python -u -m mmonc.pipeline4 --data $P --out $O --exp opt --configs base,long_auc,long_loss --tag h0a || echo "h0a FAILED" ) &
( export CUDA_VISIBLE_DEVICES=1
  python -u -m mmonc.pipeline4 --data $P --out $O --exp nested --models carf_k,carfm_k,carf_25d,carfm_25d --tag h1a || echo "h1a FAILED"
  python -u -m mmonc.pipeline4 --data $P --out $O --exp overfit --tag h1b || echo "h1b FAILED"
  python -u -m mmonc.pipeline4 --data $P --out $O --exp opt --configs fixed,small_loss --tag h1c || echo "h1c FAILED" ) &
wait
python -m mmonc.export2 --dir $O
rm -rf $P/*.npz
