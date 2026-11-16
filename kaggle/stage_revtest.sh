#!/usr/bin/env bash
# Quick real-data test of the revision stage (40 cases, 1 epoch, 2 folds). Interactive session only.
set -uo pipefail
cd /kaggle/working/repo
P=/kaggle/working/t_prepared2; W=/kaggle/working/t_prepared3; O=/kaggle/working/t_rev
rm -rf $O; mkdir -p $O
find /kaggle/input -name predictions_external.csv 2>/dev/null | head -5
find /kaggle/input -name frozen.json 2>/dev/null | head -3
python -m mmonc.prep2 --out $P --workers 4 --limit 40 | tail -2 || echo "PREP2 FAILED"
python -m mmonc.prep3 --out $W --workers 4 --limit 40 | tail -2 || echo "PREP3 FAILED"
python -c "import numpy as np; a=np.load('$W/wb.npy'); print('wb', a.shape, a.mean(), (a>0).mean())"
Q="--quick --data $P --out $O"
python -u -m mmonc.pipeline3 $Q --exp devext --models carfm_k,carfm_25d --tag a 2>&1 | tail -3
python -u -m mmonc.pipeline3 $Q --whole $W --exp devext --models wb_fusion --tag b 2>&1 | tail -3
python -u -m mmonc.pipeline3 $Q --exp dino --tag c 2>&1 | tail -4
python -u -m mmonc.pipeline3 $Q --exp posctl --models s25d --tag d 2>&1 | tail -3
python -u -m mmonc.pipeline3 $Q --exp overfit --models image_only_k --tag e 2>&1 | tail -2
python -u -m mmonc.pipeline3 $Q --exp lc --models s25d --tag f 2>&1 | tail -2
nvidia-smi --query-gpu=memory.used --format=csv
ls $O; cat $O/dependency_a.csv
