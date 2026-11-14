#!/usr/bin/env bash
# Run 3: preprocess, then CARF ablations on GPU 0 and two extra CARF seeds on GPU 1, in parallel.
set -euo pipefail
cd /kaggle/working/repo
python -m mmonc.prep --out /kaggle/working/prepared --workers 4
P=/kaggle/working/prepared
( CUDA_VISIBLE_DEVICES=0 python -u -m mmonc.pipeline --data $P --out /kaggle/working/run3a --config configs/kaggle_run3a.yaml 2>&1 | sed 's/^/[A] /' ) &
( CUDA_VISIBLE_DEVICES=1 python -u -m mmonc.pipeline --data $P --out /kaggle/working/run3b --config configs/kaggle_run3b.yaml 2>&1 | sed 's/^/[B] /' && \
  CUDA_VISIBLE_DEVICES=1 python -u -m mmonc.pipeline --data $P --out /kaggle/working/run3c --config configs/kaggle_run3c.yaml 2>&1 | sed 's/^/[C] /' ) &
wait
ls -la /kaggle/working/run3a /kaggle/working/run3b /kaggle/working/run3c | head -40
