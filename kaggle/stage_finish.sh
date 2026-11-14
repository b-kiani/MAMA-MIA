#!/usr/bin/env bash
# Inputs: run1 output (prepared + run1 checkpoints) and run3 output. Score run1 externally once, then build the report.
set -euo pipefail
bash /kaggle/working/repo/kaggle/stage_external.sh
bash /kaggle/working/repo/kaggle/stage_final.sh
