#!/usr/bin/env bash
# Final report: merge run1 (after external scoring) and run3 outputs attached as inputs.
set -euo pipefail
cd /kaggle/working/repo
PREP=$(dirname "$(find /kaggle/input -name table.csv -path '*prepared*' | head -1)")
RUNS=$(for r in run1 run3a run3b run3c; do d=$(find /kaggle/input /kaggle/working -maxdepth 6 -type d -name $r 2>/dev/null | head -1); [ -n "$d" ] && echo "$d"; done)
echo "prepared: $PREP"; echo "runs: $RUNS"
python -u -m mmonc.report --prepared "$PREP" --runs $RUNS --out /kaggle/working/final
mkdir -p /kaggle/working/final/runs
for d in $RUNS; do n=$(basename $d); mkdir -p /kaggle/working/final/runs/$n; cp $d/*.csv $d/*.json /kaggle/working/final/runs/$n/ 2>/dev/null || true; cp -r $d/figures /kaggle/working/final/runs/$n/ 2>/dev/null || true; done
cd /kaggle/working && tar czf final_results.tgz final && ls -la final_results.tgz
for f in final/primary.csv final/table5_vs_clinical.csv; do echo "== $f"; cat $f; done
