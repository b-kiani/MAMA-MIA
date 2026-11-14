#!/usr/bin/env bash
# Final report: merge run1 (after external scoring) and run3 outputs attached as inputs.
set -euo pipefail
cd /kaggle/working/repo
PREP=$(dirname "$(find /kaggle/input -name table.csv -path '*prepared*' | head -1)")
# prefer /kaggle/working (run1 re-scored there with external predictions) over attached inputs
RUNS=$(for r in run1 run3a run3b run3c; do d=$( (find /kaggle/working -maxdepth 2 -type d -name $r; find /kaggle/input -maxdepth 7 -type d -name $r) 2>/dev/null | head -1); [ -n "$d" ] && echo "$d"; done)
echo "prepared: $PREP"; echo "runs: $RUNS"
python -u -m mmonc.report --prepared "$PREP" --runs $RUNS --out /kaggle/working/final
mkdir -p /kaggle/working/final/runs
for d in $RUNS; do n=$(basename $d); mkdir -p /kaggle/working/final/runs/$n; cp $d/*.csv $d/*.json /kaggle/working/final/runs/$n/ 2>/dev/null || true; cp -r $d/figures /kaggle/working/final/runs/$n/ 2>/dev/null || true; done
cd /kaggle/working && tar czf final_results.tgz final && ls -la final_results.tgz
for f in final/primary.csv final/table5_vs_clinical.csv final/table3_external.csv final/table2_internal.csv final/tableS_clinical_variants.csv final/tableS_gate.csv final/tableS_survival.csv final/tableS_site_pooled.csv final/tableS_loco_tabular.csv final/tableS_carf_vs_best_image.csv final/table1_cohort.csv; do echo "== $f"; python -c "import pandas as pd,sys; pd.set_option('display.width',250); pd.set_option('display.max_columns',40); print(pd.read_csv('$f').round(3).to_string(index=False))"; done
