# Reporting checklist map

Download the official TRIPOD+AI checklist (27 items; Collins et al., BMJ 2024;385:e078378)
and CLAIM, and record the manuscript section for each item. Where the evidence comes from:

| Topic | Where it is produced |
| --- | --- |
| Data sources, eligibility, flow | `data/README.md`, `table1_cohort.csv` |
| Outcome and predictors | `docs/PROTOCOL.md`, `paper/manuscript.md` Methods |
| Sample size | Protocol: 124 external events |
| Missing data | `table1_cohort.csv` missing columns; model description |
| Model development and tuning | `configs/default.yaml`, run logs |
| Performance: discrimination, calibration, utility | `table2_internal.csv`, `table3_external.csv`, figures 2-4 |
| Fairness and subgroups | `tableS_subgroups.csv` |
| Open science: code, data, protocol | README, registration DOI |
