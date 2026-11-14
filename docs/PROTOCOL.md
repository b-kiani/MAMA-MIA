# Protocol and statistical analysis plan (to register before external scoring)

**Question.** Does pretreatment DCE-MRI improve prediction of pCR beyond clinical variables?

**Data.** MAMA-MIA. Development: ISPY2 (n = 980). External: DUKE (291), ISPY1 (171),
NACT (64); 511 with pCR status, 124 events. Cases without pCR status are excluded.

**Primary analysis.** Difference in pooled external AUC, CARF minus clinical logistic
model, paired DeLong test, two-sided alpha 0.05, margin 0.03.
- Lower bound above 0: imaging adds value.
- Interval inside (-0.03, 0.03): no clinically meaningful added value.
- Otherwise: inconclusive.

**Secondary analyses (Holm-adjusted).** Every other model against the clinical model;
average precision; calibration intercept and slope; net benefit at thresholds 0.10-0.60;
NRI and IDI; per-collection results; subgroups (age band, ethnicity, vendor, field
strength); recurrence-free survival (exploratory if under 100 events).

**Fixed before external scoring.** Model weights, thresholds (Youden on ISPY2 out-of-fold
predictions), preprocessing, `frozen.json` and its split hash, committed to the repository.

**Not predictors.** Treatment arm, collection, scanner variables.

**Sensitivity analyses.** Automatic instead of expert masks; ComBat-harmonised radiomics;
complete cases; ISPY2 control-arm patients only; leave-one-collection-out.

**Deviations.** Any change after registration is listed in the paper with its reason.

Reporting: TRIPOD+AI and CLAIM checklists; PROBAST+AI self-assessment.
