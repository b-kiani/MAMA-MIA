# Does pretreatment breast MRI add to clinical variables for predicting response to neoadjuvant chemotherapy? An explainable multimodal multicentre validation study

> DRAFT SKELETON. Methods are written from the protocol. Every item marked
> `[[RESULT: ...]]` is a placeholder to be filled from `results/` after the real run.
> No number in this file is an experimental result unless it cites a published source.

Authors: [[names, affiliations; include a clinical co-author]]

## Abstract

**Background.** Pathological complete response (pCR) after neoadjuvant chemotherapy is a
surrogate for long-term outcome in breast cancer. In the 2025 MAMA-MIA challenge, models
using pretreatment dynamic contrast-enhanced MRI alone were largely indistinguishable from
chance on external European data [2]. Whether MRI adds information to routine clinical
variables has not been tested on this benchmark under institution-level validation.

**Methods.** We used MAMA-MIA (1,506 patients, four public collections) [1]. Models were
developed on ISPY2 (n = 980) with five-fold cross-validation and tested once on DUKE,
ISPY1 and NACT (n = 526; 511 with pCR status, 124 events). We compared clinical-only,
radiomics-only, image-only and fusion models with a clinically anchored residual fusion
model (CARF). The primary outcome was the difference in pooled external AUC between CARF
and the clinical model (DeLong test; pre-specified margin 0.03). Reporting follows
TRIPOD+AI [3].

**Results.** [[RESULT: external AUC (95% CI) for clinical and CARF; difference (95% CI), p]].
[[RESULT: calibration slope and intercept; net benefit range]].
[[RESULT: per-collection AUCs; subgroup findings]].

**Conclusions.** [[One of: MRI added / did not add clinically meaningful discrimination
beyond clinical variables. State which, with the interval.]]

## Introduction

1. Clinical problem: who will not respond to neoadjuvant chemotherapy, and why that matters.
2. What is known: single-centre imaging studies report high accuracy; multicentre evidence
   is weaker. In MAMA-MIA, 26 teams took part and the best balanced accuracy for pCR was
   0.558 [2]. Clinical variables were not available to models at test time.
3. Gap: incremental value of MRI over receptor subtype and age is rarely tested; calibration
   and clinical utility are rarely reported [[cite systematic reviews after reading them]].
4. Contribution: (i) a pre-registered test of incremental value with leave-collection-out
   validation; (ii) a fusion design that reports per-patient imaging contribution and
   falls back to the clinical model; (iii) faithfulness-tested explanations and subgroup
   analysis; (iv) open code, frozen splits and predictions.

## Results

### Cohort
Table 1. [[RESULT: `table1_cohort.csv`; flow diagram with exclusions (15 without pCR status)]]

### Internal validation (ISPY2, out-of-fold)
Table 2. [[RESULT: `table2_internal.csv`]]

### External validation
Table 3 and Fig. 2-5. [[RESULT: `table3_external.csv`, ROC/PR, calibration, decision curves]]

Primary analysis. [[RESULT: `summary.json` -> primary: difference, CI, p, verdict]]

### Ablation and comparison with prior work
Table 5. [[RESULT: `table5_ablation.csv`: each model against the clinical model, Holm-adjusted]]
Table 4. Published results (`results/table4_prior_studies.csv`) with our external results
added. [[RESULT: Codabench score for the image-only branch]]

### Explainability, error and bias analysis
Table 6 and Fig. 7. [[RESULT: gate distribution by collection; tumour enrichment of
Grad-CAM; deletion test; top clinical contributions; subgroup AUCs by age, ethnicity,
vendor and field strength]]

### Survival (exploratory)
[[RESULT: n, events, C-index, Kaplan-Meier by tertile; or state that events were too few]]

## Discussion

- Principal finding in one sentence, with the interval. [[RESULT]]
- Relation to the challenge result [2] and to multimodal studies that used post-treatment
  imaging or private cohorts.
- Why imaging may add little before treatment: subtype and regimen dominate response.
- What the gate and the per-collection results say about distribution shift. [[RESULT]]

### Limitations
Retrospective public data from the United States; external collections span 1995-2014 and
two are sagittal 1.5 T; treatment arms in ISPY2 are not predictors and may confound;
expert masks were used [[report automatic-mask sensitivity analysis]]; survival data are
incomplete; no prospective or reader evaluation [[unless added]].

### Clinical implications
[[Depends on the result. If null: pretreatment MRI models should not be used to select
therapy; effort should move to early-treatment imaging. If positive: size of benefit at
clinically relevant thresholds from the decision curves.]]

### Future work
Early-treatment MRI, conformal abstention, prospective validation with a clinical partner.

## Conclusion
[[Two sentences, matching the abstract.]]

## Methods

**Study design and data.** Retrospective prediction-model study on public, de-identified
data. MAMA-MIA aggregates pretreatment T1-weighted DCE-MRI from ISPY1 (n = 171), ISPY2
(n = 980), DUKE (n = 291) and NACT (n = 64), with expert primary-tumour segmentations and
49 harmonised variables [1]. [[Ethics statement from your institution.]]

**Outcome.** pCR as recorded in the source collections. Secondary: recurrence-free survival.

**Predictors.** Clinical: age, hormone-receptor status, HER2 status, menopausal status,
multifocality. Imaging: pre-contrast, first post-contrast and subtraction volumes cropped
to a 64-voxel cube (1 mm) around the mask. Radiomics: [[n]] features from first
post-contrast and subtraction images [[PyRadiomics version, settings]]. Acquisition
variables were used for subgroup analysis only.

**Models.** Clinical: L2 logistic regression. Radiomics and clinical + radiomics: L1
logistic regression. Image only: 3D residual network. Concatenation fusion. CARF:
logit = z_clin + gate x delta, where z_clin is the cross-fitted clinical logit, delta is
produced by cross-attention of tabular tokens over image tokens with a contrast-timing
embedding, and gate in [0, 1] depends on the fused state and the distance of the image
embedding from the training distribution. Training: AdamW, cosine schedule, binary
cross-entropy with an auxiliary open-gate term and a gate penalty, image dropout 0.3.
[[final hyperparameters, seeds, hardware, training time]]

**Validation.** Five-fold cross-validation on ISPY2 with an inner split for early stopping.
Thresholds (Youden) and all model choices were fixed on ISPY2; the split hash and frozen
file were committed [[commit hash, registration DOI]] before external data were scored.

**Statistics.** AUC and average precision with 2,000-sample stratified bootstrap intervals;
sensitivity, specificity, precision, F1, accuracy at the frozen threshold; calibration
intercept, slope, expected calibration error and Brier score; net benefit over thresholds
0.10-0.60. Paired DeLong tests against the clinical model with Holm correction; NRI and
IDI. Survival: Harrell's C and Kaplan-Meier curves. Two-sided alpha 0.05.

**Explainability.** Exact SHAP values for the clinical model; Grad-CAM on the imaging
contribution with tumour-mask enrichment and a deletion test; gate values; image-removal
ablation.

**Data and code availability.** MAMA-MIA: Synapse syn60868042 (CC-BY-NC). Code, frozen
splits and predictions: [[repository URL, release DOI]].

**Use of AI tools.** [[Disclose according to the target journal's policy.]]

## References
1. Garrucho L, et al. A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations. Sci Data. 2025;12:453. doi:10.1038/s41597-025-04707-4
2. Garrucho L, Joshi S, et al. The MAMA-MIA Challenge: advancing generalizability and fairness in breast MRI tumor segmentation and treatment response prediction. arXiv:2603.01250 (2026).
3. Collins GS, et al. TRIPOD+AI statement. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378
4. [[Add the remaining references only after reading each paper; see the blueprint's literature matrix.]]

---

### Notes for an IEEE TMI version
Reorder as Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion.
Lead with CARF and the timing-aware embedding; move clinical framing to a short paragraph;
add component ablations (no gate, no timing embedding, no auxiliary loss, no image dropout)
and at least one further backbone family. TMI will expect a method gain; submit there only
if the ablations show one.
