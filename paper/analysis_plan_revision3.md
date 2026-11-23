# Analysis plan: third review (round 3 continued, post hoc, exploratory)

Written and committed before the run. Code: `src/mmonc/prep4.py` (preprocessing and audit) and
`src/mmonc/pipeline5.py` (training queues `q0`, `q1`); run script `kaggle/stage_rev3.sh`;
local analysis `paper/jbhi/revision3_analyses.py`. All analyses are post hoc and exploratory: the
external patients have been examined in earlier rounds. No additional untouched external cohort is
available, so the reviewer's highest-value experiment (untouched external validation) cannot be done
here and is stated as a limitation.

## 1. Orientation audit and sensitivity (reviewer comment 2)

The round-2 preprocessing (`prep2.crop_case`, and `data.preprocess_case` in round 1) resampled each image
to 1 mm while keeping its stored direction and did not reorient, so array axes are not anatomically
equivalent across storage orientations (MAMA-MIA stores axial acquisitions as LAS and sagittal as PSR;
I-SPY2 and Duke are axial, I-SPY1 and NACT sagittal). `prep4` reads each case once and writes
* `orig`: identical to round 2 (stored voxel order);
* `lps`: images and mask reoriented with `sitk.DICOMOrient(·, "LPS")` before cropping (phases resampled
  onto the mask grid if their geometry differs);
* `lps_fov`: as `lps`, fixed 96-mm field of view (section 6).

Audit (`audit.csv`), per case: stored orientation code of every phase and of the mask; size, spacing,
origin and direction agreement between phases and mask; fraction of the requested crop inside the image;
tumour truncation; orientation code after reorientation; maximum change in the kinetic features between
`orig` and `lps` (expected ≈ 0, since the features do not depend on axis order).

Sensitivity: the pretrained 2.5D image-only model (`s25d`) and corrected fusion (`carfm_25d`, nested anchors)
trained on `orig` and `lps` with seeds 0–4 (paired: same folds, initialisation and code); the kinetic 3D
models (`carfm_k`, `image_only_k`) on `lps`, seed 0, compared with the round-3 runs on `orig` with the same
code and seed (`results/revision2/raw`). Reported: external AUC difference lps − orig (paired DeLong,
pooled and within Duke, I-SPY1, NACT), I-SPY2 out-of-fold difference, calibration intercept and slope within
each collection. With seeds, the mean and range of the differences across seeds.

## 2. Conditional signal-recovery control (comments 1 and 8)

Semi-synthetic outcome for every patient: y* ~ Bernoulli(sigmoid(z + β·s)), where z is the clinical-model
logit (age, HR, HER2; fitted on the real I-SPY2 labels) and s is the mean early percentage enhancement in
the tumour (`pe_mean` of the `lps` default features), residualised on age/HR/HER2 by linear regression on
I-SPY2 and standardised on I-SPY2. s is therefore an MRI-derived signal that is conditionally independent
of the clinical variables. β was calibrated before the run by simulation (1,000 draws, round-2 features)
to give an expected external oracle gain AUC(z + βs) − AUC(z) of 0, 0.02, 0.035, 0.05 and 0.10:
β = 0, 0.22, 0.33, 0.43, 0.74. Two independent label draws per level (random seeds 1000, 1001).

The complete pipeline is retrained on y* with clinical-only tabular inputs (so the signal can enter only
through the image): `s25d`, `carfm_25d`, and cross-fitted late fusion (below). The clinical comparator is
refitted on y* on the same folds. Reported per level and draw: realised oracle gain; each model's
external and out-of-fold AUC gain over the refitted clinical model (paired DeLong); recovery ratio
(model gain / oracle gain). A level is "recovered" when the paired lower 95% bound exceeds 0 in both draws.

Null controls: β = 0 (no planted signal) and real pCR labels with the images permuted within
collection × clinical-risk quintile (two permutations). Expected gain ≈ 0; this tests for spurious gains,
not clinical efficacy.

Late fusion (cross-fitted): logistic regression on [logit clinical, logit image-only] fitted on I-SPY2
out-of-fold predictions; for I-SPY2 evaluation the stacker is itself cross-fitted on the same outer folds;
for external evaluation it is fitted on all I-SPY2 out-of-fold predictions and applied to the external
clinical logits and the mean of the five fold image models.

## 3. Repeated training (comment 7)

Seeds 0–4 (seed r > 0 changes the outer folds, the inner early-stopping split and initialisation; seed 0
reproduces earlier rounds) for the clinical model (paired nested refit in every run), `s25d`, late fusion,
`carfm_25d`, and clinical + kinetic radiomics (L1 logistic, C = 0.3, as round 2, refitted locally on the
same folds); `carfm_k` seeds 0–2 on `lps`. Reported: per seed, OOF and external AUC and paired external
difference from the clinical model of the same seed; across seeds, mean, SD and range; external AUC of each
individual fold model versus the five-model average; mean |gate × delta| and gate (anchored models).

## 4. Focused fusion ablation (comment 9)

2.5D encoder, nested anchors, factorial token mixing (on/off) × bounded residual (on/off) × gate
regularisation (0.01 × mean gate, on/off); (1,1,1) is `carfm_25d`, (0,0,1) the original CARF design.
Run on real labels (seeds 0 and 1, clinical + kinetic tabular inputs) and on the planted signal
(β for 0.05, draw 0, clinical-only inputs). Reported: external and OOF gain over clinical, calibration
slope, mean |correction|, mean gate, recovery ratio, and the same for late fusion.

## 5. Treatment-aware analyses (comment 5)

(a) Imaging increments within I-SPY2 regimen groups (control: paclitaxel ± trastuzumab; HER2-targeted
experimental; pembrolizumab; other experimental): OOF gain of late fusion and `carfm_25d` over clinical
within each group (seed-averaged predictions, bootstrap CI). (b) Regimen-group-held-out folds: `s25d` and
`carfm_25d` trained with one regimen group held out per fold. (c) Matched training size: `s25d` (and late
fusion) trained on the control arm (n = 209) and on five random all-arm subsets of the same size; external
gain over the clinical model trained on the same subset. (d) A richer clinical comparator with additional
pretreatment variables available in MAMA-MIA (separate ER and PR, menopausal status, multifocality,
BMI group, ethnicity; missing indicators; collection-wide missingness noted), distinguished from the
richer mathematical specifications of age/HR/HER2 already reported; imaging increment over it.

## 6. Physical scale and segmentation (comment 6)

`s25d` and `carfm_25d` on `lps_fov` (96-mm fixed field of view resized to 64³ and 112² slices, so image-only
models keep absolute size), seeds 0–2, compared with `lps`. Fraction of tumours truncated by the fixed
field of view. Larger-context model: renamed "larger-context tumour-centred model"; approximate coverage
of ipsilateral tissue by its 72 × 240 × 240-mm box reported from the audit. Mask erosion and dilation by
one voxel: kinetic-feature variants (section 7).

## 7. Kinetic features as implemented (comment 3)

Methods equations rewritten to match the code (smoothed and floored denominator, SER floor and clipping,
≥ 10-voxel rule). "FTV" renamed "PE-threshold volume" and distinguished from the official I-SPY FTV.
Variants (prespecified, `prep4.VARIANTS`): default; no smoothing; floor 0.05 and 0.5 × mean pre-contrast
signal; SER floor 0.01 without clipping; mask eroded and dilated by one voxel. For each: clinical +
kinetic radiomics refitted on I-SPY2 and its external gain over clinical; median absolute feature change.

## 8. Decision-specific evaluation (useful extension)

Intended use stated as counselling about the likelihood of pCR before standard-of-care neoadjuvant therapy
(not treatment escalation or de-escalation). Thresholds fixed now: predicted probability 0.2, 0.3, 0.4.
External sensitivity, specificity, PPV, NPV and net benefit for clinical and for the best-performing fusion.

## 9. Reporting

Results are linked to the commit of this plan and of the run, to the run summary (`runs_g*.csv`), the
prediction files, the case manifest and `audit.csv`. Pretrained weights: the run sets
`MMONC_REQUIRE_PRETRAINED=1` (failure instead of random initialisation) and logs the weight identifier,
checkpoint SHA-256 and a checksum of the loaded first layer. Interpretation: 0.03 is the prespecified
margin; detectability is a sensitivity assessment; LOCO folds are not independent studies.
