# Working draft: sections that do not depend on results (14 Nov 2026)

## Introduction

Pathological complete response (pCR) after neoadjuvant chemotherapy is associated with longer event-free and overall survival in breast cancer, particularly in triple-negative and HER2-positive disease. A reliable pretreatment estimate of pCR would allow escalation or de-escalation decisions before the first cycle. Dynamic contrast-enhanced MRI (DCE-MRI) is acquired routinely for local staging, so many groups have tried to predict pCR from it.

Published results are hard to reconcile. A 2025 systematic review of 51 deep-learning studies reported a median AUC of 0.88, but most studies were single-centre and retrospective, and few were validated externally [Krasniqi 2025]. When a large, harmonised public benchmark was assembled (MAMA-MIA: 1,506 patients from four US collections [Garrucho 2025]) and 26 teams were invited to predict pCR from pretreatment DCE-MRI on an external European cohort, the best balanced accuracy was 0.558. Only one submission was significantly better than a random classifier [Garrucho 2026]. Clinical variables were withheld at test time in that challenge, so it could not say whether imaging adds anything to what is already known about the tumour's receptor status.

That is the clinically relevant question. Receptor subtype is a strong determinant of pCR, and an imaging model is useful only if it adds to it. Where this has been examined on public data, the answer has been sobering. In an ISPY2-to-ISPY1 validation, a clinical-feature model matched an attention-based multimodal model (external AUC 0.71 for both), even though the multimodal model also used post-treatment MRI [Nishizawa 2025].

We therefore designed a study around one pre-specified question: does pretreatment DCE-MRI improve pCR prediction beyond age and receptor status when models are developed in one trial and tested in different institutions, eras and scanners? We make four contributions.

1. **A collection-held-out benchmark.** It covers six imaging and fusion architectures plus clinical and radiomics baselines on MAMA-MIA, with development restricted to ISPY2 and testing on DUKE, ISPY1 and NACT. External data were scored once, after models and thresholds were frozen.
2. **Clinically anchored residual fusion (CARF).** In this fusion model, imaging can change a calibrated clinical prediction only through a bounded, distribution-shift-aware gate. The model therefore reports, for each patient, how much the image changed the clinical estimate, and it reduces exactly to the clinical model when imaging is unavailable or uninformative.
3. **A contrast-timing embedding.** Phase-acquisition times differ several-fold between collections, so the embedding conditions image features on when each phase was acquired.
4. **Evaluation beyond discrimination.** This covers calibration, decision-curve analysis, institution-level heterogeneity across 22 ISPY2 sites, subgroup performance, faithfulness-tested explanations, component ablations and seed variability.

## Methods

### Study design and reporting
This is a retrospective prognostic model development and external validation study using public, de-identified data. It is reported according to TRIPOD+AI [Collins 2024] and CLAIM. The analysis plan, model configurations and code were committed to a public repository before external data were scored (commits 4b30b5b and 48ceb27, 14 November 2026). [[Ethics statement from the authors' institution.]]

### Data
We used MAMA-MIA [Garrucho 2025], which harmonises pretreatment T1-weighted DCE-MRI and clinical data from four collections in The Cancer Imaging Archive: I-SPY2 (n = 980), I-SPY1 (n = 171), Duke (n = 291) and NACT (n = 64). Each case has an expert primary-tumour segmentation. We accessed a mirror of the official release hosted on Kaggle. Its per-collection case counts and pCR counts were identical to those of the dataset publication. Fifteen patients without a recorded pCR status (Duke 11, I-SPY1 4) were excluded.

The development cohort was I-SPY2: 980 patients, 316 with pCR (32.2%), acquired 2010–2016 at 22 sites. The external cohort comprised Duke (280; 64 pCR), I-SPY1 (167; 49 pCR) and NACT (64; 11 pCR): 511 patients with 124 events in total. The external cohort differs from development in era (1995–2014), acquisition plane (I-SPY1 and NACT are sagittal) and contrast timing.

### Outcome and predictors
The outcome was pCR as recorded in the source collections. The clinical predictors were age, hormone-receptor status and HER2 status. Menopausal status and multifocality were pre-specified in the original protocol but were dropped before any modelling. Menopausal status is missing for every patient in I-SPY1 and NACT, and multifocality for 94% of I-SPY1. Treatment arm was not used as a predictor because it is assigned after imaging. Acquisition variables (vendor, field strength, plane, site) were used only for subgroup analysis.

### Image preprocessing
For each patient, the pre-contrast and first post-contrast volumes and the expert mask were resampled to 1 mm isotropic spacing. Intensities were z-scored jointly across both phases. A cube centred on the tumour bounding box was cropped (side 1.25 × the largest tumour extent, minimum 32 mm) and resized to 64³ voxels. The three input channels were pre-contrast, first post-contrast and their subtraction. Because the resize removes absolute size, tumour volume (mm³, log-transformed) and largest extent (mm) were recorded as tabular features. Preprocessing of all 1,506 cases completed without failures.

### Radiomic features
Twenty features were computed inside the mask: first-order statistics (mean, standard deviation, 10th/50th/90th percentile, skewness, kurtosis, energy) on the post-contrast and subtraction images, a surface-to-volume ratio, a rim-to-core enhancement ratio, log tumour volume and largest extent. This compact, dependency-free set is not an IBSI-compliant texture panel (see Limitations).

### Models
All models were trained on I-SPY2 only.

- **Clinical:** L2-penalised logistic regression on age, HR and HER2, with median imputation and missing-value indicators.
- **Radiomics** and **clinical + radiomics:** L1-penalised logistic regression (C = 0.3).
- **Image only:** four 3D backbones sharing the same input and schedule — a compact residual network (ResNet; widths 32–256, GroupNorm) and three MONAI networks: DenseNet-121, EfficientNet-B0 (no pretraining) and a 6-layer Vision Transformer (patch 16³, width 384).
- **Concatenation fusion:** pooled ResNet features concatenated with standardised tabular features (clinical + radiomic) and a multilayer perceptron.
- **CARF (proposed):**

```latex
\mathrm{logit}(\hat p_i) = z_i^{\mathrm{clin}} + g_i\,\delta_i,\qquad g_i \in [0,1]
```

z_i^clin is the cross-fitted logit of the frozen clinical model. Each clinical and radiomic variable becomes a token with a learned missing-value embedding. These tokens attend over the ResNet's spatial feature tokens through a cross-attention block. The contrast-timing embedding is added to the image tokens: an MLP of log acquisition times of the three phases. The class token yields the residual δ_i. The gate g_i is a sigmoid of the class token and of a distribution-shift score: the mean squared Mahalanobis distance of the pooled image embedding from running training statistics. During training, the image is replaced by a learned null token with probability 0.3, and the gate is then fixed at zero. The loss is binary cross-entropy on the final logit, plus 0.5 × the cross-entropy of z^clin + δ (an auxiliary term that trains the residual with the gate open) and 0.01 × the mean gate value. With the image removed, CARF returns the clinical prediction exactly; this is verified by unit test.

**Training.** AdamW (learning rate 3 × 10⁻⁴, weight decay 10⁻²), cosine schedule, up to 60 epochs and batch size 16. Mixed precision was used on one NVIDIA T4 GPU. Augmentation was random flips along each axis and global intensity scaling and shift. Early stopping (patience 12) used an inner 15% split of each training fold, so out-of-fold predictions were never used for model selection. Development used 5-fold stratified cross-validation (seed 0). External predictions are the mean of the five fold models.

**Ablations and seeds.** Three CARF variants were trained under the same protocol: without the gate (g = 1), without the timing embedding, and without the auxiliary loss. CARF was also retrained with two further seeds. These configurations were fixed before any external result was seen.

### Statistical analysis
Discrimination was summarised by AUC and average precision, with 95% confidence intervals from 2,000 stratified bootstrap resamples. Thresholds were set by Youden's index on I-SPY2 out-of-fold predictions and carried unchanged to external data. At those thresholds we report sensitivity, specificity, precision, F1, accuracy and balanced accuracy. Calibration was assessed by calibration-in-the-large, calibration slope, expected calibration error, Brier score and reliability curves. Clinical utility was assessed by net benefit over threshold probabilities of 0.10–0.60.

The pre-specified primary analysis was the paired DeLong difference in pooled external AUC between CARF and the clinical model. Its pre-specified interpretation used a margin of 0.03:
- if the lower 95% bound exceeds 0, imaging adds value;
- if the whole interval lies within ±0.03, there is no clinically meaningful added value;
- otherwise the result is inconclusive.

All other models were compared with the clinical model by DeLong tests with Holm correction, and with net reclassification and integrated discrimination improvement.

Secondary analyses were:
- random-effects (DerSimonian–Laird) pooling of site-level AUCs across I-SPY2 sites with at least five patients in each class;
- I-SPY2 performance by acquisition period;
- subgroup AUCs by age band, ethnicity, vendor, field strength, subtype and plane;
- tabular sensitivity models (extended receptor variables; tumour size);
- leave-one-collection-out validation of tabular models;
- an exploratory survival analysis.

In the survival analysis, recurrence-free survival was available only for Duke, I-SPY1 and NACT. We related the predicted probability of non-response to recurrence with Harrell's C, Kaplan–Meier curves by tertile, and Cox models adjusted for HR and HER2.

### Explainability
Clinical contributions were computed as exact SHAP values of the logistic model. For CARF, Grad-CAM was computed on the imaging contribution g·δ rather than the full logit, so maps show what the image added. Each map was summarised by its enrichment inside the tumour mask (1.0 = chance) and by a deletion test: the change in logit after removing the top-attributed 25% of voxels. Gate values were summarised by collection and subtype.

### Code and data availability
Code, configurations, Kaggle notebooks and frozen predictions: https://github.com/b-kiani/MAMA-MIA. MAMA-MIA is available from Synapse (syn60868042) under CC-BY-NC.

## Related work for the comparison table (verified 14 Nov 2026)

| Study | Data | Inputs | Validation | Reported result |
|---|---|---|---|---|
| Garrucho et al. 2026, MAMA-MIA Challenge (arXiv 2603.01250) | Train MAMA-MIA 1,506; test 516 European | Pretreatment DCE-MRI only | External, 3 European centres | Best balanced accuracy 0.558; only 1 of 15 better than random |
| Nishizawa et al. 2025, J Transl Med | I-SPY2 660 -> I-SPY1 114 | Pre- and post-treatment MRI + clinical | External, 1 collection | Multimodal AUC 0.71; clinical-only 0.71; MRI-only 0.70 |
| Xu et al. 2025, Cancers | 282 TNBC (MD Anderson) -> I-SPY2 | Pretreatment DCE, DWI, clinical | External (I-SPY2) | Best external AUC 0.72 |
| BreastDCEDL preprint (Fridman et al., arXiv 2506.12190) | I-SPY1, I-SPY2, Duke | Pretreatment DCE | Pooled random split | AUC 0.72 overall; 0.54 Duke subset |
| García Marcos et al. 2026, arXiv | ACRIN 6698 / I-SPY2 | DCE + ADC + subtype | 5-fold CV only | AUC 0.86 |
| Braman et al. 2020, arXiv 2001.08570 | 157 HER2+ | Pretreatment DCE | 2 external sets (28, 29) | External AUC 0.85, 0.77 |
| Krasniqi et al. 2025, systematic review | 51 studies | Multimodal DL | Mostly internal | Median AUC 0.88 |
