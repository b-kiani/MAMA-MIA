# Analysis plan: second review (exploratory, post hoc)

Committed before the run. All analyses use I-SPY2 for every choice; external collections are scored once
per configuration and every result is reported.

## 1. Optimization study for the kinetic 3D ResNet trained from scratch (review 2, point 3)
Configurations, fixed here (src/mmonc/pipeline4.py, CONFIGS):

| name | lr | weight decay | epochs | warm-up | early stopping (inner 15% split) | widths |
|---|---|---|---|---|---|---|
| base | 3e-4 | 1e-2 | 60 | 0 | AUC, patience 12 (round-2 protocol) | 32-256 |
| long_auc | 1e-3 | 1e-4 | 150 | 5 | AUC, patience 40 | 32-256 |
| long_loss | 1e-3 | 1e-4 | 150 | 5 | loss, patience 40 | 32-256 |
| fixed | 1e-3 | 1e-4 | 150 | 5 | none (final weights) | 32-256 |
| small_loss | 1e-3 | 1e-4 | 150 | 5 | loss, patience 40 | 16-128 |

All: AdamW, cosine decay, batch 16, mixed precision, gradient clipping at 1, augmentation as in round 2,
same outer folds (StratifiedKFold(5, shuffle, seed 0)) and inner splits as round 2.
Selection rule: the configuration with the highest mean I-SPY2 out-of-fold AUC is "the optimized 3D ResNet".
Its internal AUC is optimistic (selected on the same folds); its external AUC is not.
Reported per configuration and fold: epochs run, epoch kept, training AUC (non-augmented training patients),
inner-validation AUC, test-fold AUC; pooled OOF and external AUC; paired DeLong against the clinical model,
pooled and within each external collection.
Interpretation fixed in advance: if no configuration reaches a mean training AUC of 0.80, the 3D network is
reported as under-fitted (an unsuccessful implementation) and its results are not used as evidence about
the images.

## 2. Anchored models with fully nested preprocessing (review 2, point 4)
carf_k, carfm_k, carf_25d, carfm_25d refitted with the round-2 protocol, except that within each outer fold
the clinical anchor of the network-training patients is cross-fitted within those patients only, the
early-stopping and test patients get logits from an anchor fitted on the training patients, and tabular
standardisation uses training-patient statistics. External anchors: model fitted on all of I-SPY2 (unchanged).
Reported: OOF AUC and external AUC, compared with the earlier (non-nested) values.

## 3. Overfitting test (review 2, point 3)
Kinetic 3D ResNet, 120 I-SPY2 patients, no augmentation, no weight decay, 200 epochs, learning rates 1e-3 and
3e-4; training and held-out (200 patients) AUC every 10 epochs.

## 4. Local analyses (no GPU)
Stronger clinical comparators selected by 20 x fivefold cross-validation on I-SPY2 (paper/jbhi/revision_clinical.py);
audit of anchor nesting (paper/jbhi/audit_nesting.py).
