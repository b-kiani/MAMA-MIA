# Round-2 analysis plan (written 15 Nov 2026, before any round-2 model was run)

Status: **secondary, post hoc.** The round-1 external results were already known when this plan was
written. The prespecified primary end point (CARF vs clinical model, pooled external AUC, margin 0.03)
is unchanged and is not re-tested here. Every round-2 result is reported, whatever its direction.

Each analysis answers one objection to the round-1 conclusion that pretreatment DCE MRI adds
nothing to age, HR and HER2 status.

| # | Objection | Analysis | Reading fixed in advance |
|---|---|---|---|
| 1 | Only two DCE phases were used | Three phases (pre, early, late), enhancement and washout channels, tumour mask channel: kinetic 3D ResNet, kinetic concatenation fusion, kinetic CARF | Same DeLong test vs clinical model, same margin |
| 2 | Networks were trained from scratch | ImageNet-pretrained 2.5D ResNet-18 with attention pooling over 12 tumour slices; CARF on its slice tokens | Same |
| 3 | Radiomics set was too small | 22 kinetic features including functional tumour volume (PE >= 70%) and signal enhancement ratio; clinical + FTV; clinical + kinetic radiomics | Same |
| 4 | One development source is too narrow | Leave-one-collection-out: train on three collections, test on the fourth, for the clinical model, clinical + kinetic radiomics, kinetic ResNet and kinetic CARF | Per-collection and pooled AUC difference vs clinical |
| 5 | The pipeline may be unable to learn anything | Control tasks with the same kinetic ResNet and crops: HR status, HER2 status, field strength (3 T vs 1.5 T), and pCR labels shuffled within I-SPY2 | Learnable targets should exceed 0.5; shuffled labels should not |
| 6 | Imaging may help only within subtypes | Within-subtype AUCs; likelihood-ratio test of adding each image score to the clinical logit, overall and by subtype | Holm correction across subtypes |
| 7 | Poor external calibration could be fixed locally | Intercept-only recalibration per collection | Change in Brier score and calibration intercept |
| 8 | The equivalence claim may be underpowered | Smallest AUC difference the external cohort could detect, from the DeLong standard error | Reported next to the margin |

Protocol, unchanged from round 1: development on I-SPY2 with the same five folds and inner 15%
early-stopping split; thresholds and models frozen on development data; AdamW, 60 epochs, patience 12,
batch 16 (learning rate 1e-4 for the pretrained network, 3e-4 otherwise); seed 0.

Data note: no further independent public cohort with pretreatment DCE MRI and pCR labels was found
(the MAMA-MIA challenge test set from three European centres is not public). Round 2 therefore adds
data from the same patients (the late DCE phase) and a multi-source design, not new patients.
