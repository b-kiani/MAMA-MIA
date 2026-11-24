# Analysis plan: fourth review (post hoc, exploratory)

Written and committed before the run (`kaggle/stage_rev4.sh`, queues `q4a`, `q4b` in `src/mmonc/pipeline5.py`).

1. **Endpoint.** The harmonized pathologic endpoint (no invasive cancer in breast and nodes, in situ disease allowed;
   Duke: complete response or only DCIS/LCIS remaining; NACT: pathologic size 0 and no positive node; I-SPY1 and I-SPY2
   as distributed, verified) becomes the headline external endpoint. All locked external predictions are rescored
   against it (509 patients, 131 with pCR). The analysis with the labels as distributed is kept as the historical
   prespecified analysis. Harmonization is a post hoc correction. Models trained on external collections (LOCO, multi-source)
   cannot be rescored without retraining and keep the distributed labels.
2. **Fully nested late fusion.** For each outer fold the 2.5D image-only network is trained by inner fivefold
   cross-validation of the outer training partition only; the stacker is fitted on inner out-of-fold predictions and
   applied to the outer test fold (mean of the five inner image models; clinical model fitted on the outer training
   partition). Seeds 0-4 and regimen-group-held-out outer folds. Internal late-fusion results are reported only from
   this procedure. External late-fusion predictions are the mean over outer folds of the stacker applied to the
   external patients.
3. **Signal recovery (local, no retraining).** Oracle and recovered gains are both computed against the clinical model
   refitted on the semi-synthetic outcome. A clinical + planted-scalar logistic model (age, HR, HER2 and s) is fitted on
   the same outcomes to separate statistical detectability from the encoder's ability to recover the signal; for this
   model, 500 independent outcome draws per level give the proportion of draws in which the gain is detected
   (lower 95% bound > 0). No general detection threshold is claimed for the imaging pipeline (two draws per level).
