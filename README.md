# mmonc-breast

Does pretreatment breast DCE-MRI add to clinical variables for predicting pathological
complete response (pCR) to neoadjuvant chemotherapy? Code for an explainable multimodal
study on the public MAMA-MIA dataset (1,506 patients, four collections).

**Status: no experiment has been run on real data.** The code has been tested end to end
on synthetic data only. Every table and figure in the paper must come from your own runs.

## What is here

| Path | Content |
| --- | --- |
| `src/mmonc/` | Package: data prep, radiomics, models, training, metrics, statistics, explainability, figures, pipeline |
| `notebooks/` | Kaggle notebooks `00` to `06` (thin wrappers around the package) |
| `tests/` | 17 tests, including an end-to-end run and two leakage guards |
| `configs/default.yaml` | Settings for the real run on a Kaggle T4 |
| `docs/` | Protocol and analysis plan, model card template, reporting checklist map |
| `paper/` | Manuscript draft with marked placeholders, verified references |
| `results/`, `figures/`, `models/` | Empty until you run the study |

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                                                   # about 30 s on CPU
python -m mmonc.pipeline --synthetic --out results/synthetic_smoke
```

Outputs from synthetic data are watermarked `SYNTHETIC DATA - NOT A RESULT`.

## Real run

1. Create a free Synapse account, accept the MAMA-MIA terms (syn60868042, CC-BY-NC) and
   make a personal access token. Store it in Kaggle Secrets as `SYNAPSE_AUTH_TOKEN`.
2. Run `notebooks/01_download_preprocess.ipynb` once per collection. It writes cropped
   arrays plus `table.csv` and `radiomics.csv`; publish them as a private Kaggle dataset.
3. Run `02` to `04` (development only: external collections are not scored).
4. Commit `frozen.json`, register the protocol, then run `05` once and `06`.

```bash
python -m mmonc.pipeline --data /kaggle/input/mamamia-crops --out results/run1 --config configs/default.yaml
```

Training checkpoints every epoch and resumes automatically, so re-running the same command
after a session time-out continues where it stopped.

## Design in one paragraph

Models are developed on ISPY2 with 5-fold cross-validation and tested on DUKE, ISPY1 and
NACT. The comparator is a calibrated clinical logistic model. The proposed model (CARF)
adds a gated imaging residual to that model's logit: `logit = z_clin + gate * delta`.
With the image removed it returns exactly the clinical prediction (tested). The primary
test is the DeLong difference in pooled external AUC, read against a pre-specified margin
of 0.03.

## Known limits of this code

- `data.py` (Synapse download, table harmonisation, NIfTI cropping) and the PyRadiomics
  path were written from the dataset paper and have not been run on the real files.
  Expect to adjust column names in `COLUMN_MAP` on first use.
- Only a compact 3D ResNet encoder is implemented. DenseNet, EfficientNet, ViT and Swin
  baselines named in the protocol still need adding to `models.build` (MONAI provides them).
- Masked-autoencoder pretraining, ComBat harmonisation, conformal abstention, multiple
  imputation and the random-effects meta-analysis are specified in the protocol but not
  yet coded.
- On synthetic data with a strong planted imaging signal, CARF recovered less of the gain
  than plain concatenation fusion (it is deliberately conservative). Tune `gate_penalty`,
  `aux_weight` and `gate_init` on development folds, and report whichever model wins.

## Licence

Code: MIT. MAMA-MIA is CC-BY-NC, so weights trained on it are for non-commercial use.
This is research software and not a medical device.
