# Data

No patient data is stored in this repository.

- Source: MAMA-MIA, Synapse `syn60868042`, licence CC-BY-NC. Cite Garrucho et al., Sci Data 2025.
- Access: free Synapse account, accept the terms, create a personal access token.
- Prepared format (written by notebook 01): one `<case_id>.npz` per patient with
  `image` (3 x 64 x 64 x 64, float16: pre, first post, subtraction) and `mask`; `table.csv`
  (case_id, collection, pcr, age, hr, her2, menopausal, multifocal, ethnicity, vendor,
  field_strength, dt0, dt1, dt2, rfs_months, rfs_event); `radiomics.csv` indexed by case_id.
- Split: development = ISPY2; external = DUKE, ISPY1, NACT. The SHA-256 of the
  development case list is written to `frozen.json`.
