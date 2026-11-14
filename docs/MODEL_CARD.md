# Model card (template; complete after the real run)

- **Intended use:** research on predicting pCR before neoadjuvant chemotherapy. Not for clinical decisions.
- **Inputs:** pretreatment DCE-MRI crop, tumour mask, age, HR, HER2, menopausal status, multifocality.
- **Output:** probability of pCR; clinical-only probability; imaging gate and residual.
- **Training data:** ISPY2 subset of MAMA-MIA (US, 2010-2016).
- **Evaluation data:** DUKE, ISPY1, NACT. [[results]]
- **Performance by subgroup:** [[results]]
- **Known limits:** US-only training data; expert masks; older external scanners; no prospective evaluation.
- **Licence:** code MIT; weights non-commercial (CC-BY-NC data).
