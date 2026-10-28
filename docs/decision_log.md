# Decision log

## 2026-10-26 – Dataset and models

- **Data:** [UCI Diabetes 130-US Hospitals 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI ID 296). Public (CC BY 4.0), no data use agreement, so the AI-aided pair may paste records and outputs into an LLM, which the AI4Vis part of the assignment needs.
- **Model A:** count / clinical-feature KNN (transparent baseline).
- **Model B:** [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) KNN on mean-pooled 768-D embeddings of each record serialized as text.

Caveats:
- **UCI is tabular, Bio_ClinicalBERT is a text model.** Records must be serialized to text. State this explicitly; it also gives a Phase II question: does the encoder respond to clinical content or to wording?
- **Mean-pooled Bio_ClinicalBERT is not trained for similarity.** It is a weak retrieval embedding by design; this is part of what we want to expose, not a bug to hide.
- **UCI lab coverage is thin:** only categorical `A1Cresult` and `max_glu_serum`, no eGFR/creatinine. Pathways must come from later-visit ICD-9 codes, and only patients with repeat encounters (`patient_nbr`) have a later outcome.
- **Candidate hypothesis:** a general clinical-text encoder is sensitive to record wording and demographic tokens rather than clinical content.

## 2026-10-28 – Shared baseline rebuilt as a static data contract

Triggered by the code review in [baseline_review.md](baseline_review.md).

1. **Label = next admission only.** The target of visit V_t is the primary diagnosis (`diag_1`) of V_t+1, mapped to four pathways (see below). The final visit of each patient is dropped because it has no next admission. No fallback to the current visit's diagnosis.
2. **No generated text.** The invented "clinical presentation" sentence was derived from the diagnosis codes and leaked the label. Model B now reads only authentic UCI fields, serialized with one fixed template.
3. **Same information for both models.** Model A and Model B see exactly the same 15 encounter fields; differences between them come from the representation, not from the inputs.
4. **Sex is part of the template.** The agreed template lists age, stay, admission type/source, specialty, payer, labs, medications, diagnosis count, glucose, A1C, insulin, med change and diabetes medication. `Sex: {gender}` (an authentic UCI field) is prepended, otherwise the sex-swap counterfactual would be a no-op by construction. Race is kept in the cohort table for bias analysis but is not a model input.
5. **ID codes are written as their official UCI labels** (`admission_type_id` 1 → "Emergency", `admission_source_id` 7 → "Emergency Room", from the dataset's `IDs_mapping.csv`), so the text model does not read bare integers. The raw IDs stay in the cohort table.
6. **No silent fallbacks.** A missing `torch`/`transformers`, a failed model load or a failed download raise an error. The synthetic-cohort generator and the TF-IDF "fallback encoder" were deleted.
7. **Full cohort, no sampling.** All repeat-admission patients are used. Baselines: majority class and random neighbors (other patients only).
8. **Decoupled data contract.** `shared/01_prepare_baseline_data.py` writes static tables to `shared/data/` (Parquet + CSV + JSON). Both pairs read these files; the human-only pair never has to run the model or touch a GPU. Embeddings for Phase II go to `shared/data_cache/phase2_model_internals.npz` and stay closed in Phase I.

### Pathway mapping (next admission `diag_1`, ICD-9)

| Pathway | Codes |
|---|---|
| Kidney / Renal | 250.4x (diabetes with renal manifestations), 580–589 |
| Eye / Neurological | 250.5x (ophthalmic), 250.6x (neurological), 357.x (neuropathy), 362.x (retinal disorders) |
| Cardiovascular | 390–459, 250.7x (peripheral circulatory) |
| Uncomplicated / Other | everything else, including V/E codes and missing |

The 250.x sub-codes are checked first, so e.g. 250.41 is Kidney, not Other. 250.6x and 250.7x are not in the original one-line spec; they are the ICD-9 codes for diabetic neuropathy and diabetic angiopathy and were added for clinical consistency.
