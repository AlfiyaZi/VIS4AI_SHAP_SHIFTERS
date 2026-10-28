# Review of the shared Phase I baseline

## Code review of `01_prepare_baseline_data.py` (2026-10-28)

These are blocking: the current Phase I numbers should not be reported until they are fixed.

1. **The "clinical presentation" text is generated from the label.** `derive_presenting_notes_from_inputs` picks symptom phrases ("elevated serum creatinine, proteinuria …", "chest tightness …") from `current_pathway`, which is computed from the diagnosis codes. UCI contains no free text, so these symptoms are invented and encode the answer. For a patient's last admission `true_pathway` falls back to `current_pathway`, so the text contains the label directly; for earlier admissions it correlates strongly with it. Both models read this text, so accuracy and agreement mostly measure how well they decode the injected phrases.
2. **Model B > Model A is built in.** The code comment says the synonyms exist "so Count-KNN (exact word match) struggles with phrasing variations while semantic models group them". The headline comparison is therefore a property of the generator, not of the models.
3. **Silent fallback encoder.** If `torch`/`transformers` fail to import, `ClinicalSemanticEncoderFallback` is used: TF-IDF + random projections with hand-set weights that up-weight demographics in "layer 4" and keep a "residual demographic bias" in "layer 12". It still outputs 768-D "layers", so any demographic-bias finding from it is designed in. Only a console print marks the switch.
4. **Silent fallback dataset.** If the UCI download fails, `generate_synthetic_cohort` runs instead, also with hand-designed patterns. The second copy of the notebook (800 records, patient IDs `P1000…`) was produced this way; only the copy with real UCI IDs (`P124686…`, 1,078 records) was pushed.
5. **Script and notebook versions differ.** The pushed notebook reads a column `modelB_encoder_used` ("Model B Encoder Verified …"), but this script does not write it. The script that produced the notebook's outputs is not in the repo.

Smaller points: same-patient neighbors are already excluded (good); the cohort quotas plus `max_patients // 2` truncation explain the 384 patients; one symptom phrase contains stray characters ("progressive視力/visual floaters").

**Suggested fix:** drop `presenting_notes` from the text (serialize only real UCI fields), make missing `torch`/`transformers` and a failed download raise errors instead of falling back, write the encoder name and dataset source into the outputs, and then rerun on a larger cohort with a majority-class baseline.

## Review of the overview figure

Reviewed from the overview figure of `02_phase1_shared_baseline.ipynb` (2026-10-27). Reported setup: UCI Diabetes 130-US, 1,078 repeat admissions from 384 patients, k = 10 KNN; Model A (count KNN) 42.9 % accuracy, Model B (Bio_ClinicalBERT) 53.2 % accuracy (majority vote of neighbors vs. actual pathway, 4 classes).

## Correctness checks (before trusting any number)

- [ ] **Same-patient neighbors.** ~2.8 admissions per patient. If KNN runs over admissions, a patient's own other visits become neighbors and almost always share the pathway, inflating agreement. Exclude neighbors with the same `patient_nbr`.
- [ ] **Label leakage into Model B's text.** The pathway is defined by ICD-9 codes (250.4x, 250.5x, …). If the serialized text contains those codes from the same or later visits, Model B can read the label. Model B's 0.52 on Eye/Neuro (usually the rarest, hardest class) is suspicious. Input must use the index visit only; the label must come from later visits only.
- [ ] **Majority-class baseline.** Report "always predict the most frequent pathway" next to 42.9 % / 53.2 %. Model A's matrix suggests it predicts mostly "Cardiovascular" (column 0.34 / 0.51 / 0.56 / 0.33).
- [ ] **Feature scaling in Model A.** Neighbor flips: sex swap ≈ 3.5/10 and age +20 y ≈ 4.9/10 for Model A, which suggests unstandardized binary/bucketed features dominating the distance.
- [ ] **Model B sex sensitivity.** Sex swap flips ≈ 6.2/10 neighbors for Model B, age +20 y only ≈ 0.5/10. If this survives the fixes above it is the strongest bias finding (H3); check exactly how the swap is written into the text.
- [ ] **Sample size.** Why 384 patients out of ~71k in UCI? The course requires a dataset large enough that patterns are systematic, not cherry-picked, so scale up for the midpoint and check class counts per pathway.
- [ ] **Model identity.** Verify in code that Model B loads `emilyalsentzer/Bio_ClinicalBERT` (print the model name, embedding dim = 768) and that no `try/except` silently falls back to another encoder.

## Visualization improvements

- **Scatter plot:** both axes are discrete (agreement in steps of 0.1, comorbidity count as integers), so points overplot on a grid. Use jitter, a heatmap or a strip/beeswarm plot.
- **"Reordered" confusion matrices:** both use the same fixed class order. Reorder rows/columns by error similarity so blocks become visible (Alsallakh et al. 2017).
- **UMAP:** the input-feature projection shows two separate clusters. Identify the feature that splits them; avoid reading cluster size or inter-cluster distance (Lecture 3).
- **Pathways:** eye and neuro are merged into one class, which differs from proposal v1. Document this.

## Fit with the official requirements

- **Not a performance report.** Accuracy and confusion matrices are a starting point only; each pattern needs a follow-up question (what distinguishes the failing region, are there sub-groups that behave differently?).
- **Two pairs, two pipelines.** The shared baseline was LLM-generated, so it belongs to the AI-aided pair's Phase I record. The human-only pair needs its own code for the midpoint, starting from the agreed common direction, not from a copy of this notebook.
- **Ambition counts in grading.** The evaluation uses the full UCI Diabetes 130-US dataset with Bio_ClinicalBERT text embeddings, so depth has to come from the analysis: the leakage-free demographic sensitivity results, clinical feature sensitivity vs. demographic shortcuts, and Phase II explanations that compare at least two techniques per pair (e.g. SHAP on Model A distances, probing / UMAP of Model B embeddings, attention).

## Process

The notebook, updated proposal PDF and slides were generated with an LLM. That is fine, but it belongs in the AI4Vis log (prompts, accepted/rejected suggestions).
