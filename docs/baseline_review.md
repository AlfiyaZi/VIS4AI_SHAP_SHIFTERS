# Review of the shared Phase I baseline

Reviewed from the overview figure of `02_phase1_shared_baseline.ipynb` (2026-10-27). Reported setup: UCI Diabetes 130-US, 1,078 repeat admissions from 384 patients, k = 10 KNN; Model A (count KNN) 42.9 % accuracy, Model B (Bio_ClinicalBERT) 53.2 % accuracy (majority vote of neighbors vs. actual pathway, 4 classes).

## Correctness checks (before trusting any number)

- [ ] **Same-patient neighbors.** ~2.8 admissions per patient. If KNN runs over admissions, a patient's own other visits become neighbors and almost always share the pathway, inflating agreement. Exclude neighbors with the same `patient_nbr`.
- [ ] **Label leakage into Model B's text.** The pathway is defined by ICD-9 codes (250.4x, 250.5x, …). If the serialized text contains those codes from the same or later visits, Model B can read the label. Model B's 0.52 on Eye/Neuro (usually the rarest, hardest class) is suspicious. Input must use the index visit only; the label must come from later visits only.
- [ ] **Majority-class baseline.** Report "always predict the most frequent pathway" next to 42.9 % / 53.2 %. Model A's matrix suggests it predicts mostly "Cardiovascular" (column 0.34 / 0.51 / 0.56 / 0.33).
- [ ] **Feature scaling in Model A.** Neighbor flips: sex swap ≈ 3.5/10 and age +20 y ≈ 4.9/10 for Model A, which suggests unstandardized binary/bucketed features dominating the distance.
- [ ] **Model B sex sensitivity.** Sex swap flips ≈ 6.2/10 neighbors for Model B, age +20 y only ≈ 0.5/10. If this survives the fixes above it is the strongest bias finding (H3); check exactly how the swap is written into the text.
- [ ] **Sample size.** Why 384 patients out of ~71k in UCI? If it is a speed shortcut, check class counts per pathway.
- [ ] **Model identity.** Verify in code that Model B loads `emilyalsentzer/Bio_ClinicalBERT` (print the model name, embedding dim = 768) and that no `try/except` silently falls back to another encoder.

## Visualization improvements

- **Scatter plot:** both axes are discrete (agreement in steps of 0.1, comorbidity count as integers), so points overplot on a grid. Use jitter, a heatmap or a strip/beeswarm plot.
- **"Reordered" confusion matrices:** both use the same fixed class order. Reorder rows/columns by error similarity so blocks become visible (Alsallakh et al. 2017).
- **UMAP:** the input-feature projection shows two separate clusters. Identify the feature that splits them; avoid reading cluster size or inter-cluster distance (Lecture 3).
- **Pathways:** eye and neuro are merged into one class, which differs from proposal v1. Document this.

## Process

The notebook, updated proposal PDF and slides were generated with an LLM. That is fine, but it belongs in the AI4VIS log (prompts, accepted/rejected suggestions), and the assignment also needs the non-AI pair's work.
