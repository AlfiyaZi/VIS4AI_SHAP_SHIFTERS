# Decision log

## 2026-10-26 – Original plan: MIMIC-IV + CLMBR-T-base

Proposal v1 ([proposal_v1_mimic/](proposal_v1_mimic/)): diabetes cohort from MIMIC-IV (`hosp` module), Model A = count-based KNN, Model B = KNN on CLMBR-T-base embeddings ([StanfordShahLab/clmbr-t-base](https://huggingface.co/StanfordShahLab/clmbr-t-base), code in [ehrshot-benchmark](https://github.com/som-shahlab/ehrshot-benchmark)).

Notes on v1, should it be revived:
- Slide 5 states "~65,000 patients in MIMIC-IV". That is roughly the ICU patient count; the `hosp` module covers several hundred thousand patients. Recent MIMIC-IV versions may also extend beyond 2008–2019. Verify on PhysioNet.
- Requires PhysioNet credentialing (CITI course) for every member, the MIMIC-IV DUA, the Stanford DUA for the weights, an OMOP conversion, and a Linux setup (`femr`).

## 2026-10-26 – Switch to public data + Bio_ClinicalBERT

Reasons:
1. CLMBR-T-base (`femr`) is Linux-only, needs OMOP conversion, and MIMIC-IV/EHRSHOT need manual DUA approvals; this would consume most of the time before the 11.11 midpoint.
2. Under the MIMIC DUA the AI-assisted pair could not paste any data into an LLM, which cripples the AI4VIS part.

New setup:
- Model A: count / clinical-feature KNN.
- Model B: [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT), optionally [MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder).
- Data: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008); optional [PMC-Patients](https://huggingface.co/datasets/zhengyun21/PMC-Patients).

Caveats to resolve:
- **UCI is tabular, Bio_ClinicalBERT is a text model.** Records must be serialized to text ("70–80 year old female, diagnoses 250.4 …"). State this explicitly; it also gives a Phase II question: does the encoder respond to clinical content or to wording?
- **Mean-pooled Bio_ClinicalBERT is a weak similarity embedding.** MedCPT is trained for retrieval; consider it as the primary Model B and Bio_ClinicalBERT as the comparison.
- **UCI lab coverage is thin:** only categorical `A1Cresult` and `max_glu_serum`, no eGFR/creatinine. Pathways must be derived from later-visit ICD-9 codes, e.g. kidney 250.4x / 585, cardiovascular 410–414 / 428, eye 250.5x / 362.0x, neuro 250.6x. Only patients with repeat encounters (`patient_nbr`) have a later outcome.
- **Two datasets split the story.** PMC-Patients is free text with ground-truth similar patients but is not diabetes-specific and does not fit the count baseline. Keep UCI as primary; PMC-Patients at most as a side check of Model B.
- **Hypothesis H4** (Stanford → Boston transfer) no longer applies; candidate replacement: "a general clinical-text encoder is sensitive to record wording rather than clinical content."
- The proposal, slides and Moodle entry must be updated to the new setup.
