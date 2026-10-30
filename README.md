# VIS4AI_SHAP_SHIFTERS

Team **SHAP Shifters**, AI and Visualization Lab WS26 (JKU Linz, Visual Data Science Lab).

## Task and Models

**Task:** Patient-similarity retrieval for clinical-trial cohort matching. Given a diabetes encounter, a $k$-nearest-neighbor model ($k = 10$) returns the most similar patients. A neighbor set is considered clinically congruent when it shares the patient's later complication pathway (*Kidney / Renal*, *Cardiovascular*, *Eye / Neurological*, or *Uncomplicated / Other*).

| Model | Representation | Description |
|---|---|---|
| **Model A** | Count / Clinical-Feature KNN | Standardized numeric features (vitals, labs, stays) + one-hot categorical features (127-D) |
| **Model B** | [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) KNN | 768-D mean-pooled embeddings over serialized authentic encounter records |

**Dataset:** [UCI Diabetes 130-US Hospitals 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI ID 296). Filtered strictly to repeat-admission patients ($N=30,248$ encounters across $16,773$ unique patients), with ground truth defined as the subsequent admission's primary ICD-9 diagnosis pathway.

---

## Research Questions

**Phase I – How do the models behave?** (Black box closed)
1. Do retrieved neighbors share the query patient's later complication pathway, and where in patient space does retrieval fail?
2. Are neighbors chosen by demographic shortcuts (sex, age) rather than clinical state?
3. How sensitive are neighbors to small clinical vs. demographic edits (counterfactual neighbor flips)?

**Phase II – What did the models learn?** (Black box open)
- To be refined after the midpoint, investigating representations, attention patterns, and latent feature attribution.

---

## Study Structure & 2x2 Protocol

The two pairs switch roles between Phase I and Phase II as mandated by the course protocol:

| Pair | Members | Phase I | Phase II |
|---|---|---|---|
| **Pair A** | Julian Bakan, Ahmad Hasan | **AI-aided** (linked visual dashboard) | Human-only |
| **Pair B** | Alfiia Ziganshina, Aqeel Jajja | **Human-only** (independent manual VIS) | AI-aided |

```
shared/                              Common static data contract
  01_prepare_baseline_data.py        Precomputation script (UCI download -> shared/data/)
  02_phase1_shared_baseline.ipynb    Minimal verification contract (no dashboard code)
  data/                              Static artifacts (Parquet + CSV + JSON)
    cohort_encounters.parquet        Encounter metadata, vitals, next pathway, UMAP (X, Y)
    retrieval_results.parquet        k=10 neighbor IDs, agreements, predicted pathways
    counterfactual_flips.parquet     Neighbor flips under Sex swap, Age shift, HbA1c elevation
    baseline_metrics.json            Cohort benchmarks (A vs. B vs. Majority vs. Random)
phase1/
  pair_A_ai_aided/                   AI-aided interactive dashboard (Dash / Plotly)
    run_dashboard.py                 Dashboard launcher
    dashboard/                       4 coordinated visual views with cross-filtering
  pair_B_human_only/                 Human-only visual analytics workspace
    README.md                        Step-by-step instructions for Pair B (pure Pandas)
phase2/
  pair_A_human_only/
  pair_B_ai_aided/
ai4vis/                              AI interaction logs and reflection
  phase1_ai_log.md
  phase2_ai_log.md
  reflection.md
presentation/
  midpoint/                          Progress video (<= 5 min) slides
  final/                             Final slide deck (10-min recorded presentation)
  presentation_video.json
docs/                                Course notes, decision logs, and audit reviews
environment.yml
team-info.json                       Required by course
workload.json                        Required by course
```

*Note: Large cached downloads (`data_cache/`) and raw binaries are git-ignored.*

---

## Setup & Quickstart

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate shap_shifters

# 2. Run shared baseline verification
jupyter lab shared/02_phase1_shared_baseline.ipynb

# 3. Launch Pair A Dashboard
python phase1/pair_A_ai_aided/run_dashboard.py
```
