# Phase I — Pair B (Human-Only Protocol)

Welcome Pair B (Alfiia Ziganshina, Aqeel Jajja).

In accordance with the study protocol for Phase I, **Pair B works strictly human-only** (no generative AI for visualization design, exploratory analysis, or analytical code generation; AI tools are restricted strictly to standard IDE code completion and API documentation lookup).

To ensure you can focus entirely on visual analysis and human exploration without touching deep learning code, CUDA environments, or model execution, **all model outputs and baseline artifacts are fully precomputed and stored as clean static tables in `shared/data/`**.

---

## 1. Static Data Contract (`shared/data/`)

You do **not** need to run `01_prepare_baseline_data.py` or install PyTorch. All necessary tables are precomputed:

| File | Format | Description |
|---|---|---|
| `shared/data/cohort_encounters.parquet` | Parquet / CSV | 30,248 repeat-admission encounters across ~16.7k patients. Contains all demographic attributes, lab counts, medications, diagnoses, ground truth next pathway, and 2D UMAP coordinates (`umap_x`, `umap_y`). |
| `shared/data/retrieval_results.parquet` | Parquet / CSV | Retrieval outputs for $k=10$ nearest neighbors: `model_a_neighbors` (list of encounter IDs), `model_b_neighbors`, neighbor agreement scores, and predicted pathways. |
| `shared/data/counterfactual_flips.parquet` | Parquet / CSV | Neighbor flip counts ($0$ to $10$) per encounter under Sex swap ($M \leftrightarrow F$), Age shift ($+20$ years), and HbA1c elevation (None/Norm $\to >8$). |
| `shared/data/baseline_metrics.json` | JSON | Overall benchmark metrics (Model A vs. Model B vs. Majority vs. Random). |

---

## 2. Quickstart: Loading in Pandas

In your own notebooks or Python scripts (e.g., inside `phase1/pair_B_human_only/`), simply load the data directly using Pandas:

```python
import pandas as pd
from pathlib import Path

# Path to shared precomputed data
DATA_DIR = Path("../../shared/data")

# 1. Load cohort encounters
cohort = pd.read_parquet(DATA_DIR / "cohort_encounters.parquet")
print(f"Cohort shape: {cohort.shape}")

# 2. Load retrieval results
retrieval = pd.read_parquet(DATA_DIR / "retrieval_results.parquet")
print(f"Retrieval results shape: {retrieval.shape}")

# 3. Load counterfactual flips
flips = pd.read_parquet(DATA_DIR / "counterfactual_flips.parquet")
print(f"Flips shape: {flips.shape}")

# 4. Merge into a single master analysis dataframe
df = cohort.merge(
    retrieval.drop(columns=["patient_nbr", "true_next_pathway"]),
    on="encounter_id",
    validate="one_to_one"
).merge(
    flips.drop(columns=["patient_nbr"]),
    on="encounter_id",
    validate="one_to_one"
)

print(f"Merged analysis table: {df.shape}")
```

---

## 3. Key Research Questions for Pair B to Explore

1. **Error Distribution (RQ 1)**:
   - Does Model B (Bio_ClinicalBERT) or Model A (Clinical Features KNN) perform better on rare pathways (e.g. *Eye / Neurological* or *Kidney / Renal*) vs. the majority class (*Uncomplicated / Other*)?
   - Compare confusion patterns and per-pathway recall without generative assistance.
2. **Demographic Sensitivity (RQ 2 & 3)**:
   - Examine `model_a_flips_sex_swap` vs. `model_b_flips_sex_swap`: Why does Model A's one-hot gender vector flip almost 9 out of 10 neighbors, while Model B's semantic embedding flips only ~4?
   - Look at age shifts: how do demographic perturbations compare to clinical perturbations (`hba1c_elevation`)?
3. **Longitudinal Trajectories**:
   - Trace repeat visits for individual patients (`patient_nbr`) using `visit_index` ($V_1 \to V_2 \to \dots$) in `umap_x` and `umap_y` to see how patient health state drifts over time.

---

## 4. Verification Check

Before writing your analysis, you can inspect `shared/02_phase1_shared_baseline.ipynb` to verify that all data files are loaded and that baseline accuracies match across the team.
