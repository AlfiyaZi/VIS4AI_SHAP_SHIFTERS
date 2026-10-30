# Phase I — Pair A (AI-Aided Interactive Dashboard)

**Team SHAP_SHIFTERS | Pair A: Julian Bakan, Ahmad Hasan**

This directory contains the interactive Visual Analytics dashboard built for Phase I. The dashboard reads directly from the precomputed static tables in `shared/data/` and provides 4 linked visual views to explore black-box behavior across Model A (Count / Clinical Features KNN) and Model B (Bio_ClinicalBERT KNN).

---

## Quickstart: Running the Dashboard

Activate the project environment and run:

```bash
python phase1/pair_A_ai_aided/run_dashboard.py
```

Then open your browser at **`http://127.0.0.1:8050`**.

---

## The 4 Coordinated Views

### 1. View 1: Jittered Multi-Channel Scatter Explorer
- **Axes:** X = Model B (or Model A) Neighbor Agreement ($k=10$), Y = Comorbidity Count (`number_diagnoses`).
- **Encodings:** Color = HbA1c status (`None`, `Norm`, `>7`, `>8`), Shape = Patient Sex (Circle = Female, Diamond = Male), Facets / Small Multiples = True Next Pathway.
- **Overplotting Controls:** Toggle categorical jittering to unpack discrete steps; toggle density contours to observe dense comorbidity clusters.
- **Coordination:** Box or Lasso selection filters all other views to the selected patient cohort.

### 2. View 2: Reordered Error & Confusion Breakdown Matrix
- **Side-by-Side Heatmaps:** Model A vs. Model B vs. Majority Baseline.
- **Hierarchical Reordering:** Reordered via hierarchical clustering on confusion vectors so confusing classes sit adjacently.
- **Interaction:** Clicking any individual cell (e.g. True *Cardiovascular* predicted as *Kidney / Renal*) filters the entire dashboard to inspect those specific misclassified patients.

### 3. View 3: Input-Space UMAP & Longitudinal Trajectories
- **2D UMAP:** Computed strictly on input clinical features (black box closed).
- **Color Modes:** Agreement disparity ($\text{Agreement}_B - \text{Agreement}_A$), Model A agreement, Model B agreement, or ground truth pathway.
- **Driver Annotations:** Identifies the primary input feature separating UMAP clusters (e.g. medication `change`, association $0.80$).
- **Longitudinal Trajectories:** Toggle to connect repeat admissions ($V_1 \to V_2 \to \dots$) as directed vectors to reveal clinical state progression over time.

### 4. View 4: Counterfactual Neighbor Flips & Patient Inspector
- **Bar Chart:** Displays mean neighbor flips ($0$ to $10$) under:
  1. Sex Swap ($M \leftrightarrow F$)
  2. Age Shift ($+20$ years)
  3. HbA1c Elevation (None / Norm $\to >8$)
- **Patient Inspector:** Shows raw UCI encounter attributes, serialized summary text, and a stacked breakdown of retrieved neighbors for any selected encounter.
