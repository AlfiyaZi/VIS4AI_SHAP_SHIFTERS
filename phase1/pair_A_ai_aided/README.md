# Phase I — Pair A (AI-Aided Interactive Dashboard & Exploration Guide)

**Team SHAP_SHIFTERS | Pair A: Julian Bakan, Ahmad Hasan**

This directory contains Pair A's analytical workspace for Phase I:
1. **`run_dashboard.py`**: The multi-tab interactive Dash studio featuring coordinated cross-filtering across 4 primary views, calibration curve analysis, model disparity battlegrounds, demographic audits, and dynamic cohort profiling.
2. **`dataset_and_models_guide.ipynb`**: The comprehensive, 28-cell interactive Jupyter notebook that walks through the clinical dataset, models, 15 input features, 4 complication pathways, calibration cliff, and Phase I behavioral discoveries.
3. **`test_dashboard.py`**: Automated headless test suite verifying all 14 unit tests, layout endpoints, and callback persistence.

---

## Quickstart

### Option 1: Launch the Interactive Web Studio (Dash)
```bash
python phase1/pair_A_ai_aided/run_dashboard.py
```
Open **`http://127.0.0.1:8050`** in your browser.

### Option 2: Explore the Jupyter Guide
```bash
jupyter lab phase1/pair_A_ai_aided/dataset_and_models_guide.ipynb
```

---

## Dashboard Architecture: 4 Analytical Modules

### Tab 1: 🎬 Curated Storyline (5-Min Video Walkthrough — 6 Chapters)
- **Chapter 1: Top-Level Summary — Reordered Confusion Matrices & 0.0% Minority Recall**: Hierarchically reordered heatmaps (Alsallakh et al. 2017) showing that both $k=10$ cosine KNN models achieve ~63.0% accuracy (losing to the 66.36% majority baseline) and suffer **0.0% recall** on Kidney/Renal ($n=757$) and Eye/Neurological ($n=563$).
- **Chapter 2: The Calibration Cliff & $\ge 8/10$ Confidence Threshold**: Test-time vote share plotted against empirical accuracy with 95% Wilson CIs, encounter density bars, and annotated signal zone ($\ge 80\%$, covering ~30.6% of stays, $n=9,240$) where accuracy (68.2%–77.6%) finally beats the 66.36% majority baseline.
- **Chapter 3: High-Dimensional Manifold & Clinical Cluster Drivers**: 2D UMAP projection strictly on 127-D input features with HDBSCAN clusters, longitudinal trajectories ($V_1 \to V_2 \to V_3$), and a Cramér's V ranking table showing Medication Change (`change`, $V=0.803$) and Diabetes Medication (`diabetesMed`, $V=0.796$) as primary organizational drivers.
- **Chapter 4: Perturbation Sensitivity (Mirror-Image Representation Artifacts)**: Grouped bar chart comparing Sex Swap (Model A replaces **8.86 / 10** neighbors due to unscaled binary one-hot columns vs. Model B **4.15 / 10**), Age Shift (+20y: **5.42** vs. **5.11**), and HbA1c Elevation (Model B replaces **8.22 / 10** neighbors due to WordPiece token pooling shifts vs. Model A **4.36 / 10**).
- **Chapter 5: Model Disagreements ($n=5,918$) & Subgroup Base-Rate Deconfounding**: Side-by-side audit showing Model A winning **58.5% vs. 34.9%** on younger diabetics ($<50$ yrs, $n=482$) and Model B winning **51.2% vs. 46.0%** on elderly patients ($80+$ yrs, $n=1,321$), plus subgroup accuracy lift proving the 60.0% African American vs. 63.7% Caucasian raw gap is a base-rate artifact ($-3.1\text{ pp}$ vs. $-3.3\text{ pp}$ lift).
- **Chapter 6: Side-by-Side 10-NN Dossier & 22.4% Neighbor Overlap**: Interactive encounter inspection table with cosine similarities ($1 - \text{dist}$), authentic pathway colors, and shared neighbor badges, proving models share only 22.4% of retrieved neighbors despite identical aggregate accuracy.

### Tab 2: 🎛️ Primary Retrieval Studio (Views 1 – 4)
- **View 1: Agreement vs. Comorbidity Explorer**: Scatter plot faceted by True Next Pathway with categorical jitter, density contours, HbA1c color encoding, and patient sex symbols.
- **View 2: Reordered Error & Confusion Breakdown Matrix**: Model A vs. Model B vs. Majority Baseline heatmaps with hierarchical clustering (optimal leaf ordering). Click any cell to cross-filter.
- **View 3: 2D UMAP Manifold & Longitudinal Trajectories**: 2D projection of clinical inputs with statistical driver annotations (Cramér's V) and repeat-admission patient trajectories ($V_1 \to V_2 \to V_3$).
- **View 4: Counterfactual Neighbor Flips & Patient Dossier**: Stability under Sex swap, Age shift, and HbA1c elevation + single-patient dossier inspector.

### Tab 3: ⚖️ Model Disparity & Reliability Audit
- **Test-Time Vote-Share Calibration Curve (with 95% Wilson CIs)**: Evaluates empirical accuracy against label-free test-time prediction confidence (`vote_share`, from 40% to 100%). Proves that KNN only beats the 66.4% majority baseline when $\ge 8/10$ neighbors agree on the predicted class (`vote_share >= 80%`, covering ~31% of stays and reaching 68–78% accuracy).
- **Model A vs. Model B Disagreement Battleground ($n=5,918$ Conflicts)**: 100%-normalized win-rate bars across encounters where models disagree, annotated with per-bracket sample sizes ($n$). Shows Tabular KNN winning 58.5% vs. 34.9% on patients under 50 ($n=482$), while `Bio_ClinicalBERT` edges out 51.2% vs. 46.0% on elderly patients ($80+$, $n=1,321$).
- **Subgroup Accuracy Lift vs. Subgroup Majority Baseline (with 95% Wilson CIs)**: Controls for subgroup base-rate differences (`% Uncomplicated / Other`) across race and admission pathway. Proves that lower raw accuracy on African American stays (60.0% vs. 63.7% Caucasian) is explained by a lower majority-class base rate (63.1% vs. 66.9%), yielding an identical relative lift ($-3.1\text{ pp}$ vs. $-3.3\text{ pp}$).

### Tab 4: 🧬 Dynamic Cohort Profiler (Selection vs. Baseline)
- **Normalized Clinical Complexity Profiler (`Baseline = 100%`)**: Indexes all 6 clinical complexity metrics to 100% of the hospital mean so small-scale counts (`number_emergency = 0.33`) and large-scale counts (`num_lab_procedures = 42.54`) are directly comparable on the same horizontal axis, with raw averages and relative `%` shifts displayed on every bar.
- **Complication Outcomes Distribution Shift**: Shows shifts in true complication proportions between the active selection and population baseline.
