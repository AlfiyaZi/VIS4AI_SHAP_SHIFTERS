# AI4Vis log – Phase I (Pair A, AI-aided: Julian Bakan, Ahmad Hasan)

In Phase I, Pair A (Julian Bakan, Ahmad Hasan) works with generative AI and Pair B (Alfiia Ziganshina, Aqeel Jajja) works without it. The pairs swap roles in Phase II.

This log documents all AI-assisted steps in accordance with the JKU AI & Visualization Lab WS26 guidelines, specifically categorizing interactions under the **5 GenAI Roles** introduced in Lab 03:
1. **Role 1: GenAI as Tool to Transform Your Data**
2. **Role 2: GenAI as Collaborator to Design & Implement Visualizations**
3. **Role 3: GenAI as Collaborator to Critique Design & Findings**
4. **Role 4: GenAI as Collaborator to Steer Your Investigation**
5. **Role 5: GenAI as Tool to Interpret Observations**

---

## 1. Structured AI Interaction Log

| Date | Member | Tool | Lab 03 Role | Prompt (Summary) | AI Suggestion | Decision & Action | Rationale & Reflection |
|---|---|---|---|---|---|---|---|
| 2026-10-26 | Julian | LLM | Role 1 (Transform) | Generate Phase I baseline notebook, proposal PDF, and initial slides | Initial prototype with 4 views and synthetic symptom text generator | Accepted as starting draft | Flagged in code review for label leakage; required clean rebuild. |
| 2026-10-28 | Julian | Antigravity AI | Role 1 & 2 (Transform & Design) | Refactor `shared/` baseline into a static data contract and build Pair A coordinated dashboard | 1) Decouple `shared/` into precomputed Parquet/JSON artifacts; 2) Eliminate synthetic label-leaking notes; 3) Build 4-view linked Dash app | Accepted and implemented | Essential to preserve Phase I 2x2 study protocol: Pair B gets static data with zero AI touch; Pair A gets rich coordinated visual analytics. |
| 2026-11-02 | Julian | Antigravity AI | Role 3 (Critique) | Audit data loading performance, schema integrity, and discrete overplotting | 1) Add `use_threads=False` to `pd.read_parquet()` to eliminate PyArrow thread locks; 2) Add uniform jitter and 2D density contour overlays to View 1; 3) Modularize dashboard into `layout.py`, `callbacks.py`, and `test_dashboard.py` | Accepted and implemented | Discovered that PyArrow spawns thread pools that hang in sandboxed Windows environments; passing `use_threads=False` reduced load time from timeout to 0.11s. |
| 2026-11-02 | Ahmad | Antigravity AI | Role 4 (Steer) | Investigate retrieval disparity: why do Model A and Model B share only 22.4% neighbors with identical ~63% accuracy? | Focus analysis on counterfactual flip divergence (View 4) and confusion matrix reordering (View 2). Inspect whether Model A relies on demographic shortcuts while Model B relies on text phrasing. | Accepted and integrated into dashboard exploration flow | Steered team away from aggregate metric comparisons towards subgroup counterfactual auditing. |
| 2026-11-02 | Julian | Antigravity AI | Role 5 (Interpret) | How to interpret HDBSCAN clusters on clinical input UMAP without manual cherry-picking? | Compute statistical association metrics: Cramér's V for categorical features and correlation ratio ($\eta$) for numeric features against cluster labels. Annotate cluster medians with top driver. | Accepted and implemented | Cluster annotations now convey statistically grounded drivers (e.g. medication `change`, association 0.80) rather than subjective visual impressions. |

---

## 2. Session Documentation: Architecture & Analytical Evolution

### 2.1 Rationale for Decoupling `shared/` from Pair B
Under the JKU study protocol, Pair B must remain strictly **human-only** (no generative AI for visualization design, hypothesis formulation, or code generation).
- **The Problem:** If `shared/` required executing deep learning models, configuring CUDA/PyTorch, or debugging transformer batchers, Pair B would either face environment barriers or be forced to use LLM debugging tools, violating the study condition.
- **The Solution:** We decoupled `shared/` into a one-time precomputation script (`shared/01_prepare_baseline_data.py`) that exports standardized, self-contained static data tables into `shared/data/` (`cohort_encounters.parquet`, `retrieval_results.parquet`, `counterfactual_flips.parquet`, and `baseline_metrics.json`). Pair B reads these tables directly with standard `pandas.read_parquet()` in any environment without installing PyTorch or touching GPU drivers.

### 2.2 Resolution of Label-Leaking Clinical Text
- **Audit Finding:** The initial prototype script had a helper function `derive_presenting_notes_from_inputs()` that synthesized symptom phrases ("elevated serum creatinine, proteinuria...", "chest tightness...") conditioned directly on `current_pathway` derived from diagnosis codes. This injected ground truth answers directly into Model B's text prompt.
- **Resolution:** All synthetic symptom generators were eliminated. Model B serialization now strictly uses authentic observable UCI fields only (`TEXT_TEMPLATE`):
  ```
  "Sex: {gender}, Age: {age}, Hospital Stay: {time_in_hospital} days, Admission Type: {admission_type}, Source: {admission_source}, Specialty: {medical_specialty}, Payer: {payer_code}, Labs: {num_lab_procedures}, Medications: {num_medications}, Diagnoses count: {number_diagnoses}, Max Glucose: {max_glu_serum}, A1C: {A1Cresult}, Insulin: {insulin}, Med Change: {change}, Diabetes Med: {diabetesMed}."
  ```
- **Ground Truth Definition:** Target pathway is defined strictly from the **subsequent** encounter's primary ICD-9 code (`diag_1` at visit $V_{t+1}$). The final visit for each patient is dropped (since no future admission exists). This guarantees zero intra-visit or forward label leakage.

### 2.3 Modular Dashboard Architecture & State Management (Pair A)
Pair A's dashboard (`phase1/pair_A_ai_aided/dashboard/`) coordinates four complementary views:
- **View 1 (Investigative Scatter Explorer):** Model B agreement vs. comorbidity count (`number_diagnoses`), faceted by true pathway, colored by HbA1c, shaped by patient sex, with configurable uniform jittering and 2D density contour overlays to unpack discrete overplotting.
- **View 2 (Reordered Error Matrix):** Side-by-side heatmaps (Model A vs. Model B vs. Majority Baseline) reordered via average linkage hierarchical clustering on error profiles with optimal leaf ordering (Alsallakh et al. 2017 style). Clicking any cell isolates those misclassified encounters across the entire dashboard.
- **View 3 (Input Space & Trajectories):** 2D UMAP projection strictly on 127-D clinical input features (black box closed). Features color modes (Agreement Disparity $B - A$, Model A Agreement, Model B Agreement, Pathway, HDBSCAN Cluster, Cluster Driver), longitudinal directed admission vectors ($V_1 \to V_2 \to V_3$), and cluster median driver annotations.
- **View 4 (Counterfactual Flips & Patient Inspector):** Grouped bar chart with 95% CIs comparing neighbor replacement under Sex Swap, Age Shift (+20 y), and HbA1c elevation, plus an encounter inspection card with a 10-neighbor ground-truth pathway breakdown.
- **State Management Choices:**
  - Implemented centralized client-side state using Dash `dcc.Store` (`store-selected-ids` and `store-active-cell`).
  - Cross-filtering supports lasso/box selections in View 1 and View 3, as well as discrete confusion cell clicks in View 2.
  - Selections highlight points while dimming unselected points (`opacity=0.06`), preserving spatial context while focusing attention.
  - Subsampling: Trajectories render a representative sample of $N=35$ repeat patients when displaying all 30,248 visits, but render all matching trajectories once a patient subset is selected.

---

## 3. Pre-Split Record (Human-Only Pair)

AI use by members of the human-only pair before the Phase I split, recorded for transparency. It concerned course material and project review, not Phase I analysis.

| Date | Member | Tool | Prompt (Summary) | Suggestion | Decision | Why |
|---|---|---|---|---|---|---|
| 2026-10-28 | Alfiia | Claude Code | Read course docs; review fallback plan and baseline figure | Leakage / baseline / scaling checks, visualization fixes | Implemented in `shared/` refactor | Correctness of baseline is prerequisite for both pairs. |

---

## 4. Failure Patterns Observed

1. **Silent Synthetic Fallbacks in Prototypes:**
   - *Failure:* Early prototype code used synthetic generators if PyTorch was missing, masking real model behavior and generating synthetic findings.
   - *Remedy:* Fail loudly and deterministically; eliminate fake random encoders.
2. **Discrete Axis Overplotting:**
   - *Failure:* In View 1, agreement ($k/10$) and comorbidity count are discrete, causing hundreds of points to stack into invisible grids.
   - *Remedy:* Configurable uniform categorical jitter ($\pm 0.035$ X, $\pm 0.38$ Y) and 2D density contour overlays.
3. **PyArrow Worker Thread Lockup in Restricted Environments:**
   - *Failure:* Calling `pd.read_parquet()` with default multi-threading on Windows sandbox or virtualized CI environments spawned background worker threads that deadlocked during initialization.
   - *Remedy:* Explicitly pass `use_threads=False` to `pd.read_parquet()`, reducing read time across all 30k rows to 0.11 seconds.
4. **Subjective Cluster Over-Interpretation in Projections:**
   - *Failure:* Attempting to interpret 2D UMAP clusters by eyeballing positions or cluster distances (violating Lecture 03 guidance).
   - *Remedy:* Ground cluster labeling in objective statistical association ranking (Cramér's V and correlation ratio).

---

## 5. Emerging Best Practices

- **Neutral Shared Data Contracts:** Always export immutable Parquet/CSV artifacts for cross-pair studies so the experimental condition (AI vs. Human) remains strictly isolated to the visual analytics phase.
- **Hierarchical Confusion Matrix Reordering:** Rather than alphabetical or arbitrary class ordering, clustering rows/columns by error profiles clusters mutually confused classes together (e.g. Cardiorenal overlap), making systematic model errors immediately apparent.
- **Coordinated Dimming vs. Hard Filtering:** In scatter plots and projections, dimming unselected points (`opacity=0.06`) is superior to removing them completely because it maintains the global structural context of the patient space.
- **Automated Headless Testing for Visual Dashboards:** Unit tests verifying figure dictionary compilation and data loader schemas ensure dashboards run out-of-the-box without manual UI clicking.
