# AI4Vis log – Phase I (Pair A, AI-aided: Julian Bakan, Ahmad Hasan)

In Phase I, Pair A (Julian Bakan, Ahmad Hasan) works with generative AI and Pair B (Alfiia Ziganshina, Aqeel Jajja) works without it. The pairs swap in Phase II. For each AI-assisted step, record the prompt, the tool, what was suggested, and whether it was accepted, rejected or refined, and why.

| Date | Member | Tool | Prompt (short) | Suggestion | Decision | Why |
|---|---|---|---|---|---|---|
| 2026-10-26 | Julian | LLM | Generate Phase I baseline notebook, proposal PDF and slides | Full baseline (UCI + Bio_ClinicalBERT), 4 Phase I views | Accepted as starting point | To be reviewed, see [baseline_review.md](../docs/baseline_review.md) |
| 2026-10-28 | Julian | Antigravity AI | Refactor `shared/` baseline into static contract and build Pair A coordinated dashboard | 1) Decouple `shared/` into precomputed Parquet/JSON artifacts; 2) Eliminate synthetic label-leaking notes; 3) Build 4-view linked Dash app in `phase1/pair_A_ai_aided/` | Accepted and implemented | Essential to preserve Phase I 2x2 study protocol: Pair B gets static data with zero AI touch; Pair A gets rich coordinated visual analytics. |

---

## Session Documentation: Phase I Architecture & Protocol Decoupling

### 1. Rationale for Decoupling `shared/` from Pair B
Under the JKU study protocol, Pair B must remain strictly **human-only** (no generative AI for visualization design, hypothesis formulation, or code generation).
- **The Problem:** If `shared/` required executing deep learning models, configuring CUDA/PyTorch, or debugging transformer batchers, Pair B would either face environment barriers or be forced to use LLM debugging tools, violating the study condition.
- **The Solution:** We decoupled `shared/` into a one-time precomputation script (`shared/01_prepare_baseline_data.py`) that exports standardized, self-contained static data tables into `shared/data/` (`cohort_encounters.parquet`, `retrieval_results.parquet`, `counterfactual_flips.parquet`, and `baseline_metrics.json`). Pair B can read these tables directly with standard `pandas.read_parquet()` in any environment without installing PyTorch or touching GPU drivers.

### 2. Resolution of Label-Leaking Clinical Text
- **Audit Finding:** The initial prototype script had a helper function `derive_presenting_notes_from_inputs()` that synthesized symptom phrases ("elevated serum creatinine, proteinuria...", "chest tightness...") conditioned directly on `current_pathway` derived from diagnosis codes. This injected ground truth answers directly into Model B's text prompt.
- **Resolution:** All synthetic symptom generators were eliminated. Model B serialization now strictly uses authentic observable UCI fields only (`TEXT_TEMPLATE`):
  ```
  "Sex: {gender}, Age: {age}, Hospital Stay: {time_in_hospital} days, Admission Type: {admission_type}, Source: {admission_source}, Specialty: {medical_specialty}, Payer: {payer_code}, Labs: {num_lab_procedures}, Medications: {num_medications}, Diagnoses count: {number_diagnoses}, Max Glucose: {max_glu_serum}, A1C: {A1Cresult}, Insulin: {insulin}, Med Change: {change}, Diabetes Med: {diabetesMed}."
  ```
- **Ground Truth Definition:** Target pathway is defined strictly from the **subsequent** encounter's primary ICD-9 code (`diag_1` at visit $V_{t+1}$). The final visit for each patient is dropped (since no future admission exists). This guarantees zero intra-visit or forward label leakage.

### 3. State Management & Coordination Architecture for Pair A
Pair A's dashboard (`phase1/pair_A_ai_aided/dashboard/app.py`) coordinates four views:
- **View 1:** Jittered Multi-Channel Scatter Explorer (Model B agreement vs. comorbidity count).
- **View 2:** Reordered Error & Confusion Breakdown Matrix (side-by-side heatmaps ordered via hierarchical clustering on confusion vectors).
- **View 3:** Input-Space UMAP & Longitudinal Trajectories (connecting patient visits $V_1 \to V_2 \to \dots$).
- **View 4:** Counterfactual Neighbor Flips & Patient Inspector (bar chart of flips + detailed encounter inspection card).
- **State Management Choices:**
  - Implemented centralized client-side state using Dash `dcc.Store` (`store-selected-ids` and `store-active-cell`).
  - Cross-filtering supports lasso/box selections in View 1 and View 3, as well as discrete confusion cell clicks in View 2.
  - Selections highlight points while dimming unselected points (`opacity=0.06`), preserving spatial context while focusing attention.

---

## Project setup before the pair split

AI use by members of the human-only pair before the Phase I split, recorded for transparency. It concerned course material and project review, not Phase I analysis.

| Date | Member | Tool | Prompt (short) | Suggestion | Decision | Why |
|---|---|---|---|---|---|---|
| 2026-10-28 | Alfiia | Claude Code | Read course docs; review fallback plan and baseline figure | Leakage / baseline / scaling checks, visualization fixes | Implemented in `shared/` refactor | Correctness of baseline is prerequisite for both pairs. |

---

## Failure patterns observed

- **Silent synthetic fallbacks in prototypes:** Early code used synthetic generators if PyTorch was missing, masking real model behavior and generating synthetic findings. *Remedy: fail loudly and deterministically; eliminate fake random encoders.*
- **Discrete axis overplotting:** In View 1, agreement ($k/10$) and comorbidity count are discrete, causing hundreds of points to stack into invisible grids. *Remedy: configurable categorical jitter and density contour overlays.*

---

## Emerging best practices

- **Neutral shared data contracts:** Always export immutable Parquet/CSV artifacts for cross-pair studies so the experimental condition (AI vs. Human) remains strictly isolated to the visual analytics phase.
- **Hierarchical confusion matrix reordering:** Rather than alphabetical or arbitrary class ordering, clustering rows/columns by error profiles clusters mutually confused classes together (e.g. Cardiorenal overlap), making systematic model errors immediately apparent.
