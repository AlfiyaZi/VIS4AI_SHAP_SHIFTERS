# AI Interaction Log — Phase I (Pair A, AI-Aided)

**AI-Aided Pair (Pair A):** Julian Bakan (`k11913166`), Ahmad Hasan (`k51821183`)  
**Human-Only Pair (Pair B):** Alfiia Ziganshina (`k12456306`), Aqeel Jajja (`k12554936`)*  
**Course:** AI and Visualization (WS26), JKU Linz (Visual Data Science Lab)  
**Team:** SHAP Shifters  

> [!NOTE]
> **\*Note on Team Participation:** Throughout Phase I, Pair B's human-only pipeline was carried out entirely by Alfiia Ziganshina (`k12456306`). Aqeel Jajja (`k12554936`) remained inactive and contributed no code or analysis to the project (0 commits across the repository).

---

## 1. Overview of Key AI Interactions (Chronological Summary)

In Phase I, Pair A used GenAI (Claude 3.5 Sonnet and Gemini via Antigravity) to write code, debug callbacks, and brainstorm visualization ideas. Below is the complete chronological summary of our interactions:

| # | Session & Date | AI Role | Prompt / User Goal | AI Proposal | Decision & Action | Outcome & Practical Impact |
|---|---|---|---|---|---|---|
| **P1** | **S1: 2026-10-28** | Investigation | Formulate clinical patient-matching retrieval task comparing two distinct encoders. | Proposed $k=10$ KNN retrieval on UCI Diabetes comparing 127-D tabular features vs. 768-D Bio_ClinicalBERT text embeddings across 4 complication pathways. | **Accepted with refinement** | Defined longitudinal target strictly as the *next admission's* diagnosis ($V_{t+1}$), excluding single-visit patients. |
| **P2** | **S2: 2026-10-28** | Data Prep | Build reproducible shared baseline pipeline and decouple data artifacts for Pair B. | Generated extraction script creating 4 static Parquet/JSON tables, but added synthetic clinical notes. | **Rejected fake notes; accepted schema** | **Purged label leakage**: deleted synthetic notes that gave away the diagnosis; locked inputs to 15 real UCI fields. |
| **P3** | **S3: 2026-11-02** | UI Design | Modularize Dash app into separate components and explain 2D UMAP cluster structure. | Modularized into `layout.py`, `callbacks.py`, `figures.py`, `data_layer.py`; suggested HDBSCAN and Cramér's V for cluster analysis. | **Accepted and implemented** | Fixed a PyArrow sandbox thread-lock bug (`use_threads=False`); proved UMAP clusters split by medication flags, not disease. |
| **P4** | **S4: 2026-11-03** | Data Prep | Create a detailed interactive guide reviewing all features, distributions, and model metrics. | Built [`dataset_and_models_guide.ipynb`](../phase1/pair_A_ai_aided/dataset_and_models_guide.ipynb) (28 cells) with cohort funnels, boxplots, and encounter cards. | **Accepted** | Provided Pair A with an educational reference explaining why 63% accuracy is driven by the 66.4% majority class. |
| **P5** | **S4: 2026-11-03** | Interpretation | Understand the clinical meaning of encounters and verify findings against dataset literature. | Cross-referenced findings with the original dataset paper (Strack et al., 2014), explaining that administrative billing flags dominate records. | **Accepted** | Grounded our findings: both models struggle on rare complications because billing data reflects hospital procedures, not biology. |
| **P6** | **S4: 2026-11-03** | UI Debugging | Fix Dash interactive selection bugs: lasso resetting state and cross-filter conflicts. | Diagnosed circular callback race condition; added `dash.no_update` guards and consistent `uirevision` keys. | **Accepted & verified with tests** | Fixed selection persistence across views and added automated headless unit tests (`test_dashboard.py`). |
| **P7** | **S4: 2026-11-03** | Design & Steer | Expand dashboard into a comprehensive clinical auditing studio with subgroup and confidence views. | Designed 3 new views: Confidence-Accuracy Calibration, Model Disagreement Battleground, and Dynamic Cohort Profiler. | **Accepted and expanded** | Expanded dashboard from a basic 4-view prototype into a deep multi-tab auditing studio. |
| **P8** | **S4: 2026-11-03** | Critique | Perform statistical self-audit of new views: check for metric validity and confounding. | Identified two critical flaws: circular calibration against true labels, and base-rate confounding in raw racial accuracy. | **Modified calculations** | Switched calibration to test-time `vote_share` (found $\ge 8/10$ cliff); switched fairness to base-rate Accuracy Lift. |
| **P9** | **S5: 2026-11-08** | Critique | Compare two diverging team branches (Ahmad's simple scroll vs. Julian's multi-tab studio). | Evaluated tradeoffs: Ahmad's branch was clean for a short video but lacked depth; Julian's branch had full depth but higher complexity. | **Refined architecture** | Planned synthesis using the "Martini Glass" pattern: Curated Storyline for the video + Exploratory Studio for grading depth. |
| **P10** | **S5: 2026-11-08** | UI Scaffolding | Implement unified Dual-Mode dashboard and automated test suite on `phase1/pair-a-ai-aided`. | Built Tab 1 (6-Chapter Storyline) and Tabs 2–4 (Linked Studio, Disparity Audit, Profiler); built 14-test test suite. | **Accepted and committed** | Unified both teammates' work cleanly; all 14 automated unit tests pass. |
| **P11** | **S5: 2026-11-09** | Presentation | Review presentation options for 5-minute video: evaluate draft slide deck vs. live dashboard demos. | AI drafted a 10-slide LaTeX Beamer presentation and markdown speaking script. | **Rejected AI slides completely** | Threw out generic AI slides; decided to record authentic live screen captures of our interactive Dash application. |

---

## 2. In-Depth Case Studies: Human Steering & Catching AI Mistakes

### Case 1: Purging Synthetic Label-Leaking Clinical Notes (Data Preparation)
* **What happened:** In the early baseline draft, the AI added a helper function `derive_presenting_notes_from_inputs()` to synthesize doctor notes for the text model (e.g., *"patient presents with acute chest pain and elevated creatinine"*).
* **The issue:** The AI created these symptom strings by reading the encounter's *diagnosis codes*. When Model B processed this text, it was effectively reading the target diagnosis, artificially inflating accuracy.
* **Our action:** We completely deleted the synthetic text generator, restricted the inputs strictly to 15 real UCI fields, and ensured ground truth was strictly taken from the *next* admission ($V_{t+1}$).

### Case 2: Explaining 2D UMAP Clusters with Cramér's V (Interpretation)
* **What happened:** When projecting encounters into 2D via UMAP, the data split into several distinct clusters. We needed to know what caused these clusters without subjective visual guessing.
* **The AI suggestion:** The AI suggested running HDBSCAN clustering and computing non-parametric correlations—Cramér's V (for categorical features) and correlation ratios $\eta$ (for numeric features)—against the cluster IDs.
* **The result:** The calculation proved mathematically that the clusters are driven by administrative medication flags—**Medication Change (`change`, $V=0.803$)** and **Prescribed Medication (`diabetesMed`, $V=0.796$)**—rather than patient disease outcomes.

### Case 3: Correcting the Circular Calibration Plot (Evaluation Metric)
* **What happened:** We asked AI to implement a calibration curve to test whether higher neighbor agreement leads to higher prediction accuracy.
* **The issue:** The AI binned stays by neighbor agreement with the *true future label* $y$. In $k=10$ KNN, having $\ge 6/10$ neighbors matching the true label means the model predicts correctly by definition (100% accuracy). This produced a flat, circular 100% line that is impossible to know at test time.
* **Our action:** We changed the metric to test-time **`vote_share`** (the fraction of neighbors agreeing on the *predicted* class $\hat{y}$). This revealed the true clinical pattern: accuracy remains around 59.6% (worse than the 66.4% majority baseline) until neighbor agreement reaches 8/10 or higher.

### Case 4: Controlling for Subgroup Base Rates in Fairness (Statistical Confounding)
* **What happened:** The AI generated a raw demographic accuracy chart showing that Model A and Model B had ~3.7% lower accuracy on African American patients compared to Caucasian patients, suggesting algorithmic bias.
* **The issue:** We checked the underlying distributions and found that the majority class (`Uncomplicated`) has a lower baseline prevalence among African American admissions (63.1%) than Caucasian admissions (66.9%).
* **Our action:** We recalculated accuracy as **Accuracy Lift over each group's baseline**. Both groups experienced almost the exact same relative drop ($-3.1$ vs. $-3.3$ pp), showing that the raw difference was caused by cohort composition rather than algorithmic discrimination.

### Case 5: The "Martini Glass" Synthesis and Rejecting AI Slides (UI & Presentation)
* **What happened:** To prepare for the 5-minute video, teammate Ahmad built a simplified single-page layout, but it removed the UMAP projection and profiler. The AI then generated a 10-slide LaTeX Beamer deck full of generic text bullet points.
* **Our action:** We rejected the AI slide deck completely because the course requires showing authentic, curated visual analytics. We synthesized both branches on `phase1/pair-a-ai-aided` using the **Martini Glass pattern**: Tab 1 provides a 6-chapter guided storyline for the 5-minute video, while Tabs 2–4 keep the full interactive studio with linked brushing, backed by 14 automated unit tests.

---

## 3. Summary of GenAI Strengths and Limitations

### Where GenAI Worked Well
1. **Scaffolding and Coding Speed**: Rapidly generated Dash UI layouts, callback logic, and Plotly figure code, reducing development time from weeks to days.
2. **Statistical Brainstorming**: Suggested useful metrics (like Cramér's V and Wilson score confidence intervals) to ground visual findings with real numbers.
3. **Automating Repetitive Tasks**: Automated counterfactual perturbations (Sex swap, Age shift, HbA1c change) across 30,248 stays, revealing that Model A is sensitive to sex edits while Model B is sensitive to text edits.

### Where GenAI Struggled
1. **Inventing Synthetic Shortcuts**: Naturally generated synthetic clinical notes that leaked future labels, requiring careful human auditing.
2. **Circular Evaluation Logic**: Generated plots that looked plausible visually but were mathematically circular (such as calibration against ground truth).
3. **Superficial Interpretations**: Jumped to conclusions about "demographic bias" without checking for basic dataset base-rate differences.

---

## 4. Complete Chronological Prompt Log (Detailed Dossiers)

Below is the detailed record of every individual prompt interaction across all five project sessions, showing our prompts, what AI provided, and what actions we took:

### Phase A: Problem Definition & Data Preprocessing (Sessions 1–2)

#### Prompt P1: Clinical Task Formulation & Dual Models
* **Date & Session:** 2026-10-28 (Session 1)
* **Goal:** Formulate our project problem for group registration.
* **What we asked:** *"Help us define a patient retrieval task on the UCI Diabetes dataset comparing a tabular feature baseline against Bio_ClinicalBERT text embeddings across complication pathways."*
* **What AI provided:** Proposed a $k=10$ nearest-neighbor retrieval task predicting 4 future complication pathways (*Uncomplicated*, *Cardiovascular*, *Kidney*, *Eye/Neuro*), comparing 127-D tabular cosine distance against 768-D Bio_ClinicalBERT text embeddings.
* **Our decision & action:** Accepted with refinement. We explicitly anchored the target to the *next admission's* primary diagnosis ($V_{t+1}$) and removed single-visit patients so the problem represents genuine longitudinal prediction.

#### Prompt P2: Shared Baseline Pipeline & Purging Synthetic Notes
* **Date & Session:** 2026-10-28 (Session 2)
* **Goal:** Create the shared data preparation pipeline (`shared/01_prepare_baseline_data.py`) and static artifacts for Pair B.
* **What we asked:** *"Build a clean preprocessing pipeline for the UCI Diabetes dataset that outputs static Parquet tables so Pair B can work directly with Pandas without needing GPUs or PyTorch."*
* **What AI provided:** Wrote the extraction script generating 4 Parquet/JSON files, but included a helper function `derive_presenting_notes_from_inputs()` that synthesized symptom notes from diagnosis codes.
* **Our decision & action:** Rejected the synthetic clinical notes (label leakage); accepted the Parquet schema. We deleted all synthetic text generators, locked Model B to 15 authentic UCI fields, and ensured Pair B had clean, verified data.

---

### Phase B: Dashboard Scaffolding & Initial Views (Session 3)

#### Prompt P3: Modular Architecture & UMAP Associations
* **Date & Session:** 2026-11-02 (Session 3)
* **Goal:** Restructure the monolithic dashboard script into clean modules and investigate 2D UMAP clusters.
* **What we asked:** *"Help us modularize our Dash app into separate files (layout, callbacks, figures, data layer) and suggest a way to analyze what drives the clusters in the 2D UMAP projection."*
* **What AI provided:** Split the code into `app.py`, `layout.py`, `callbacks.py`, `data_layer.py`, and `figures.py`. Suggested using HDBSCAN clustering paired with Cramér's V to measure correlation with cluster IDs.
* **Our decision & action:** Accepted and implemented. We also fixed a PyArrow sandbox thread-lock bug (`use_threads=False`), reducing data load time from hang to 0.11s. Cramér's V mathematically confirmed that UMAP clusters split by medication changes ($V=0.803$), not disease biology.

---

### Phase C: Clinical Auditing, Calibration & Subgroups (Session 4)

#### Prompt P4: Interactive Exploratory Guide Notebook
* **Date & Session:** 2026-11-03 (Session 4)
* **Goal:** Create an educational reference notebook reviewing the full cohort, features, and model benchmarks.
* **What we asked:** *"Create a comprehensive interactive guide notebook that explains the dataset features, distributions, extraction funnel, and single-encounter inspections with pre-rendered Plotly figures."*
* **What AI provided:** Authored `dataset_and_models_guide.ipynb` (28 cells) with cohort extraction funnels, feature boxplots, and encounter inspection cards.
* **Our decision & action:** Accepted and committed to `phase1/pair_A_ai_aided/`. It served as the central reference for understanding why both models achieve ~63% accuracy by relying on the 66.4% majority class.

#### Prompt P5: Clinical Grounding & Literature Cross-Check
* **Date & Session:** 2026-11-03 (Session 4)
* **Goal:** Understand clinical encounter features and check why models perform poorly on rare complications.
* **What we asked:** *"As non-clinical domain researchers, we need to understand the medical context of these hospital billing records: how do clinical encounters map to ICD-9 complication pathways, and how do our findings align with the original Strack et al. (2014) paper?"*
* **What AI provided:** Explained how billing codes represent administrative procedures and cross-referenced Strack et al. (2014), confirming that hospital operational flags dominate the data.
* **Our decision & action:** Accepted as our core analytical narrative: retrieval models inherit hospital administrative patterns rather than true disease progression.

#### Prompt P6: Fixing Interactive Selection Bugs
* **Date & Session:** 2026-11-03 (Session 4)
* **Goal:** Debug callback state issues where lasso selections reset unexpectedly.
* **What we asked:** *"Investigate and fix interactive selection bugs in the Dash application: lasso selections reset unexpectedly, cross-filtering between View 1 and View 3 loses state, and clicks on confusion matrix cells reset active filters."*
* **What AI provided:** Diagnosed a circular callback race condition; added `dash.no_update` guards and consistent `uirevision` keys.
* **Our decision & action:** Accepted and verified by writing automated headless unit tests (`test_dashboard.py`).

#### Prompt P7: Expanding into a Multi-Tab Auditing Studio
* **Date & Session:** 2026-11-03 (Session 4)
* **Goal:** Add dedicated views for calibration, model disagreement, and demographic fairness.
* **What we asked:** *"Review the course requirements from the lab materials and help us design additional views to expose model behavior that aggregate metrics hide."*
* **What AI provided:** Designed 3 new views: Confidence-Accuracy Calibration Curve, Model Disagreement Battleground ($n=5,918$ conflicts), and Dynamic Cohort Profiler.
* **Our decision & action:** Accepted and implemented across 3 new studio tabs.

#### Prompt P8: Statistical Self-Audit (Calibration & Demographic Lift)
* **Date & Session:** 2026-11-03 (Session 4)
* **Goal:** Review all mathematical calculations and check for flawed assumptions in the new views.
* **What we asked:** *"Audit every statistical calculation in our new dashboard views: check for metric validity, mathematical identities, and confounding variables."*
* **What AI provided & audit found:** Discovered that the initial calibration curve was circular (plotted against ground-truth agreement), and the racial accuracy disparity was confounded by majority-class prevalence.
* **Our decision & action:** Modified calculations: replaced agreement with test-time `vote_share` (uncovering the $\ge 8/10$ consensus cliff), and evaluated demographic fairness via Accuracy Lift over subgroup base rates.

---

### Phase D: Branch Reconciliation, Testing & Presentation (Session 5)

#### Prompt P9: Comparative Branch Review (Ahmad vs. Julian)
* **Date & Session:** 2026-11-08 (Session 5)
* **Goal:** Resolve code divergence between teammate Ahmad's simplified scroll and Julian's multi-tab studio.
* **What we asked:** *"Review Ahmad's simplified single-page branch vs. Julian's multi-tab branch and recommend how to reconcile both approaches for the midpoint."*
* **What AI provided:** Analyzed tradeoffs: Ahmad's branch was clean for a short video but deleted projections; Julian's branch had full depth but higher complexity.
* **Our decision & action:** Formulated a reconciliation plan using the "Martini Glass" pattern: Curated Storyline for the video + Exploratory Studio for grading depth.

#### Prompt P10: Dual-Mode Martini Glass Dashboard & Unit Test Suite
* **Date & Session:** 2026-11-08 (Session 5)
* **Goal:** Implement the unified layout and verify stability with automated tests.
* **What we asked:** *"Implement the Dual-Mode Martini Glass architecture on branch phase1/pair-a-ai-aided and build an automated test suite verifying data contracts and callback execution."*
* **What AI provided:** Implemented Tab 1 (6-Chapter Storyline) and Tabs 2–4 (Linked Studio, Disparity Audit, Profiler); built `test_dashboard.py` with 14 unit tests.
* **Our decision & action:** Accepted and committed. All 14 tests pass.

#### Prompt P11: Rejection of Automated Slide Decks
* **Date & Session:** 2026-11-09 (Session 5)
* **Goal:** Determine the best demonstration format for the 5-minute video.
* **What we asked:** *"Evaluate presentation options for the 5-minute midpoint video: review draft slide outlines, compare against the live interactive Dash dashboard, and decide on the most effective demonstration format."*
* **What AI provided:** Generated a 10-slide LaTeX Beamer presentation and markdown speaking script.
* **Our decision & action:** Rejected the AI slides completely. Presented authentic live screen captures of the interactive Dash studio instead.
