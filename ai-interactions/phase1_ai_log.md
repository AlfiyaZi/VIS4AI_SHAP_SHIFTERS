# AI Interaction Log — Phase I (Pair A, AI-Aided)

**AI-Aided Pair (Pair A):** Julian Bakan (`k11913166`), Ahmad Hasan (`k51821183`)  
**Human-Only Pair (Pair B):** Alfiia Ziganshina (`k12456306`), Aqeel Jajja (`k12554936`)*  
**Course:** AI and Visualization (WS26), JKU Linz (Visual Data Science Lab)  
**Team:** SHAP Shifters  

> [!NOTE]
> **\*Note on Team Participation:** Throughout Phase I, Pair B's human-only pipeline was executed solely by Alfiia Ziganshina (`k12456306`). Aqeel Jajja (`k12554936`) remained inactive and contributed no code or analysis (0 commits across the repository).  

---

## 1. Overview & High-Impact Interactions

In Phase I, Pair A used Generative AI (Antigravity AI / Claude 3.5 Sonnet & Gemini 1.5 Pro) as both a code-generation **tool** and an analytical **collaborator**. Below are the key interactions that directly steered our visual analytics pipeline and provide evidence of GenAI's strengths and failure modes.

| # | Interaction Focus | Operating Mode & Role | GenAI Output | Human Decision & Action | Impact on Analysis |
|---|---|---|---|---|---|
| **1** | **Shared Baseline & Data Contract** | 🛠️ Tool (Transform Data) | Generated data preprocessing pipeline with simulated clinical notes. | **REJECTED notes; accepted schema.** Excised synthetic text; restricted inputs to 15 authentic fields. | **Purged label leakage**: eliminated artificial model accuracy inflation before splitting pairs. |
| **2** | **UMAP Latent Space Interpretation** | 🤝 Collaborator (Interpret) | Suggested non-parametric association metrics to explain 2D clusters. | **ACCEPTED.** Implemented Cramér's V and correlation ratio $\eta$ against HDBSCAN clusters. | **Proved UMAP splits by medication changes (`change` $V=0.803$)**, not future complication pathways. |
| **3** | **Confidence-Accuracy Calibration** | 🤝 Collaborator (Critique) | Plotted accuracy against neighbor `agreement` with ground-truth label $y$. | **REJECTED tautological metric; MODIFIED.** Replaced with test-time `vote_share`. | **Uncovered consensus cliff**: predictions are worse than baseline unless $\ge 8/10$ neighbors agree. |
| **4** | **Subgroup Demographic Fairness** | 🤝 Collaborator (Critique) | Flagged a 3.7 pp raw accuracy gap (60.0% African American vs. 63.7% Caucasian). | **REFINED.** Controlled for subgroup majority-class base rates via **Accuracy Lift**. | **Prevented false bias claim**: proved both groups have identical relative drop ($-3.1$ vs $-3.3\text{ pp}$). |
| **5** | **UI Architecture & Presentation** | 🤝 Collaborator (Design) | Ahmad branch simplified UI by deleting projections; AI drafted Beamer slides. | **REFINED architecture; REJECTED AI slides.** Created Dual-Mode Martini Glass app; presented live UI. | **Satisfied grading rubric**: preserved high-dimensional analysis while keeping video under 5 minutes. |

---

## 2. Key Case Studies: Strengths vs. Limitations

### Case 1: Preprocessing Hallucination & Label Leakage (Limitation)
* **Prompt:** *"Build the shared baseline preprocessing pipeline for UCI Diabetes 130-US Hospitals predicting 4 complication pathways."*
* **GenAI Proposal:** Generated a helper function `derive_presenting_notes_from_inputs()` that synthesized text descriptions of symptoms (e.g., *"elevated creatinine, proteinuria"*).
* **Flaw Identified:** The symptoms were synthesized by looking at the encounter's *diagnosis codes*, directly leaking the target outcome into Model B's text embeddings.
* **Correction:** Human audit deleted all synthetic text generators, locked Model B to 15 authentic UCI fields, and strictly anchored ground truth to the *subsequent admission's* primary ICD-9 code ($V_{t+1}$).

### Case 2: Explaining Latent Clusters with Non-Parametric Statistics (Strength)
* **Prompt:** *"Why does the 127-D UMAP projection separate into discrete islands? How can we explain these clusters quantitatively without guessing?"*
* **GenAI Proposal:** Implemented automated HDBSCAN clustering paired with Cramér's V (for categorical variables) and correlation ratio $\eta$ (for numeric variables) across all 15 features.
* **Finding:** Mathematically proved that cluster separation is driven by administrative treatment flags—**Medication Change (`change`, $V=0.803$)** and **Diabetes Med Prescribed (`diabetesMed`, $V=0.796$)**—rather than patient biology.

### Case 3: Circular Calibration Tautology vs. Test-Time Consensus (Limitation)
* **Prompt:** *"Implement a calibration curve to evaluate whether retrieval confidence correlates with accuracy."*
* **GenAI Proposal:** Binned encounters by `agreement` (the fraction of retrieved neighbors sharing the *true* label $y$).
* **Flaw Identified:** In $k=10$ KNN, $\ge 6/10$ neighbors matching $y$ guarantees the majority vote prediction is $y$ (100% accuracy by definition), which is completely unknowable at test time.
* **Correction:** Replaced `agreement` with test-time **`vote_share`** (fraction of neighbors agreeing on the *predicted* class $\hat{y}$). This revealed a vital clinical finding: below 80% consensus (69.4% of stays), KNN accuracy drops to 59.6% (worse than the 66.4% majority baseline).

### Case 4: Controlling for Subgroup Base-Rate Confounding (Limitation)
* **Prompt:** *"Audit model performance across demographic subgroups (race and sex) to identify bias."*
* **GenAI Proposal:** Displayed raw accuracy bar charts, highlighting a 3.7 pp deficit for African American patients and suggesting an algorithmic fairness defect.
* **Flaw Identified:** The majority class (`Uncomplicated / Other`) has a lower prevalence among African American encounters (63.1%) than Caucasian encounters (66.9%).
* **Correction:** Recomputed metric as **Accuracy Lift over subgroup base rate**. Both groups show virtually identical relative performance ($-3.1\text{ pp}$ vs. $-3.3\text{ pp}$), demonstrating that the raw gap was an artifact of cohort composition.

### Case 5: The "Martini Glass" Synthesis & Rejection of AI Slides (Human Oversight)
* **Prompt:** *"Review Ahmad's simplified single-page branch vs. Julian's multi-tab branch and recommend how to structure the 5-minute presentation."*
* **GenAI Proposal:** GenAI drafted automated 10-slide LaTeX Beamer slides and a markdown script.
* **Correction:** The AI slide deck was generic and text-heavy, violating the course principle to show curated visual analytics. We rejected the slide deck completely, opting to record authentic live captures of our Dash studio, and reconciled both branches using the **Martini Glass pattern**: Tab 1 (Curated 6-Chapter Storyline for the 5-minute video) + Tabs 2–4 (Interactive Exploratory Studio verified by 14 automated unit tests).

---

## 3. Evidence of GenAI Strengths and Limitations

### Core Strengths
1. **Implementation Velocity**: Rapidly scaffolded a 4-module interactive Dash web application, complete callback graphs, and 14 automated unit tests (`test_dashboard.py`).
2. **Quantitative Exploration**: Effortlessly computed non-parametric driver rankings (Cramér's V) and Wilson score 95% confidence intervals across 30,248 stays.
3. **Hypothesis Formulation**: Automated counterfactual perturbations (Sex swap, Age shift, HbA1c elevation), uncovering the models' mirror-image representation vulnerabilities.

### Critical Limitations
1. **Plausible Data Hallucinations**: Propensity to invent synthetic text or fallbacks that introduce severe label leakage unless audited.
2. **Circular Evaluation Logic**: Tendency to construct mathematically trivial or tautological evaluation metrics (e.g., calibration against ground truth).
3. **Superficial Bias Interpretations**: Prone to confusing cohort base-rate disparities with algorithmic model bias without controlling for underlying prevalence.

---

## 4. Complete Prompt & Decision History

Below is the complete log of all prompts and steering decisions across Phase I, documenting the development trajectory from initial setup to the midpoint submission:

| ID | Development Phase | AI Role | Prompt Objective | AI Output | Decision & Action | Outcome & Repo Artifact |
|:---:|:---|:---|:---|:---|:---|:---|
| **P1** | **Task Formulation** | Steer Investigation | Formulate a clinical patient-matching retrieval task comparing two distinct encoders. | Proposed $k=10$ KNN retrieval on UCI Diabetes comparing 127-D tabular features vs. 768-D Bio_ClinicalBERT text embeddings across 4 complication pathways. | **Accepted with refinement** | Defined longitudinal target strictly as next admission ($V_{t+1}$); recorded in `README.md`. |
| **P2** | **Shared Baseline Data** | Data Transformation | Build reproducible baseline pipeline and decouple static data tables for Pair B. | Generated extraction script creating 4 static Parquet/JSON tables, but added synthetic clinical notes. | **Rejected fake notes; accepted schema** | Purged label leakage; locked inputs to 15 real UCI fields in `shared/01_prepare_baseline_data.py`. |
| **P3** | **Dashboard Architecture** | UI Design & Implementation | Modularize Dash app into separate components and explain 2D UMAP cluster structure. | Modularized into `layout.py`, `callbacks.py`, `figures.py`, `data_layer.py`; suggested HDBSCAN and Cramér's V for cluster analysis. | **Accepted and implemented** | Fixed PyArrow sandbox thread lock (`use_threads=False`); proved UMAP clusters split by medication flags ($V=0.803$). |
| **P4** | **Exploratory Guide** | Data Transformation | Create an interactive exploratory guide detailing features, distributions, and model metrics. | Built [`dataset_and_models_guide.ipynb`](../phase1/pair_A_ai_aided/dataset_and_models_guide.ipynb) (28 cells) with cohort funnels, boxplots, and encounter cards. | **Accepted** | Provided Pair A with an educational reference explaining why 63% accuracy is driven by majority-class prevalence. |
| **P5** | **Clinical Interpretation** | Interpretation | Understand the clinical meaning of encounters and verify findings against published literature. | Cross-referenced findings with the original Strack et al. (2014) paper, explaining that administrative billing flags dominate records. | **Accepted** | Grounded findings: both models struggle on rare complications because billing data reflects hospital procedures, not biology. |
| **P6** | **UI Debugging** | UI Debugging | Fix Dash interactive selection bugs: lasso resetting state and cross-filter conflicts. | Diagnosed circular callback race condition; added `dash.no_update` guards and consistent `uirevision` keys. | **Accepted & verified with tests** | Ensured selection persistence across views; added automated headless unit tests (`test_dashboard.py`). |
| **P7** | **Auditing Studio Views** | Investigation & Design | Expand dashboard into a comprehensive clinical auditing studio with subgroup and confidence views. | Designed 3 new views: Confidence-Accuracy Calibration, Model Disagreement Battleground, and Dynamic Cohort Profiler. | **Accepted and expanded** | Expanded dashboard from a basic 4-view prototype into a multi-tab clinical auditing studio. |
| **P8** | **Statistical Self-Audit** | Critique Findings | Perform statistical self-audit on new views: check for metric validity and confounding. | Identified two critical flaws: circular calibration against true labels, and base-rate confounding in raw racial accuracy. | **Modified calculations** | Switched calibration to test-time `vote_share` (found $\ge 8/10$ cliff); switched fairness to base-rate Accuracy Lift. |
| **P9** | **Branch Review** | Critique Design | Compare two diverging team branches (Ahmad's simple scroll vs. Julian's multi-tab studio). | Evaluated tradeoffs: Ahmad's branch was clean for a short video but lacked depth; Julian's branch had full depth but higher complexity. | **Refined architecture** | Formulated the "Martini Glass" synthesis plan: Curated Storyline for the video + Exploratory Studio for grading depth. |
| **P10** | **Martini Glass Synthesis** | UI Implementation | Implement unified Dual-Mode dashboard and automated test suite on `phase1/pair-a-ai-aided`. | Built Tab 1 (6-Chapter Storyline) and Tabs 2–4 (Linked Studio, Disparity Audit, Profiler); built 14-test test suite. | **Accepted and committed** | Unified both teammates' work cleanly; all 14 automated unit tests pass in `test_dashboard.py`. |
| **P11** | **Presentation Strategy** | Presentation Strategy | Review presentation options for 5-minute video: evaluate draft slide deck vs. live dashboard demos. | AI drafted a 10-slide LaTeX Beamer presentation and markdown speaking script. | **Rejected AI slides completely** | Threw out generic AI slides; decided to record authentic live screen captures of our interactive Dash application. |

---

## 5. Pre-Split Collaborative Record (Pair B Transparency)

For complete transparency, prior to the official Phase I split, team members of the human-only pair (Alfiia Ziganshina, Aqeel Jajja) utilized AI to audit the initial problem setup:

| Phase | Member | Tool | Interaction Scope | Outcome & Impact |
|:---|:---|:---|:---|:---|
| **Pre-Split** | Alfiia | Claude Code | Review course materials, baseline pipeline structure, and initial figures | Identified label leakage risks, proposed static data contract, informed `shared/` refactor |
