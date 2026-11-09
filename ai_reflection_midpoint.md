# AI4Vis Reflection Report — Midpoint (Phase I)

**Team Name:** SHAP Shifters  
**Course:** AI and Visualization (WS26), JKU Linz  
**Pair A (AI-Aided):** Julian Bakan (`k11913166`), Ahmad Hasan (`k51821183`)  
**Pair B (Human-Only):** Alfiia Ziganshina (`k12456306`), Aqeel Jajja (`k12554936`)  

---

## 1. Executive Summary & Study Protocol

Under the JKU AI & Visualization study protocol, our 4-student team evaluated black-box behavior for clinical patient retrieval across $30,248$ repeat-admission encounters ($16,773$ unique patients) from the UCI Diabetes 130-US Hospitals dataset (UCI ID 296). We compared two distinct encoders on a $k=10$ cosine nearest-neighbor retrieval task (excluding same-patient visits) predicting 4 future complication pathways (*Uncomplicated / Other*: 66.36%, *Cardiovascular*: 29.27%, *Kidney / Renal*: 2.50%, *Eye / Neurological*: 1.86%):
- **Model A (127-D Tabular Cosine KNN)**: 5 $z$-standardized numeric clinical counts + 10 unscaled one-hot encoded categorical variables (63.01% overall accuracy).
- **Model B (768-D Bio_ClinicalBERT Cosine KNN)**: Attention-masked mean-pooled embeddings from `emilyalsentzer/Bio_ClinicalBERT` over authentic serialized encounter records (62.96% overall accuracy).

In Phase I, **Pair A worked with Generative AI assistance**, while **Pair B worked strictly human-only** (restricted to IDE auto-completion and static API documentation). The pairs swap roles in Phase II.

This report reflects on our Phase I experience using Generative AI across the **5 GenAI Roles** established in Lab 03, evaluating empirical benefits, critical failure modes, human-in-the-loop verification overhead, and lessons learned.

---

## 2. Structured Analysis Across the 5 GenAI Roles (Lab 03)

### Role 1: GenAI as a Tool to Transform Data
* **Application**: Generating data schemas, Parquet conversion scripts, and text serialization pipelines.
* **Benefit**: Rapidly transformed the raw UCI dataset ($101,766$ raw rows) into an immutable static data contract (`cohort_encounters.parquet`, `retrieval_results.parquet`, `counterfactual_flips.parquet`, `baseline_metrics.json`). This decoupled the deep learning runtime from the analytical layer, enabling Pair B to explore data with standard Pandas without GPU or CUDA dependencies.
* **Failure Pattern & Human Intervention**: The initial AI-generated baseline script created synthetic "presenting clinical notes" by hallucinating symptom text (e.g., *"elevated serum creatinine, proteinuria"*) derived directly from the encounter's diagnosis codes, and included a silent TF-IDF fallback encoder. Human code review identified this as severe **label leakage** that artificially inflated Model B's accuracy. We completely excised the synthetic text generator and fallback encoder, restricting Model B strictly to 15 authentic UCI fields and anchoring the ground-truth target strictly to the *subsequent admission's* primary ICD-9 code (`diag_1` at $V_{t+1}$).

### Role 2: GenAI as a Collaborator to Design & Implement Visualizations
* **Application**: Rapid engineering of an interactive, state-managed Plotly/Dash analytical web application (`phase1/pair_A_ai_aided/dashboard/`).
* **Benefit**: GenAI enabled Pair A to construct a multi-module coordinated visual analytics system featuring:
  1. *Curated Storyline (6-Chapter Walkthrough)*: Author-driven narrative mode designed for rapid comprehension and our $\le 5$-minute midpoint presentation.
  2. *Primary Retrieval Studio (Views 1–4)*:
     - **View 1**: Multi-channel scatter explorer with uniform categorical jitter and 2D density contour overlays to unpack discrete overplotting.
     - **View 2**: Hierarchically clustered confusion matrices with optimal leaf ordering (Alsallakh et al. 2017).
     - **View 3**: 2D UMAP projection with automated HDBSCAN clustering, Cramér's V driver rankings, and longitudinal repeat-patient trajectories ($V_1 \to V_2 \to V_3$, Hinterreiter et al. 2021).
     - **View 4**: Counterfactual perturbation flip bars and a 10-nearest-neighbor encounter dossier with cosine similarity ranking ($1 - \text{dist}$) and overlap badges.
  3. *Disparity & Calibration Audit*: Dual-axis calibration curves with 95% Wilson score confidence intervals, head-to-head disagreement battlegrounds ($n=5,918$ conflicts), and subgroup accuracy lift controlling for base-rate confounding.
  4. *Dynamic Cohort Profiler*: Clinical complexity profile indexed to 100% of the hospital baseline mean so disparate clinical scales (`number_emergency = 0.33` vs. `num_lab_procedures = 42.54`) are directly comparable.
* **Verification & Testing**: Pair A built a 14-test headless unit test suite (`test_dashboard.py`) using Flask's `test_client()`, verifying data contracts, callback execution, selection persistence, layout ordering, and zero duplicate component IDs.

### Role 3: GenAI as a Collaborator to Critique Design & Findings
* **Application**: Auditing visual channel choices, statistical validity, and confounding variables.
* **Key Interventions**:
  - *Resolving Circular Calibration Tautology*: An early prototype binned accuracy by `agreement` (fraction of neighbors matching the *true future label* $y$). Because $k=10$ KNN predicts via majority vote, $\ge 6/10$ agreement with $y$ guarantees $\hat{y}=y$ (100% accuracy by arithmetic identity) and is unknown at test time. We rebuilt the calibration curve to bin by test-time `vote_share` (fraction of neighbors voting for the *predicted class* $\hat{y}$), uncovering that in **69.4% of stays** (`vote_share < 80%`), accuracy drops to **59.6%–59.7%** (below both the 66.36% majority baseline and 62.13% random baseline), and only beats the majority baseline when **$\ge 8/10$ neighbors agree** (~30.6% of stays, reaching **68.2%–77.6%** accuracy).
  - *Deconfounding Subgroup Fairness*: A naive demographic audit suggested African American patients suffered a 3.7 pp accuracy penalty (60.0% vs. 63.7% Caucasian). Auditing subgroup base rates revealed that African American encounters have a lower majority-class (`Uncomplicated / Other`) base rate (63.1% vs. 66.9%). When evaluated as **Accuracy Lift over each subgroup's own majority baseline**, performance is virtually identical (**$-3.1\text{ pp}$ vs. $-3.3\text{ pp}$**).

### Role 4: GenAI as a Collaborator to Steer the Investigation
* **Application**: Formulating counterfactual hypotheses and drilling into representation disparity.
* **Discovery (Mirror-Image Representation Artifacts)**: Although Model A (63.01%) and Model B (62.96%) achieve nearly identical headline accuracy, they **share only 22.4% of their retrieved neighbors** (2.24 / 10) and disagree on **5,918 encounters (19.6%)**. Single-attribute counterfactual perturbations across all encounters revealed opposite vulnerabilities:
  - **Sex Swap (`Male` $\leftrightarrow$ `Female`)**: Model A replaces **8.86 / 10 neighbors** (producing 94.7% same-sex neighbor homophily because flipping two unscaled binary one-hot columns adds $\Delta\|x\|^2 = 2.0$ before cosine normalization), whereas Model B replaces only **4.15 / 10 neighbors** (60.8% same-sex homophily).
  - **HbA1c Elevation (`None/Norm` $\to$ `>8`)**: Model B replaces **8.22 / 10 neighbors** (because changing `none` $\to$ `>`, `8` alters multiple WordPiece tokens in mean-pooled 768-D space), whereas Model A replaces only **4.36 / 10 neighbors**.
  - **Age Shift (`+20 years`)**: Both models show moderate sensitivity (Model A: **5.42 / 10**, Model B: **5.11 / 10**).

### Role 5: GenAI as a Tool to Interpret Observations
* **Application**: Grounding 2D projection clusters without subjective visual cherry-picking.
* **Methodology**: Rather than visually guessing why the 127-D UMAP projection separates into distinct islands, we computed **Cramér's V** (for categorical features) and **correlation ratio $\eta$** (for numeric features) across all 15 input features against HDBSCAN cluster labels.
* **Finding**: Proved mathematically that the UMAP macro-clusters are **not** organized by future complication pathway, but by current treatment workflow flags: **Medication Change (`change`, Cramér's $V = 0.803$)**, **Diabetes Medication (`diabetesMed`, $V = 0.796$)**, **Max Glucose ($V = 0.518$)**, **Admission Source ($V = 0.488$)**, and **Insulin Regimen ($V = 0.486$)**.

---

## 3. The Design Dilemma: "Exploratory Cockpit" vs. "Curated Story"

Following our Feedback 1 session (Nov 5), where the instructor emphasized curating visualizations and telling a coherent story of input-output relationships rather than showing raw performance metrics or overwhelming the viewer, a key architectural challenge arose within Pair A:
* **The Exploratory Cockpit (`phase1/pair-a-ai-aided`)**: Provided deep linked brushing, UMAP driver rankings, and cohort profiling, but required jumping across multiple tabs during a strict $\le 5$-minute video.
* **The Linear Scroll Attempt (`Ahmad/phase1/ai_aided/dashboard`)**: Attempted to simplify the app into 6 vertical sections for presentation clarity, but inadvertently removed the UMAP projection, HDBSCAN driver rankings, and Cohort Profiler, while breaking 9 automated tests.
* **The Dual-Mode Synthesis (Martini Glass Architecture)**: We reconciled both branches by combining an author-driven **Tab 1: 🎬 Curated Storyline (6-Chapter Video Walkthrough)** with the reader-driven **Tabs 2–4: Interactive Retrieval Studio, Disparity Audit, and Dynamic Cohort Profiler**. All 6 linear chapters and all 4 coordinated studio views are now unified and verified by **14 passing automated unit tests**.

---

## 4. Empirical Comparison: AI-Aided (Pair A) vs. Human-Only (Pair B)

| Dimension | Pair A (AI-Aided) | Pair B (Human-Only) |
| :--- | :--- | :--- |
| **Development Velocity** | High: built a 4-module interactive Dash web application and 26-cell audited guide notebook. | Deliberate: wrote clean, focused Pandas/Matplotlib scripts directly from the static Parquet contracts. |
| **System & Interaction Depth** | Coordinated multi-view brushing, automated HDBSCAN + Cramér's V ranking, 95% Wilson CIs, 100%-indexed Cohort Profiler, 14 unit tests. | Targeted static visualizations examining class imbalance, confusion blocks, and demographic flip distributions. |
| **Verification Overhead** | **Very High**: Required catching LLM label leakage, fixing circular `agreement` vs. `vote_share` binning, and reconciling UI layout branches. | **Low**: Every line of analysis and aggregation was written and verified manually from scratch. |
| **Key Complementary Insights** | Uncovered the $\ge 8/10$ calibration cliff, base-rate confounding in racial accuracy ($-3.1\text{ pp}$ vs. $-3.3\text{ pp}$ lift), and the mirror-image Sex Swap (8.86 vs. 4.15) vs. HbA1c (4.36 vs. 8.22) representation artifacts. | Confirmed the 0.0% minority-pathway recall collapse on Kidney ($n=757$) and Eye ($n=563$) stays and age-bracket complementarity ($<50$ yrs vs. $80+$ yrs). |

---

## 5. Emerging Best Practices for AI in Visual Analytics

1. **Audit Data Lineage & Target Definitions First**: Never trust AI-generated preprocessing scripts or summary metrics without verifying that input features cannot leak future labels and that confidence metrics (`vote_share`) are non-circular at test time.
2. **Pair Projections with Quantitative Driver Rankings**: 2D embeddings (UMAP/t-SNE) invite subjective storytelling. Computing Cramér's V and correlation ratios against cluster labels turns visual clusters into rigorous, falsifiable explanations.
3. **Control for Base-Rate Confounding in Imbalanced Clinical Cohorts**: When a majority class accounts for 66.4% of stays, raw subgroup accuracy reflects subgroup base rates rather than model skill; always plot accuracy lift over each subgroup's own baseline with Wilson confidence intervals.
4. **Use the Martini Glass Pattern (Story + Studio)**: Pair a curated, linear narrative tab (for clear communication under strict time limits) with a linked multi-view studio (for open-ended exploratory drilling).
