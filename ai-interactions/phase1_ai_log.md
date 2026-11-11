# AI Interaction Log — Phase I (Pair A, AI-Aided)

**AI-Aided Pair (Pair A):** Julian Bakan (`k11913166`), Ahmad Hasan (`k51821183`)  
**Human-Only Pair (Pair B):** Alfiia Ziganshina (`k12456306`), Aqeel Jajja (`k12554936`)*  
**Course:** AI and Visualization (WS26), JKU Linz (Visual Data Science Lab)  
**Team:** SHAP Shifters  

> [!NOTE]
> **\*Note on Team Participation:** Throughout Phase I, Pair B's human-only pipeline was carried out entirely by Alfiia Ziganshina (`k12456306`). Aqeel Jajja (`k12554936`) remained inactive and contributed no code or analysis to the project (0 commits across the repository).

---

## 1. Master Interaction Log (Chronological History)

This log documents our complete interaction history with Generative AI (Antigravity AI / Claude 3.5 Sonnet & Gemini 1.5 Pro) throughout Phase I. It records the key prompts, what AI proposed, what we decided, and how it shaped our analysis.

| # | Session & Date | AI Role | Prompt / User Goal | AI Proposal | Decision & Action | Outcome & Practical Impact |
|---|---|---|---|---|---|---|
| **P1** | **S1: 2026-10-28** | Steer Investigation | Formulate a clinical patient-matching retrieval task comparing two distinct encoders. | Proposed $k=10$ KNN retrieval on UCI Diabetes comparing 127-D tabular features vs. 768-D Bio_ClinicalBERT text embeddings across 4 complication pathways. | **Accepted with refinement** | Defined longitudinal target strictly as the *next admission's* diagnosis ($V_{t+1}$), excluding single-visit patients. |
| **P2** | **S2: 2026-10-28** | Data Transformation | Build a reproducible shared baseline pipeline and decouple data artifacts for Pair B. | Generated extraction script creating 4 static Parquet/JSON tables, but added synthetic clinical notes. | **Rejected fake notes; accepted schema** | **Purged label leakage**: deleted synthetic notes that gave away the diagnosis; locked inputs to 15 real UCI fields. |
| **P3** | **S3: 2026-11-02** | UI Design & Implementation | Modularize Dash app into separate components and explain 2D UMAP cluster structure. | Modularized into `layout.py`, `callbacks.py`, `figures.py`, `data_layer.py`; suggested HDBSCAN and Cramér's V for cluster analysis. | **Accepted and implemented** | Fixed a PyArrow sandbox thread-lock bug (`use_threads=False`); proved UMAP clusters split by medication flags, not disease. |
| **P4** | **S4: 2026-11-03** | Data Transformation | Create a detailed interactive guide reviewing all features, distributions, and model metrics. | Built [`dataset_and_models_guide.ipynb`](../phase1/pair_A_ai_aided/dataset_and_models_guide.ipynb) (28 cells) with cohort funnels, boxplots, and encounter cards. | **Accepted** | Provided Pair A with an educational reference explaining why 63% accuracy is driven by the 66.4% majority class. |
| **P5** | **S4: 2026-11-03** | Interpretation | Understand the clinical meaning of encounters and verify findings against dataset literature. | Cross-referenced findings with the original dataset paper (Strack et al., 2014), explaining that administrative billing flags dominate records. | **Accepted** | Grounded our findings: both models struggle on rare complications because billing data reflects hospital procedures, not biology. |
| **P6** | **S4: 2026-11-03** | UI Debugging | Fix Dash interactive selection bugs: lasso resetting state and cross-filter conflicts. | Diagnosed circular callback race condition; added `dash.no_update` guards and consistent `uirevision` keys. | **Accepted & verified with tests** | Fixed selection persistence across views and added automated headless unit tests (`test_dashboard.py`). |
| **P7** | **S4: 2026-11-03** | Investigation & Design | Expand dashboard into a comprehensive clinical auditing studio with subgroup and confidence views. | Designed 3 new views: Confidence-Accuracy Calibration, Model Disagreement Battleground, and Dynamic Cohort Profiler. | **Accepted and expanded** | Expanded dashboard from a basic 4-view prototype into a deep multi-tab auditing studio. |
| **P8** | **S4: 2026-11-03** | Critique Findings | Perform statistical self-audit of new views: check for metric validity and confounding. | Identified two critical flaws: circular calibration against true labels, and base-rate confounding in raw racial accuracy. | **Modified calculations** | Switched calibration to test-time `vote_share` (found $\ge 8/10$ cliff); switched fairness to base-rate Accuracy Lift. |
| **P9** | **S5: 2026-11-08** | Critique Design | Compare two diverging team branches (Ahmad's simple scroll vs. Julian's multi-tab studio). | Evaluated tradeoffs: Ahmad's branch was clean for a short video but lacked depth; Julian's branch had full depth but higher complexity. | **Refined architecture** | Planned synthesis using the "Martini Glass" pattern: Curated Storyline for the video + Exploratory Studio for grading depth. |
| **P10** | **S5: 2026-11-08** | UI Implementation | Implement unified Dual-Mode dashboard and automated test suite on `phase1/pair-a-ai-aided`. | Built Tab 1 (6-Chapter Storyline) and Tabs 2–4 (Linked Studio, Disparity Audit, Profiler); built 14-test test suite. | **Accepted and committed** | Unified both teammates' work cleanly; all 14 automated unit tests pass. |
| **P11** | **S5: 2026-11-09** | Presentation Strategy | Review presentation options for 5-minute video: evaluate draft slide deck vs. live dashboard demos. | AI drafted a 10-slide LaTeX Beamer presentation and markdown speaking script. | **Rejected AI slides completely** | Threw out generic AI slides; decided to record authentic live screen captures of our interactive Dash application. |

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
