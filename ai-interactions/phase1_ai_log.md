# AI Interaction Log — Phase I (Pair A, AI-Aided)

**AI-Aided Pair (Pair A):** Julian Bakan (`k11913166`), Ahmad Hasan (`k51821183`)  
**Human-Only Pair (Pair B):** Alfiia Ziganshina (`k12456306`), Aqeel Jajja (`k12554936`)*  
**Course:** AI and Visualization (WS26), JKU Linz (Visual Data Science Lab)  
**Team:** SHAP Shifters  

> [!NOTE]
> **\*Note on Pair B Participation:** Throughout Phase I, Pair B's human-only pipeline was executed solely by Alfiia Ziganshina (`k12456306`). Aqeel Jajja (`k12554936`) remained inactive and contributed no code or analysis (0 commits across the repository).  

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
