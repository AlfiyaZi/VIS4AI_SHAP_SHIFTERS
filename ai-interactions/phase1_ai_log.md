# AI Interaction Log — Phase I (Pair A, AI-Aided)

**AI-Aided Pair (Pair A):** Julian Bakan (`k11913166`), Ahmad Hasan (`k51821183`)  
**Human-Only Pair (Pair B):** Alfiia Ziganshina (`k12456306`), Aqeel Jajja (`k12554936`)*  
**Course:** AI and Visualization (WS26), JKU Linz (Visual Data Science Lab)  
**Team:** SHAP Shifters  

> [!NOTE]
> **\*Note on Team Participation:** Throughout Phase I, Pair B's human-only work was carried out entirely by Alfiia Ziganshina (`k12456306`). Aqeel Jajja (`k12554936`) remained inactive and contributed no code or analysis (0 commits across the repository).

---

## 1. Overview of Key AI Interactions

In Phase I, Pair A used GenAI (Claude 3.5 Sonnet and Gemini via Antigravity) to write code and discuss visualization ideas. Here are the 5 most important interactions that shaped our pipeline, showing both where AI was useful and where it failed:

| # | What We Worked On | AI Role | What AI Suggested | What We Decided & Did | Impact on Analysis |
|---|---|---|---|---|---|
| **1** | **Shared Baseline Pipeline** | Data Prep | Wrote preprocessing code, but added synthetic doctor notes. | **Rejected fake notes.** Cleaned pipeline to use only 15 real dataset fields. | **Fixed label leakage**: prevented the text model from getting the diagnosis in advance. |
| **2** | **Explaining 2D UMAP Clusters** | Interpretation | Suggested using Cramér's V to measure correlation with clusters. | **Accepted.** Calculated Cramér's V and $\eta$ against HDBSCAN clusters. | **Proved clusters split by medication changes (`change` $V=0.803$)**, not disease. |
| **3** | **Confidence vs. Accuracy** | Critique | Binned accuracy by neighbor agreement with the *true* label. | **Rejected circular metric.** Switched to test-time neighbor vote share. | **Found the consensus cliff**: predictions are worse than baseline unless $\ge 8/10$ neighbors agree. |
| **4** | **Demographic Fairness** | Critique | Showed a 3.7% accuracy gap for African American patients and claimed bias. | **Corrected calculation.** Measured Accuracy Lift relative to subgroup base rates. | **Prevented false bias claim**: relative to baseline, both groups dropped equally ($-3.1$ vs $-3.3$ pp). |
| **5** | **UI Layout & Presentation** | Design | Ahmad branch removed views for video; AI generated Beamer slides. | **Kept all views; rejected AI slides.** Combined Story tab + Studio tabs; presented live UI. | **Met grading requirements**: kept full visual depth while keeping the video walkthrough under 5 minutes. |

---

## 2. Key Case Studies: Strengths vs. Failures

### Case 1: AI Invented Fake Data that Leaked the Label (Failure)
* **What we asked:** *"Build the shared preprocessing pipeline for the UCI Diabetes dataset predicting 4 future complication pathways."*
* **What AI did:** Generated a function `derive_presenting_notes_from_inputs()` that synthesized text descriptions of symptoms (e.g., *"patient presents with acute chest pain..."*).
* **What went wrong:** The AI created these symptom strings by reading the encounter's *diagnosis codes*, giving the target answer directly to Model B's text embeddings!
* **How we fixed it:** We deleted all synthetic text generation, restricted the inputs strictly to 15 real UCI fields, and made sure the target only came from the *next* admission ($V_{t+1}$).

### Case 2: Finding Out What Drives the 2D Clusters (Strength)
* **What we asked:** *"Why does the 127-D UMAP projection separate into distinct groups? How can we explain what separates them without visual guessing?"*
* **What AI did:** Suggested running HDBSCAN clustering and calculating Cramér's V (for categories) and correlation ratios (for numeric features) for all 15 features.
* **Result:** Confirmed with real numbers that the UMAP islands are separated by hospital treatment flags—**Medication Change (`change`, $V=0.803$)** and **Prescribed Meds (`diabetesMed`, $V=0.796$)**—rather than patient disease outcomes.

### Case 3: Catching a Circular Plot: Accuracy vs. Confidence (Failure)
* **What we asked:** *"Create a calibration plot to see if higher retrieval confidence leads to higher accuracy."*
* **What AI did:** Binned stays by how many neighbors matched the *true* label $y$.
* **What went wrong:** In $k=10$ KNN, having $\ge 6/10$ neighbors match the true label means the model predicts correctly by definition (100% accuracy). This was circular and impossible to know at test time.
* **How we fixed it:** We switched the x-axis to **`vote_share`** (the fraction of neighbors agreeing on the *predicted* class). This showed the real clinical insight: accuracy stays around 59.6% (worse than the 66.4% majority baseline) until neighbor agreement reaches 8/10 or higher.

### Case 4: Checking for Demographic Bias vs. Class Imbalance (Failure)
* **What we asked:** *"Audit model accuracy across demographic groups (race and sex) to see if the models are biased."*
* **What AI did:** Made a raw bar chart showing ~3.7% lower accuracy for African American patients and warned of algorithmic bias.
* **What went wrong:** In this dataset, the majority class (`Uncomplicated`) is less frequent among African American stays (63.1%) than Caucasian stays (66.9%).
* **How we fixed it:** We recalculated accuracy relative to each group's baseline (Accuracy Lift). Both groups showed almost the exact same relative drop ($-3.1$ vs $-3.3$ pp), showing that the raw difference was caused by dataset distribution, not model discrimination.

### Case 5: Structuring the App & Throwing Out AI Presentation Slides (Human Oversight)
* **What we asked:** *"Help us organize our presentation for the 5-minute video and reconcile our two branch layouts."*
* **What AI did:** Generated a 10-slide LaTeX Beamer presentation with dense bullet points.
* **What went wrong:** The slides were generic, text-heavy, and didn't show our actual dashboard.
* **How we fixed it:** We threw away the AI slides completely and decided to present live screen recordings of our Dash app. We also unified the app into a Dual-Mode layout: **Tab 1** has a 6-chapter walkthrough for the 5-minute video, while **Tabs 2–4** keep the full interactive studio with linked brushing, verified by 14 automated unit tests.

---

## 3. Summary of GenAI Strengths and Limitations

### Where GenAI Worked Well
1. **Coding Speed**: Quickly built a modular Dash web app with complex Plotly figures and a 14-test unit test suite (`test_dashboard.py`).
2. **Statistical Brainstorming**: Suggested non-parametric metrics (Cramér's V) and Wilson confidence intervals to back up visual findings.
3. **Data Perturbations**: Automated counterfactual tests (Sex swap, Age shift, HbA1c elevation) across 30,248 rows, revealing how the two models react differently to edits.

### Where GenAI Struggled
1. **Hallucinating Synthetic Data**: Tendency to invent synthetic text or shortcuts that introduce data leakage.
2. **Circular Logic in Metrics**: Generating plots that look good visually but are mathematically tautological (like calibration against the true label).
3. **Superficial Conclusions**: Jumping to conclusions (like "model bias") without checking for basic confounding factors like class imbalance.
