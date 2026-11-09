# Phase I Midpoint Presentation Script & Delivery Guide
**Course:** AI & Visualization WS26 (JKU Linz)  
**Instructors:** Prof. Marc Streit, Andreas Hinterreiter, Amal Alnouri  
**Team:** SHAP Shifters (`VIS4AI_SHAP_SHIFTERS`)  
**Speakers:** Julian Bakan (`k11913166`), Ahmad Hasan (`k51821183`) [Pair A] & Alfiia Ziganshina (`k12456306`), Aqeel Jajja (`k12554936`) [Pair B]  
**Target Video Duration:** 4 minutes 50 seconds (Hard Ceiling: ≤ 5 minutes)  
**LaTeX Beamer Source:** [`presentation/midpoint/Latex/phase1_presentation.tex`](file:///C:/Users/juba9/PycharmProjects/VIS4AI_SHAP_SHIFTERS/presentation/midpoint/Latex/phase1_presentation.tex)  
**Compiled PDF:** [`phase1_presentation.pdf`](file:///C:/Users/juba9/PycharmProjects/VIS4AI_SHAP_SHIFTERS/presentation/midpoint/Latex/phase1_presentation.pdf)  
**Interactive Dashboard:** `http://127.0.0.1:8051`

---

## Executive Summary of the Presentation Structure

This presentation embeds the **authentic, high-resolution figures directly exported from our dashboard and experimental pipelines**, addressing every piece of course feedback:
1. **Authentic Dashboard Figures:** Every slide features the actual Plotly/Matplotlib visualization produced in the project (`figures/fig1_confusion_pair.png`, `fig2_calibration.png`, `fig3_umap.png`, `fig4_flips.png`, `fig5_conflicts.png`, `fig_pairB_demographic_lift.png`, and `fig7_scatter.png`).
2. **Deep Input-Output Relationships:** Replaces raw error reporting with high-dimensional manifold analysis (UMAP + Cramér's $V$) and counterfactual single-attribute perturbations.
3. **Actionable Decision Boundary:** Formulates the test-time `vote_share \ge 80%` rule, separating trustworthy clinical signal from retrieval noise.
4. **Coherent AI vs. Human Story:** Compares Pair A (AI-Aided) and Pair B (Human-Only) as a complementary synthesis: Pair A provides the global system-level observatory, while Pair B explains the local geometric crowding mechanism.
5. **No Text Walls on Slides:** Every slide adheres strictly to the rule of **1 Visual Graphic + 1 Bold Headline Takeaway**.

---

## Detailed Timing & Run-of-Show (300 Seconds Total)

| Slide | Section / Topic | Visual Displayed | Time Window | Duration |
| :---: | :--- | :--- | :---: | :---: |
| **0** | **Title Slide** | JKU Beamer Title Frame with Team SHAP Shifters & all 4 members | `0:00 – 0:10` | 10s |
| **1** | **Baseline Retrieval Paradox** | **`fig1_confusion_pair.png`**: Side-by-side Alsallakh-reordered confusion heatmaps | `0:10 – 0:50` | 40s |
| **2** | **RQ1: Decision Calibration ($\ge 80\%$)** | **`fig2_calibration.png`**: Vote share calibration curve with green Signal Zone ($\ge 80\%$) | `0:50 – 1:30` | 40s |
| **3** | **RQ2: 127-D Manifold Drivers** | **`fig3_umap.png`**: UMAP density clusters & longitudinal trajectories + Cramér's $V$ table | `1:30 – 2:10` | 40s |
| **4** | **RQ4: Perturbation Shortcuts** | **`fig4_flips.png`**: Counterfactual neighbor replacement bar chart (Sex, HbA1c, Age) | `2:10 – 2:50` | 40s |
| **5** | **Neighborhood Geometry (Pair B)** | **`fig_pairB_demographic_lift.png`**: Pair B plot of demographic vs. administrative lift | `2:50 – 3:35` | 45s |
| **6** | **RQ3: Specialization & Fairness** | **`fig5_conflicts.png`**: Age-stratified conflict win rates + Subgroup fairness audit | `3:35 – 4:15` | 40s |
| **7** | **Coordinated Studio & Profiler** | **`fig7_scatter.png`**: View 1 multi-channel scatter explorer + Dynamic Profiler | `4:15 – 4:45` | 30s |
| **8** | **Synthesis & Phase II Roadmap** | AI-Aided vs. Human-Only synthesis matrix + 4 actionable Phase II milestones | `4:45 – 5:00` | 15s |

---

## Slide-by-Slide Script & Visual Staging

### Slide 0: Title Slide (`0:00 – 0:10`, 10 seconds)
- **Visual:** Official JKU Beamer Title frame with Team SHAP Shifters and both pairs clearly listed.
- **Verbatim Voiceover:**
  > *"Hello everyone, we are Team SHAP Shifters. Today we present our Phase I comparative audit of two patient retrieval systems: Model A, trained on 127-dimensional tabular features, and Model B, using Bio_ClinicalBERT text embeddings."*

---

### Slide 1: Baseline Retrieval Paradox & Error Structure (`0:10 – 0:50`, 40 seconds)
- **Slide Reference:** Slide 1 (`phase1_presentation.tex`) / Dashboard Tab 1 Chapter 1 (`#chapter-performance`).
- **Visual Displayed:** Actual side-by-side Alsallakh-reordered confusion heatmaps (`figures/fig1_confusion_pair.png`).
- **Verbatim Voiceover:**
  > *"Our cohort comprises 30,248 repeat hospital encounters across 16,773 diabetic patients. Our objective is to evaluate whether $k=10$ cosine nearest-neighbor retrieval accurately predicts a patient's next-admission complication pathway.*
  > *At first glance, both models tie at 63.0% global accuracy—underperforming a trivial majority-class baseline of 66.4% by 3.4 percentage points. But identical accuracy masks two deep failures:*
  > *First, reordering the confusion matrix via Alsallakh's confusion-similarity hierarchy reveals that both models suffer a complete block collapse: they achieve zero percent recall across all 757 Kidney and 563 Eye admissions.*
  > *Second, despite sharing the same accuracy, the two models share only 22.4% of their retrieved neighbors—recommending almost entirely disjoint patient cohorts."*

---

### Slide 2: RQ1: Calibrating Trust --- The $\ge 80\%$ Signal Zone (`0:50 – 1:30`, 40 seconds)
- **Slide Reference:** Slide 2 (`phase1_presentation.tex`) / Dashboard Tab 1 Chapter 2 (`#chapter-calibration`).
- **Visual Displayed:** Actual calibration curve (`figures/fig2_calibration.png`) showing test-time vote share vs. true accuracy with Wilson 95% CIs and shaded green Signal Zone ($\ge 80\%$).
- **Verbatim Voiceover:**
  > *"To answer RQ1—when can a clinician actually trust these neighbors?—we audited an early AI-generated chart that used ground-truth labels to measure confidence. Because future labels are unavailable at test time, we reformulated calibration around neighbor consensus—the fraction of neighbors that agree with each other.*
  > *This uncovers a sharp clinical decision boundary. For 69.5% of hospital stays, fewer than 8 out of 10 neighbors agree. In this noise zone, accuracy drops to 59.6%—worse than random guessing at 62.1%.*
  > *However, in the green Signal Zone where at least 8 out of 10 neighbors agree—covering 30.5% of the cohort—accuracy leaps to 70.7%, reaching 77.6% when consensus is unanimous. Clinicians should strictly abstain whenever neighbor consensus falls below 80%."*

---

### Slide 3: RQ2: 127-D Manifold Drivers --- Workflow vs. Outcome (`1:30 – 2:10`, 40 seconds)
- **Slide Reference:** Slide 3 (`phase1_presentation.tex`) / Dashboard Tab 1 Chapter 3 (`#chapter-drivers`).
- **Visual Displayed:** Actual 127-D UMAP scatter with HDBSCAN density islands and patient trajectories (`figures/fig3_umap.png`) + Cramér's $V$ driver ranking table.
- **Verbatim Voiceover:**
  > *"Why are two-thirds of patient queries trapped in low consensus? In Chapter 3, Pair A projected the 127-dimensional tabular space with UMAP and extracted HDBSCAN density clusters. To understand what drives these clusters, we automated a Cramér's V and correlation ratio eta audit across every input attribute.*
  > *The data reveals that global manifold topology is driven almost exclusively by current treatment workflows: medication change (V = 0.803), diabetes medication status (V = 0.796), glucose flags (V = 0.518), and insulin adjustments (V = 0.486).*
  > *In contrast, the patient's future complication pathway exhibits zero structural correlation with the manifold (V = 0.042). Unsupervised cosine distance clusters patients by hospital treatment protocols, completely blind to downstream disease progression."*

---

### Slide 4: RQ4: Perturbation Sensitivity --- Mirror-Image Shortcuts (`2:10 – 2:50`, 40 seconds)
- **Slide Reference:** Slide 4 (`phase1_presentation.tex`) / Dashboard Tab 1 Chapter 4 (`#chapter-shortcuts`).
- **Visual Displayed:** Actual counterfactual neighbor turnover bar chart (`figures/fig4_flips.png`).
- **Verbatim Voiceover:**
  > *"To analyze how specific inputs govern retrieved neighbors, we evaluated counterfactual single-attribute perturbations across all 30,248 stays—holding 14 attributes constant and modifying only one.*
  > *This exposes a striking mirror-image vulnerability:*
  > *When swapping biological sex, Model A replaces 8.86 out of 10 neighbors—exhibiting 94.7% same-sex homophily. In Model A's 127-dimensional pipeline, only numeric features were z-standardized; binary one-hot flags remained unscaled. Swapping sex adds Euclidean distance norm-squared of 2.0, overriding clinical severity.*
  > *Conversely, when elevating HbA1c to greater than 8%, Model B replaces 8.22 out of 10 neighbors compared to only 4.36 for Model A. Changing the clinical string alters multiple WordPiece tokens in Bio_ClinicalBERT, completely scrambling the text representation."*

---

### Slide 5: Neighborhood Geometry: Crowding & The Voting Mask (Pair B Discovery) (`2:50 – 3:35`, 45 seconds)
- **Slide Reference:** Slide 5 (`phase1_presentation.tex`) / Pair B Notebook Findings.
- **Visual Displayed:** Actual Pair B matplotlib plot (`figures/fig_pairB_demographic_lift.png`) comparing field matching lift over random.
- **Verbatim Voiceover:**
  > *"Investigating the geometric mechanism behind Model B's fragility, our human-only teammates in Pair B uncovered a crucial structural phenomenon: crowded neighborhoods.*
  > *Model B's median cosine distance to its 10th neighbor is extraordinarily tiny—just 0.0006, compared to 0.113 in Model A. Pair B discovered that 10,684 stays—over 35% of the entire cohort—are emergency admissions with missing medical specialties. Their clinical summaries consist of identical, generic administrative text, packing them into an ultra-dense latent cluster. For these patients, a single token change pushes them into adjacent boilerplate clusters, explaining why 8.2 neighbors flip.*
  > *Furthermore, Pair B investigated why rare pathways fail: in Eye and Neurological stays, Model A actually retrieves same-pathway neighbors well above random chance. The true signal exists, but flat $k=10$ majority voting mechanically silences it in favor of the 66% majority class."*

---

### Slide 6: RQ3: Conflict Specialization & Demographic Deconfounding (`3:35 – 4:15`, 40 seconds)
- **Slide Reference:** Slide 6 (`phase1_presentation.tex`) / Dashboard Tab 1 Chapter 5 (`#chapter-disagreements`) & Tab 3.
- **Visual Displayed:** Actual age conflict win rates chart (`figures/fig5_conflicts.png`) across $n=5,918$ conflicts + demographic fairness table.
- **Verbatim Voiceover:**
  > *"When we isolate the 5,918 encounters where the two models predict different complication pathways, patient age reveals a distinct clinical specialization:*
  > *For younger diabetics under 50, Model A outperforms Model B 58.5% to 34.9% because acute numeric utilization counts govern early readmissions. But for geriatric patients aged 80 and older, Bio_ClinicalBERT wins 51.2% to 46.0%, synthesizing complex multimorbidity across text fields.*
  > *We also conducted a rigorous demographic audit: raw accuracy appears 3.7 percentage points lower for African American patients. However, our subgroup analysis proves this is entirely a majority-class base-rate artifact—the base rate of uncomplicated stays is 3.8 percentage points lower in this group. Relative to group complexity, both models perform identically."*

---

### Slide 7: Coordinated Retrieval Studio & Dynamic Profiler (`4:15 – 4:45`, 30 seconds)
- **Slide Reference:** Slide 7 (`phase1_presentation.tex`) / Dashboard Tab 2 & 4.
- **Visual Displayed:** Actual multi-channel jittered scatter explorer and density contours (`figures/fig7_scatter.png`).
- **Verbatim Voiceover:**
  > *"All of these analytical findings are interactively accessible in our Dash platform. In our 4-View Coordinated Studio, clinicians can lasso sub-cohorts in View 1 to filter the UMAP in View 3, update the confusion matrix in View 2, and inspect individual patient dossiers in View 4.*
  > *Furthermore, our Dynamic Cohort Profiler in Tab 4 evaluates the clinical complexity shift of any brushed cohort against the 100% hospital baseline, providing transparent visual justification for every retrieved neighbor."*

---

### Slide 8: Team Synthesis & Phase II Actionable Roadmap (`4:45 – 5:00`, 15 seconds)
- **Slide Reference:** Slide 8 (`phase1_presentation.tex`).
- **Visual Displayed:** Comparative synthesis matrix (Pair A vs. Pair B + 3 AI Auditing Catches) + 4 Phase II actionable milestone cards.
- **Verbatim Voiceover:**
  > *"In summary, GenAI accelerated our Dash engineering and manifold math, while human domain auditing caught three severe AI flaws: label leakage, circular calibration, and unnormalized bias.*
  > *For Phase II, we will implement calibrated confidence thresholding, harmonize one-hot scaling, disentangle ER boilerplate, and fit surrogate SHAP explainers.*
  > *Thank you very much, and we look forward to your questions!"*
