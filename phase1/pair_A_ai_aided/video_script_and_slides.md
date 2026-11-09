# Phase I Midpoint Progress Video: Master Script, Timing & Slide Playbook

**Team Name:** SHAP Shifters  
**Pair A (AI-Aided):** Julian Bakan (`k11913166`), Ahmad Hasan (`k51821183`)  
**Pair B (Human-Only):** Alfiia Ziganshina (`k12456306`), Aqeel Jajja (`k12554936`)  
**Target Duration:** **4 minutes 40 seconds** *(Strictly $\le 5:00$ — overtime results in automatic point deductions)*  
**Video Link File:** Add final link to `presentation/presentation_video.json` before submitting.

---

## 1. Golden Rules from Instructor Feedback & Past Grading Rubrics

| Rubric / Instructor Rule | What We Show | Why We Show It (Grading Justification) |
| :--- | :--- | :--- |
| **1. "One figure + one takeaway line per slide"** *(Failed Team lost Major points for "Too much text on slides")* | Full-screen figure or live dashboard zoom + **1 bold takeaway title** + **2–3 metric badges**. Zero bullet paragraphs on slides. | Keeps visual attention on the data; all mechanistic explanations are spoken in the voiceover. |
| **2. "Confusion matrix as top-level summary before drilling down"** *(Instructor Feedback Transcript)* | Start Pair A's walkthrough with the Alsallakh-reordered confusion matrices (Chapter 1) showing ~63.0% accuracy < 66.4% Majority Baseline and 0.0% Kidney/Eye recall, then immediately drill down into *why*. | Satisfies the instructor's exact guidance: treat performance as the entry point to deeper behavioral investigation. |
| **3. "Visualize high-dimensional data driving your patterns"** *(Failed Team lost Major points here)* | Show the 127-D UMAP manifold paired directly with the **Cramér's V / Correlation Ratio driver table** (`change` $V=0.803$, `diabetesMed` $V=0.796$) and the **100%-Normalized Cohort Profiler**. | Proves mathematically what separates clusters instead of visually guessing from 2D UMAP shapes (Lecture 3, Hinterreiter). |
| **4. "Analyze attribute influence when inputs are modified"** *(Instructor Feedback Transcript)* | Show the **Mirror-Image Counterfactual Flips** (Chapter 4): Sex Swap flips **8.86/10** in Model A vs. **4.15/10** in Model B; HbA1c Elevation flips **8.22/10** in Model B vs. **4.36/10** in Model A. | Demonstrates causal input-output sensitivity and reveals each representation's exact mathematical vulnerability. |
| **5. "Compare AI-aided vs. non-aided in a coherent story"** *(Instructor Feedback Transcript)* | Contrast Pair A's 4-module linked Dash studio & statistical self-audits (catching circular `agreement` vs. `vote_share` and racial base-rate confounding) with Pair B's deliberate manual exploration. | Directly answers the AI4Vis research question: speed & breadth vs. verification overhead. |

---

## 2. Master Timing Schedule (Total: 4m 40s / 280 seconds)

```mermaid
timeline
    title 4:40 Midpoint Video Master Timeline
    00:00 - 00:40 (40s) : Act 1: Clinical Task, Models & Leakage-Free Contract
    00:40 - 02:35 (115s) : Act 2: Pair A (AI-Aided) — 6-Chapter Story + Linked Studio Demo
    02:35 - 03:25 (50s) : Act 3: Pair B (Human-Only) — Independent Visual Exploration
    03:25 - 04:10 (45s) : Act 4: AI4Vis Story — How GenAI Changed Our Pipeline & Audits
    04:10 - 04:40 (30s) : Act 5: Phase II Plan — Opening the Black Box (SHAP vs. BERT Attention)
```

---

## 3. Scene-by-Scene Visuals, "Why We Show It", and Verbatim Script

### Act 1: Clinical Problem, Models & Leakage-Free Data Contract (`0:00 – 0:40` | 40 sec)

* **What to Show (Slide 1)**:
  - **Headline**: *"Predicting Next-Visit Diabetic Complications via $k=10$ Cosine Patient Retrieval ($N=30,248$)"*
  - **Visual**: Clean pipeline diagram:
    - **Cohort**: UCI Diabetes 130-US ($30,248$ repeat-admission stays across $16,773$ patients; target = primary ICD-9 pathway of **next admission $V_{t+1}$**).
    - **Model A (Tabular Cosine KNN)**: 127-D vector (5 $z$-standardized numeric counts + 10 unscaled one-hot categoricals) $\to$ **63.01% acc**.
    - **Model B (`Bio_ClinicalBERT` Cosine KNN)**: 768-D mean-pooled embedding of 15 authentic serialized fields $\to$ **62.96% acc**.
    - **Baselines**: Majority Class (`Uncomplicated / Other`) = **66.36%** | Random Neighbors = **62.13%**.
* **Why We Show It**: Establishes the rigorous, leakage-free experimental setup (no synthetic text, no same-patient neighbors) and poses the central puzzle: *Why do both models achieve identical ~63.0% accuracy and lose to a 66.4% dummy baseline?*
* **Verbatim Script (`0:00 – 0:40`, ~95 words)**:
  > *"Hello, we are Team SHAP Shifters. We investigate patient-similarity retrieval on 30,248 repeat-admission encounters from the UCI Diabetes dataset. Given an index visit, two k=10 cosine nearest-neighbor models predict the patient's complication pathway at their NEXT hospital admission: Uncomplicated, Cardiovascular, Kidney, or Eye disorders.*
  > *Model A uses a 127-dimensional tabular vector of standardized counts and one-hot categoricals; Model B uses 768-dimensional Bio_ClinicalBERT embeddings over authentic serialized records, with zero synthetic text or label leakage.*
  > *Both models achieve 63.0% accuracy—underperforming the 66.4% majority baseline. Let's see why."*

---

### Act 2: Pair A (AI-Aided) — Behavioral Walkthrough & Linked Studio (`0:40 – 2:35` | 115 sec)

* **What to Show (Live Screen Recording of `http://127.0.0.1:8050`)**:
  - **Scene 2A (`0:40 – 1:00`, 20s) — Chapter 1 (Reordered Confusion Matrices) & Chapter 2 (Calibration Cliff)**:
    - Start on **Tab 1: 🎬 Curated Storyline**. Point briefly to **Chapter 1** (0.0% recall on Kidney $n=757$ and Eye $n=563$), then scroll smoothly to **Chapter 2 (`graph-story-calibration`)**, highlighting the green **Signal Zone ($\ge 8/10$ vote share)**.
  - **Scene 2B (`1:00 – 1:30`, 30s) — Chapter 3 (127-D UMAP Manifold + Cramér's V Driver Table)**:
    - Scroll to **Chapter 3 (`graph-story-umap`)**. Point to the HDBSCAN clusters, longitudinal trajectories ($V_1 \to V_2 \to V_3$), and the **Top High-Dimensional Manifold Drivers** table (`change` $V=0.803$, `diabetesMed` $V=0.796$).
  - **Scene 2C (`1:30 – 2:00`, 30s) — Chapter 4 (Counterfactual Flips) & Chapter 6 (10-NN Dossier)**:
    - Scroll to **Chapter 4 (`graph-story-flips`)**: highlight **Sex Swap** (`8.9` vs `4.1`) and **HbA1c Elevation** (`4.4` vs `8.2`). Glance at **Chapter 6** showing **22.4% neighbor overlap** and green `Shared` badges.
  - **Scene 2D (`2:00 – 2:35`, 35s) — Chapter 5 (Disagreements & Deconfounded Subgroup Lift) + Quick Tab Switch to Tab 2/4 (Linked Studio & 100%-Normalized Cohort Profiler)**:
    - Point to **Chapter 5** (`<50 yrs` 58.5% vs 34.9%; `80+ yrs` 51.2% vs 46.0%; Racial Accuracy Lift $-3.1\text{ pp}$ vs $-3.3\text{ pp}$), then click **Tab 2 (Studio)** / **Tab 4 (Cohort Profiler)** to show how lassoing a cohort compares its clinical complexity against the 100% hospital baseline.
* **Why We Show It**: Hits every single requirement from Lectures 2 & 3, the Failed Team's rubric, and the Instructor's Feedback:
  1. Uses confusion matrix reordering as a top-level summary before drilling down.
  2. Exposes the test-time calibration cliff ($\ge 8/10$ vote share).
  3. Connects 2D UMAP patterns to high-dimensional drivers via Cramér's V and the 100%-normalized Cohort Profiler.
  4. Explains causal input-output sensitivity via counterfactual flips.
* **Verbatim Script (`0:40 – 2:35`, ~275 words)**:
  > *"In our AI-aided dashboard, Chapter 1's reordered confusion matrices show WHY aggregate accuracy fails: rare Kidney and Eye pathways make up under 4.5% of stays and achieve 0.0% recall under 10-neighbor voting.*
  > *Drilling down in Chapter 2, we plot empirical accuracy against test-time neighbor vote share with 95% Wilson confidence intervals. In 69.4% of stays, fewer than 8 neighbors agree, and accuracy drops to 59.6%—worse than random neighbors! Retrieval only beats the 66.4% majority baseline when at least 8 of 10 neighbors agree, reaching up to 77.6% accuracy.*
  > *Why don't neighbors group by future complication? In Chapter 3, we pair HDBSCAN clusters on our 127-D UMAP with Cramér's V association rankings. Rather than separating by future disease, the manifold is driven by current treatment flags: medication change at Cramér's V 0.803 and diabetes medication at 0.796.*
  > *In Chapter 4, single-attribute counterfactual edits reveal mirror-image representation artifacts: swapping patient sex replaces 8.86 of 10 neighbors in Model A—creating 94.7% same-sex homophily from unscaled binary one-hot columns—versus only 4.15 in ClinicalBERT. Conversely, elevating HbA1c to '>8' replaces 8.22 of 10 neighbors in ClinicalBERT due to WordPiece token pooling shifts, versus 4.36 in Model A.*
  > *In Chapter 5, across 5,918 model disagreements, Model A wins 58.5% to 34.9% on patients under 50, while ClinicalBERT wins 51.2% to 46.0% on patients over 80. Crucially, controlling subgroup accuracy for each race's majority base rate proves the raw African American gap—60.0% versus 63.7% Caucasian—is a base-rate artifact, with identical relative lifts of -3.1 and -3.3 percentage points.*
  > *Finally, switching to our coordinated Studio and Cohort Profiler lets us lasso any UMAP sub-cohort and compare its clinical vitals on a 100%-normalized hospital baseline scale."*

---

### Act 3: Pair B (Human-Only) — Independent Visual Exploration (`2:35 – 3:25` | 50 sec)

* **What to Show (Slide 2 & Slide 3)**:
  - **Slide 2 Headline**: *"Pair B (Human-Only): Class Imbalance Collapse & Error Structure"*
  - **Slide 3 Headline**: *"Pair B (Human-Only): Demographic Homophily & Perturbation Distributions"*
  - **Visual**: Pair B's clean static notebook visualizations created directly from `shared/data/*.parquet` without GenAI.
* **Why We Show It**: Demonstrates that Pair B independently investigated the shared research questions from the static data contract without AI code generation, establishing the baseline for the AI4Vis comparison.
* **Verbatim Script (`2:35 – 3:25`, ~115 words)**:
  > *"Working strictly human-only from the precomputed Parquet tables, Pair B conducted an independent visual investigation in Python.*
  > *First, examining per-pathway recall and prediction distributions, Pair B confirmed that severe class imbalance causes both cosine KNN models to collapse almost exclusively into Uncomplicated and Cardiovascular predictions, completely missing Renal and Ophthalmic progression.*
  > *Second, by plotting the full distributions of neighbor flips and neighbor demographic shares across age and sex brackets, Pair B independently verified that Tabular KNN strongly locks onto patient sex, whereas Bio_ClinicalBERT preserves cross-sex neighbor retrieval but shifts strongly when laboratory glucose and HbA1c strings are modified."*

---

### Act 4: AI4Vis Story — How GenAI Changed Our Process & Verification (`3:25 – 4:10` | 45 sec)

* **What to Show (Slide 4)**:
  - **Headline**: *"AI4Vis Reflection: Engineering Velocity vs. Methodological Verification Overhead"*
  - **Visual**: Side-by-side comparison diagram of **Pair A (AI-Aided)** vs. **Pair B (Human-Only)** across 3 concrete stories:
    1. **What AI Accelerated**: 4-module linked Dash studio, automated HDBSCAN + Cramér's V, 14 headless unit tests.
    2. **Where AI Failed & Required Human Audits**:
       - *Trap 1*: Early LLM prototype hallucinated symptom notes from diagnosis codes (label leakage $\to$ purged).
       - *Trap 2*: Binning calibration by ground-truth `agreement` (mathematical tautology $\to$ fixed to test-time `vote_share`).
       - *Trap 3*: Raw racial accuracy gap (confounded by base rate $\to$ fixed to subgroup majority lift).
    3. **Design Synthesis**: Uniting our exploratory 4-view cockpit with a curated 6-chapter storyline (Martini Glass architecture).
* **Why We Show It**: The instructor explicitly asked: *"Come up with a coherent story of how AI was beneficial for your analysis... criticize the model's suggestions where relevant."*
* **Verbatim Script (`3:25 – 4:10`, ~110 words)**:
  > *"Comparing our two workflows tells a clear story about AI in visualization. Generative AI gave Pair A massive engineering leverage—allowing us to build a 4-module coordinated dashboard with automated HDBSCAN clustering, Cramér's V rankings, and 14 unit tests that Pair B could not build manually in time.*
  > *However, AI assistance required intense methodological auditing. Early AI suggestions introduced synthetic symptom notes that leaked labels, plotted accuracy against ground-truth agreement—a circular tautology—and reported raw racial accuracy gaps without checking base rates.*
  > *Catching these pitfalls—and reconciling our complex exploratory studio with a curated 6-chapter narrative mode after Feedback 1—proved that human statistical reasoning remains essential."*

---

### Act 5: Phase II Plan — Opening the Black Box (`4:10 – 4:40` | 30 sec)

* **What to Show (Slide 5)**:
  - **Headline**: *"Phase II Roadmap: Opening the Black Box (Pairs Swap Roles)"*
  - **Visual**: Two-branch Phase II investigation diagram linking directly to Phase I findings:
    - **Technique 1 (Tabular KNN & Distance Attribution)**: Feature-wise cosine distance decomposition & SHAP attributions explaining the **8.86/10 sex-swap flip** and **<50 yrs advantage (58.5% vs. 34.9%)**.
    - **Technique 2 (`Bio_ClinicalBERT` Internals)**: Layer-wise representation probing (`phase2_model_internals.npz`) & multi-head self-attention maps explaining why `'>8'` HbA1c tokens trigger **8.22/10 neighbor flips** and why `change` ($V=0.803$) dominates the embedding geometry.
* **Why We Show It**: Directly connects Phase I black-box behavioral discoveries to Phase II white-box explanations using at least two complementary techniques per pair.
* **Verbatim Script (`4:10 – 4:40`, ~75 words)**:
  > *"In Phase II, our pairs swap AI roles and open the black box to explain three specific Phase I findings.*
  > *First, we will apply SHAP and exact cosine distance decomposition to Model A to quantify how unscaled one-hot columns override clinical vitals.*
  > *Second, using Bio_ClinicalBERT's saved layer activations and attention maps, we will trace across transformer layers how punctuation and lab tokens like 'A1C >8' dominate mean-pooled representations. Thank you!"*
