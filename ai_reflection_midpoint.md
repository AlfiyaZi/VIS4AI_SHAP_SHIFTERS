# AI Reflection Report — Midpoint (Phase I)

**AI-Aided Pair (Pair A):** Julian Bakan (`k11913166`), Ahmad Hasan (`k51821183`)  
**Human-Only Pair (Pair B):** Alfiia Ziganshina (`k12456306`), Aqeel Jajja (`k12554936`)*  
**Course:** AI and Visualization (WS26), JKU Linz  
**Team:** SHAP Shifters  

> [!NOTE]
> **\*Note on Team Participation & Pair B Composition:**  
> Throughout the entirety of Phase I, Pair B's human-only analytical pipeline was conceived, developed, and executed solely and independently by **Alfiia Ziganshina** (`k12456306`). **Aqeel Jajja** (`k12554936`) has been completely inactive from the project start to the midpoint submission, making zero contributions to discussions, code, analysis, visualization, or documentation (0 commits across all repository branches).  

---

## 1. Phase I Experience: GenAI as Tool vs. Collaborator

In Phase I, Pair A investigated black-box patient similarity retrieval ($k=10$ KNN) over $30,248$ repeat admissions from the UCI Diabetes 130-US Hospitals dataset, comparing 127-D Tabular features against 768-D Bio_ClinicalBERT text embeddings across 4 complication pathways (*Uncomplicated/Other*: 66.4%, *Cardiovascular*: 29.3%, *Kidney/Renal*: 2.5%, *Eye/Neurological*: 1.9%).

Our experience confirmed the central thesis of **Lab 03**:
- **GenAI as a Tool** (executing specifications) accelerated routine engineering—scaffolding Dash layouts, vectorizing metrics, and authoring our 28-cell interactive guide notebook ([`dataset_and_models_guide.ipynb`](phase1/pair_A_ai_aided/dataset_and_models_guide.ipynb)).
- **GenAI as a Collaborator** (iterative back-and-forth) provided significant value in brainstorming hypotheses and auditing dimensionality, but required **constant, rigorous human skepticism**. Without active steering, GenAI introduced subtle, dangerous flaws (e.g., label-leaking synthetic text and circular calibration metrics).

---

## 2. Empirical Comparison: AI-Aided (Pair A) vs. Human-Only (Pair B)

| Dimension | Pair A (AI-Aided: Julian & Ahmad) | Pair B (Human-Only: Alfiia Ziganshina*) |
| :--- | :--- | :--- |
| **Development Velocity** | **High**: Rapidly built a 4-module interactive Dash web application and a 28-cell exploratory guide. | **Deliberate**: Authored clean, focused static Pandas/Matplotlib analysis directly over the shared data contract. |
| **System & Analysis Depth** | **Broad & Coordinated**: Built a linked multi-view studio (jittered scatter, Alsallakh confusion matrix, UMAP with patient trajectories, dynamic profiler, and 14 automated unit tests). | **Targeted & Hypothesis-Driven**: Conducted targeted slice investigations of minority class recall collapse and age-bracket complementarity. |
| **Verification Overhead** | **Very High**: Required catching LLM label leakage, resolving circular `agreement` vs. `vote_share` metrics, and deconfounding subgroup base rates. | **Low**: Every line of analytical logic and data aggregation was verified manually by construction. |
| **Key Complementary Insights** | Discovered the $\ge 8/10$ consensus cliff, proved UMAP clusters reflect medication changes ($V=0.803$), and uncovered mirror-image Sex Swap (8.86 vs 4.15) vs. HbA1c (4.36 vs 8.22) sensitivities. | Independently confirmed the 0.0% minority-pathway recall collapse on Kidney ($n=757$) and Eye ($n=563$) stays and analyzed model conflict brackets ($n=5,918$). |

---

## 3. Critical Failure Patterns Observed

1. **Preprocessing Hallucination & Label Leakage**: GenAI naturally tends to fill data gaps by synthesizing plausible text (e.g., generating clinical symptoms conditioned on diagnosis codes). Human code review was required to purge this synthetic text and enforce an authentic, decoupled data contract.
2. **Circular Evaluation Logic**: GenAI readily constructs tautological visualizations that look convincing but are mathematically trivial (e.g., plotting accuracy against ground-truth agreement). Human intervention replaced this with test-time `vote_share`.
3. **Subgroup Base-Rate Confounding**: When evaluating demographic fairness in imbalanced cohorts, GenAI flagged raw accuracy gaps as algorithmic bias, missing that they were an artifact of unequal baseline prevalence. Controlling for base-rate lift proved demographic parity.
4. **Presentation Disconnect**: When tasked with slide generation, GenAI produced dense, generic text slides. The team rejected automated slides, presenting directly from authentic live Dash screen captures.

---

## 4. Emerging Best Practices for AI in Visual Analytics

1. **Audit Data Lineage Before Model Evaluation**: Never trust AI-generated preprocessing without verifying that input features cannot leak future labels and that evaluation metrics are test-time valid.
2. **Pair 2D Projections with Quantitative Driver Rankings**: Supplement visual clustering (UMAP/t-SNE) with non-parametric association metrics (Cramér's V, $\eta$) to ground interpretations in data rather than subjective visual storytelling.
3. **Control for Subgroup Base Rates in Imbalanced Cohorts**: In skewed datasets (66.4% majority class), evaluate Accuracy Lift over subgroup baselines with Wilson score confidence intervals rather than raw accuracy.
4. **Use the "Martini Glass" Architecture**: Reconcile presentation brevity and grading depth by pairing an author-driven linear narrative (Tab 1: 6-Chapter Storyline for $\le 5$-minute videos) with a reader-driven exploratory cockpit (Tabs 2–4: Linked Studio).
5. **Back Interactive UI with Headless Automated Tests**: Ensure multi-view brushing and callback persistence remain stable through simulated HTTP unit test suites (14 passing tests in `test_dashboard.py`).

---

## 5. Transition to Phase II

In Phase II (*"What did the model learn?"*), **the pairs will switch roles**: Pair B will become AI-aided, while Pair A will operate human-only. We will open the black box to analyze internal representations, layer-wise embeddings, and SHAP/attention attributions, building directly upon our Phase I findings.
