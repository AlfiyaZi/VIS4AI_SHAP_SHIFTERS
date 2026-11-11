# AI Reflection Report — Midpoint (Phase I)

**AI-Aided Pair (Pair A):** Julian Bakan (`k11913166`), Ahmad Hasan (`k51821183`)  
**Human-Only Pair (Pair B):** Alfiia Ziganshina (`k12456306`), Aqeel Jajja (`k12554936`)*  
**Course:** AI and Visualization (WS26), JKU Linz  
**Team:** SHAP Shifters  

> [!NOTE]
> **\*Note on Team Participation & Pair B:**  
> Throughout Phase I, Pair B's human-only work was carried out entirely by **Alfiia Ziganshina** (`k12456306`). **Aqeel Jajja** (`k12554936`) was inactive from the beginning and contributed no code, analysis, or text to the project (0 commits across all branches).

---

## 1. Our Experience Working with GenAI in Phase I

In Phase I, we evaluated two $k=10$ nearest-neighbor retrieval models on the UCI Diabetes dataset (30,248 repeat admissions across 4 complication pathways: Uncomplicated 66.4%, Cardiovascular 29.3%, Kidney 2.5%, Eye/Neuro 1.9%). We compared a 127-D tabular clinical feature baseline against 768-D Bio_ClinicalBERT text embeddings.

Our pair (Pair A) used GenAI tools (Claude 3.5 Sonnet and Gemini via Antigravity) to help us build an interactive visualization dashboard and analyze model behavior.

**Where AI helped us a lot:**
- **Fast implementation**: It helped us write boilerplate code for our Dash web app, build Plotly charts, set up callback structures, and create our exploratory guide notebook (`dataset_and_models_guide.ipynb`). What would have taken weeks to code by hand was up and running in a few days.
- **Brainstorming code ideas**: It suggested useful statistical methods, such as using Cramér's V to measure which features correlate most strongly with the clusters in our 2D UMAP projection.

**Where AI created problems and required constant checking:**
- AI output can look completely convincing while being mathematically or logically wrong. We quickly learned that we could not trust any metric, plot, or preprocessing script without personally checking the raw numbers and code logic.
- We spent a huge portion of our time debugging AI suggestions, fixing race conditions in Dash callbacks, and correcting bad analytical assumptions.

---

## 2. Comparing Pair A (AI-Aided) and Pair B (Human-Only)

| Aspect | Pair A (AI-Aided: Julian & Ahmad) | Pair B (Human-Only: Alfiia Ziganshina*) |
| :--- | :--- | :--- |
| **Speed & Scope** | Built a full 4-view interactive Dash app with coordinated cross-filtering, UMAP projections, patient trajectories, and a test suite. | Worked directly in Python/Pandas, creating clean, targeted static plots without UI overhead. |
| **Verification Effort** | **Very high**: Had to catch subtle data leakage, fix circular plots, debug callback loops, and rewrite AI-generated slides. | **Low**: Wrote every line of code manually, so data flow and logic were understood from the start. |
| **Key Findings** | Discovered that UMAP clusters reflect administrative medication changes (`change` $V=0.803$) rather than disease, and found the $\ge 8/10$ consensus cliff. | Focused on the class-imbalance problem, confirming 0% minority recall on rare complications (Kidney and Eye pathways) and inspecting model disagreement brackets. |

---

## 3. Four Major AI Mistakes We Had to Catch and Fix

1. **AI Invented Fake Clinical Notes (Label Leakage):**  
   In our first shared baseline script, the AI wrote a function to generate "doctor notes" from the data. But it created these notes using the encounter's *diagnosis codes*, which gave the model the answer in advance! We caught this in code review, deleted the generator, and restricted the model strictly to 15 real dataset fields.

2. **Circular Calibration Plot (Fake 100% Accuracy):**  
   The AI created a calibration plot that binned accuracy against neighbor "agreement" with the *true* label. In $k=10$ KNN, having $\ge 6/10$ neighbors match the true label means the model always predicts correctly by definition. This resulted in a useless, circular 100% line. We fixed it by using the model's test-time vote share instead, which revealed the real finding: accuracy drops below baseline unless at least 8 out of 10 neighbors agree.

3. **Confusing Cohort Base Rates with Model Bias:**  
   When checking demographic fairness, the AI pointed out that African American patients had ~3.7% lower accuracy and suggested this was algorithmic bias. When we checked the underlying numbers, we found that this group simply had fewer cases of the majority class (63.1% vs 66.9%). When measured relative to each group's baseline (Accuracy Lift), both groups were affected almost identically ($-3.1$ vs $-3.3$ pp).

4. **Generic, Text-Heavy Presentation Slides:**  
   When asked to help prepare presentation slides, the AI produced walls of generic text and Beamer bullet points that completely disconnected from our actual visual findings. We threw out the AI slides and decided to present live screen recordings of our interactive dashboard instead.

---

## 4. Practical Best Practices We Learned

1. **Check Data Pipelines First**: Always verify that input features do not accidentally leak the prediction target, especially when AI generates preprocessing code.
2. **Ground Visual Patterns with Concrete Numbers**: Don't rely on visual interpretation of 2D scatter plots or UMAP clusters alone; calculate correlations (like Cramér's V) to confirm what is actually causing the grouping.
3. **Account for Base Rates in Imbalanced Data**: When one class makes up 66% of the dataset, raw subgroup accuracy is misleading. Always measure performance relative to the group's baseline rate.
4. **Automate Testing for Interactive Apps**: Having an automated test suite (`test_dashboard.py`, 14 tests) was crucial to prevent callbacks and data filters from breaking when refining the UI.

---

## 5. Next Steps for Phase II

In Phase II (*"What did the model learn?"*), the two pairs will switch roles as required by the course: Pair B will use AI assistance, and Pair A will work human-only. We will look inside the models (internal embeddings, layer representations, and feature attributions) to understand why the two models rely on such different features despite achieving similar overall accuracy.
