# Midpoint Progress Presentation Playbook (≤ 5 min, due 11.11.2026)

**Deliverables:**
- **Compiled LaTeX Beamer Presentation:** [`phase1_presentation.pdf`](phase1_presentation.pdf) (10 slides, official JKU theme)
- **LaTeX Source Code:** [`Latex/phase1_presentation.tex`](Latex/phase1_presentation.tex)
- **Full 5-Minute Master Script & Cues:** [`phase1_presentation_script.md`](phase1_presentation_script.md)
- **Alternate Markdown Script:** [`../../phase1/pair_A_ai_aided/video_script_and_slides.md`](../../phase1/pair_A_ai_aided/video_script_and_slides.md)
- **Final Video URL:** Save link to [`../presentation_video.json`](../presentation_video.json) once recorded.

---

## Master Timing & Slide Structure (300 Seconds / 5:00 Total)

| Slide | Title & Theme | Visual Content (1 Figure / Table + 1 Takeaway) | Key Takeaway (On Slide Banner) | Speaker Window |
| :---: | :--- | :--- | :--- | :---: |
| **0** | **Title Slide** | JKU Beamer frame with Team SHAP Shifters & all 4 members | Course Overview & Sub-Team Identification | `0:00 – 0:10` (10s) |
| **1** | **Clinical Retrieval & Baseline Paradox** | 4-bar accuracy chart (66.4% Maj vs 63.0% A & B vs 62.1% Rand) + Alsallakh error table | *Identical 63.0% global accuracy masks disjoint neighborhoods (22.4% overlap) and 0% minority recall under flat voting.* | `0:10 – 0:50` (40s) |
| **2** | **RQ1: Calibrating Trust ($\ge 80\%$)** | Test-time consensus calibration curve: Noise Zone (<80%, 59.6%) vs Signal Zone ($\ge 80\%$, 70.7%--77.6%) | *Clinicians should abstain when neighbor consensus is <80%; unanimous votes yield +11.2 pp accuracy gain.* | `0:50 – 1:30` (40s) |
| **3** | **RQ2: 127-D Manifold Drivers** | Cramér's $V$ feature driver table paired with 127-D UMAP island clustering schematic | *127-D UMAP partitions patients by medication management protocol, not future disease trajectory (V=0.04).* | `1:30 – 2:10` (40s) |
| **4** | **RQ4: Perturbation Shortcuts** | Paired bar chart: Sex Swap (8.9 A vs 4.1 B) vs HbA1c elevation (4.4 A vs 8.2 B) | *Mirror vulnerabilities: Model A is distorted by unscaled binary flags; Model B is fragile to token edits.* | `2:10 – 2:50` (40s) |
| **5** | **Neighborhood Geometry (Pair B)** | ER text bottleneck ($d_{10}=0.0006$, 35.3% stays) + $k=10$ majority voting mask | *Model B fragility stems from 35% generic ER text crowding; majority voting masks real minority signals.* | `2:50 – 3:35` (45s) |
| **6** | **RQ3: Specialization & Fairness** | Age conflict win rates ($<50$y Model A 58.5% vs $80+$y Model B 51.2%) + Subgroup base-rate deconfounding | *Tabular KNN excels on acute youth readmissions; ClinicalBERT captures geriatric multimorbidity.* | `3:35 – 4:15` (40s) |
| **7** | **Team Synthesis: AI vs. Human** | Division of strengths matrix (Macro-observatory vs Micro-mechanisms) + 3 AI flaws caught | *AI multiplies implementation velocity; human visual auditing and domain inquiry ensure clinical validity.* | `4:15 – 4:45` (30s) |
| **8** | **Phase II Roadmap & Conclusion** | 4 actionable milestone cards: Confidence, Scaling Fix, Text Disentangle, SHAP Explainers | *Phase I established the diagnostic baseline; Phase II delivers calibrated, equitable patient retrieval.* | `4:45 – 5:00` (15s) |

---

## Pair A (AI-Aided) vs. Pair B (Human-Only) Comparative Analysis

| Analytical Dimension | Pair A (AI-Aided: Julian & Ahmad) | Pair B (Human-Only: Alfiia & Aqeel) | Team Synthesis & Impact |
| :--- | :--- | :--- | :--- |
| **Primary Scope** | Macro-Observatory & System Analytics | Micro-Mechanisms & Geometry | Macro views identify global phenomena; micro views explain the physical cause. |
| **Artifacts Produced** | Full Dash web app (Dual-Mode Story + Studio, 4 tabs, 14 unit tests) | Static Jupyter analysis (`pairB.ipynb`, scripts `01`--`06`) | AI accelerated complex front-end engineering; human pair focused on deep geometric modeling. |
| **Latent Space Insights** | 127-D UMAP + Cramér's $V$ ($V=0.803$ med change drives clusters, not future target) | Crowded neighborhoods: Model B median $d_{10}=0.0006$ vs Model A $0.113$ | UMAP shows macroscopic cluster separation; distance gaps prove microscopic crowding. |
| **Model B Fragility Cause** | Quantified 8.22/10 neighbor turnover under HbA1c elevation | Discovered the **35.3% emergency + missing specialty generic text bottleneck** | Explain why BERT flips: generic administrative boilerplate collapses 10,684 stays into identical space. |
| **Rare Pathway Failure** | Reordered confusion matrix exposed 0.0% Kidney & Eye recall | Proved Model A retrieves Eye/Neuro neighbors above chance; majority vote silences them | Separates model blindness from voting aggregation artifacts. |
| **Decision Reliability** | Disproved circular agreement; established test-time `vote_share \ge 80%` rule | Analyzed relative distance gap $(d_{10}-d_1)/d_{10}$ as an instability indicator | Operational threshold for clinical practice. |
| **AI Role & Failures** | Code generation & metric formulation; **caught 3 AI errors** (leakage, tautology, bias) | Baseline human control without generative AI | Proves GenAI requires domain-literate human supervision. |
