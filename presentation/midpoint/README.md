# Midpoint Progress Presentation (≤ 5 min, due 11.11.2026)

In accordance with course instructions (`course_materials/Moodle_page.txt` and midpoint announcements):
- The **Midpoint Progress Update Video** (duration ≤ 5 minutes) is uploaded to Google Drive with public access (`Anyone with the link can view`).
- The public video link and public GitHub repository link are submitted directly via Moodle by editing the existing group registration entry at:
  https://moodle.jku.at/mod/data/view.php?id=13885583
- Optional tracking in repository: [`../presentation_video.json`](../presentation_video.json)

---

## Video Content & Storyline Structure (≤ 5:00 Total)

1. **Problem & Clinical Retrieval Task (0:00 – 0:45)**:
   - Patient similarity retrieval on $N=30,248$ repeat admissions (UCI Diabetes 130-US Hospitals).
   - Comparing Model A (127-D Tabular KNN) vs. Model B (768-D Bio_ClinicalBERT KNN) across 4 complication pathways.
   - The aggregate accuracy illusion: both models achieve 63.0%, failing to beat the naive 66.4% majority baseline and sharing only 22.4% of retrieved neighbors.
2. **Pair A (AI-Aided) Behavioral Discoveries (0:45 – 2:30)**:
   - *Calibration Cliff & Danger Zone*: Below 80% vote share, KNN retrieval cannot be trusted; at $\ge 80\%$ consensus, accuracy reaches 70.7%–77.6%.
   - *127-D UMAP Manifold Drivers*: Cramér's $V$ proves clusters reflect administrative medication change ($V=0.803$), not biological complication pathways ($V=0.04$).
   - *Counterfactual Sensitivity*: Model A takes demographic shortcuts (Sex swap flips 8.86/10 neighbors); Model B is perturbed by clinical token edits (HbA1c elevation flips 8.22/10 neighbors).
   - *Specialization & Fairness*: Tabular KNN dominates on young patients ($<50$y, 58.5% win rate); Bio_ClinicalBERT edges out on geriatric multimorbid stays ($80+$y, 51.2%).
3. **Pair B (Human-Only) Comparative Analysis (2:30 – 3:45)**:
   - Human-only exploration in Pandas/Matplotlib (`phase1/pair_B_human_only/`).
   - Discovery of the 35.3% emergency generic text bottleneck causing BERT neighbor crowding ($d_{10}=0.0006$).
   - Proof that flat majority voting silences valid minority signals for Kidney and Eye complications.
4. **AI4Vis Methodological Reflection & Phase II Roadmap (3:45 – 5:00)**:
   - GenAI accelerated complex dashboard implementation (Dual-Mode Dash studio, 14 unit tests, interactive cohort profiling).
   - Critical human auditing: catching synthetic text label leakage, tautological calibration metrics, and base-rate confounding.
   - Phase II Roadmap: Opening the black box via SHAP feature attributions on Model A distance metrics and attention map probing across Bio_ClinicalBERT layers.
