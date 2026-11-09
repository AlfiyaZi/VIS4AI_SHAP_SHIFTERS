# AI4Vis reflection: AI-aided vs. human-only

Filled in at the end of each phase, when both pairs compare processes and findings.
*(See also [`../ai_reflection_midpoint.md`](../ai_reflection_midpoint.md) in the repository root for the full Midpoint Reflection Report.)*

## Phase I (Pair A: Julian Bakan `k11913166`, Ahmad Hasan `k51821183` [AI-Aided] vs. Pair B: Alfiia Ziganshina `k12456306`, Aqeel Jajja `k12554936` [Human-Only])

- **Insights found by one pair but missed by the other:**
  - *Pair A (AI-aided)* uncovered the **$\ge 8/10$ test-time vote-share calibration cliff** (showing that in 69.4% of stays where neighbor consensus is $<80\%$, both models perform at 59.6%–59.7%, worse than the 66.36% majority baseline and 62.13% random baseline, only beating baseline at $\ge 8/10$ where accuracy reaches 68.2%–77.6%), proved via **HDBSCAN + Cramér's V** that the 127-D UMAP manifold is organized by treatment workflow flags (`change` $V=0.803$, `diabetesMed` $V=0.796$) rather than future complication pathway, and proved that the 3.7 pp raw racial accuracy gap (60.0% African American vs. 63.7% Caucasian) is a **majority base-rate confounding artifact** ($-3.1\text{ pp}$ vs. $-3.3\text{ pp}$ accuracy lift).
  - *Pair B (Human-only)* independently confirmed the **0.0% minority-pathway recall collapse** on Kidney/Renal ($n=757$) and Eye/Neurological ($n=563$) stays and inspected age-bracket complementarity across the 5,918 model conflicts.
- **Differences in visualization choices:**
  - *Pair A* engineered a 4-module reactive Dash web application combining a **6-Chapter Curated Storyline** with a **4-View Coordinated Studio** (jittered scatter with 2D contours, Alsallakh-reordered confusion matrices, UMAP with longitudinal patient trajectories $V_1 \to V_2 \to V_3$, 100%-normalized Cohort Profiler, and a 10-NN cosine similarity dossier).
  - *Pair B* relied on clean, reproducible static Pandas/Matplotlib/Seaborn plots directly over the shared Parquet tables.
- **Breadth vs. depth:**
  - GenAI allowed *Pair A* to achieve both high breadth (simultaneously testing all 15 input features via Cramér's V / correlation ratios and profiling 6 clinical vitals) and statistical depth (95% Wilson CIs, base-rate deconfounding). *Pair B* focused deeply on targeted, hypothesis-driven slices without UI engineering overhead.
- **Speed vs. effort spent verifying and correcting AI output:**
  - While GenAI accelerated UI scaffolding and statistical code generation by an order of magnitude, **verification overhead was substantial**: human code review caught synthetic clinical note label leakage in the initial prototype, identified circular `agreement` vs. `vote_share` calibration binning, and resolved a UI design divergence between an exploratory multi-tab studio and a linear presentation scroll (leading to our unified Dual-Mode Martini Glass architecture verified by 14 automated unit tests).
- **Confidence in conclusions, and the evidence behind it:**
  - High confidence achieved only *after* human mathematical auditing: every headline claim is backed by exact `baseline_metrics.json` counts, 95% Wilson score confidence intervals, per-bracket sample size annotations ($n$, flagging $n < 50$), and 14 passing headless unit tests.

## Phase II

*(To be completed at the end of Phase II after pairs switch roles.)*
