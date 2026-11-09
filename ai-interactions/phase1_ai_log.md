# AI4Vis log – Phase I (Pair A, AI-aided: Julian Bakan, Ahmad Hasan)

In Phase I, Pair A (Julian Bakan `k11913166`, Ahmad Hasan `k51821183`) works with generative AI and Pair B (Alfiia Ziganshina `k12456306`, Aqeel Jajja `k12554936`) works without it. The pairs swap roles in Phase II.

This log documents all AI-assisted steps in accordance with the JKU AI & Visualization Lab WS26 guidelines, specifically categorizing interactions under the **5 GenAI Roles** introduced in Lab 03:
1. **Role 1: GenAI as Tool to Transform Your Data**
2. **Role 2: GenAI as Collaborator to Design & Implement Visualizations**
3. **Role 3: GenAI as Collaborator to Critique Design & Findings**
4. **Role 4: GenAI as Collaborator to Steer Your Investigation**
5. **Role 5: GenAI as Tool to Interpret Observations**

---

## 1. Structured AI Interaction Log

| Date | Member | Tool | Lab 03 Role | Prompt (Summary) | AI Suggestion | Decision & Action | Rationale & Reflection |
|---|---|---|---|---|---|---|---|
| 2026-10-26 | Julian | LLM | Role 1 (Transform) | Generate Phase I baseline notebook, proposal PDF, and initial slides | Initial prototype with 4 views and synthetic symptom text generator | Accepted as starting draft | Flagged in code review for label leakage; required clean rebuild. |
| 2026-10-28 | Julian | Antigravity AI | Role 1 & 2 (Transform & Design) | Refactor `shared/` baseline into a static data contract and build Pair A coordinated dashboard | 1) Decouple `shared/` into precomputed Parquet/JSON artifacts; 2) Eliminate synthetic label-leaking notes; 3) Build 4-view linked Dash app | Accepted and implemented | Essential to preserve Phase I 2x2 study protocol: Pair B gets static data with zero AI touch; Pair A gets rich coordinated visual analytics. |
| 2026-11-02 | Julian | Antigravity AI | Role 3 (Critique) | Audit data loading performance, schema integrity, and discrete overplotting | 1) Add `use_threads=False` to `pd.read_parquet()` to eliminate PyArrow thread locks; 2) Add uniform jitter and 2D density contour overlays to View 1; 3) Modularize dashboard into `layout.py`, `callbacks.py`, and `test_dashboard.py` | Accepted and implemented | Discovered that PyArrow spawns thread pools that hang in sandboxed Windows environments; passing `use_threads=False` reduced load time from timeout to 0.11s. |
| 2026-11-02 | Ahmad | Antigravity AI | Role 4 (Steer) | Investigate retrieval disparity: why do Model A and Model B share only 22.4% neighbors with identical ~63% accuracy? | Focus analysis on counterfactual flip divergence (View 4) and confusion matrix reordering (View 2). Inspect whether Model A relies on demographic shortcuts while Model B relies on text phrasing. | Accepted and integrated into dashboard exploration flow | Steered team away from aggregate metric comparisons towards subgroup counterfactual auditing. |
| 2026-11-02 | Julian | Antigravity AI | Role 5 (Interpret) | How to interpret HDBSCAN clusters on clinical input UMAP without manual cherry-picking? | Compute statistical association metrics: Cramér's V for categorical features and correlation ratio ($\eta$) for numeric features against cluster labels. Annotate cluster medians with top driver. | Accepted and implemented | Cluster annotations now convey statistically grounded drivers (e.g. medication `change`, association 0.803) rather than subjective visual impressions. |
| 2026-11-03 | Julian | Antigravity AI | Role 3 & 5 (Critique & Interpret) | Deep statistical self-audit of calibration curves, subgroup fairness, and cohort profiler scales | 1) Replace circular `agreement` binning with test-time `vote_share` + 95% Wilson CIs; 2) Control subgroup accuracy for majority-class base rates (Accuracy Lift); 3) Normalize Cohort Profiler to 100% of hospital baseline mean | Accepted and implemented (`b1e3b42`) | Uncovered the $\ge 8/10$ calibration cliff (30.6% of stays, 68.2%–77.6% accuracy) and proved the African American vs. Caucasian raw accuracy gap (60.0% vs. 63.7%) is a base-rate artifact ($-3.1\text{ pp}$ vs. $-3.3\text{ pp}$ lift). |
| 2026-11-08 | Ahmad & Julian | Antigravity AI | Role 2 & 3 (Design & Critique) | Compare `Ahmad/phase1/ai_aided/dashboard` (single-page 6-section scroll) vs. `phase1/pair-a-ai-aided` (3-tab studio) after Feedback 1 | Harmonize into a Dual-Mode Martini Glass architecture: **Tab 1: Curated Storyline (6-Chapter Video Walkthrough)** + **Tabs 2–4: Retrieval Studio, Disparity Audit & Dynamic Cohort Profiler**, verified by 14 unit tests | Accepted and implemented | Resolves the presentation vs. exploration trade-off: gives a linear, curated 6-chapter narrative for the $\le 5$-minute video while preserving full multi-view linked brushing and high-dimensional driver analysis. |

---

*(See [`../ai4vis/phase1_ai_log.md`](../ai4vis/phase1_ai_log.md) and [`../ai_reflection_midpoint.md`](../ai_reflection_midpoint.md) for the complete architecture evolution, failure pattern catalog, and comparative reflection.)*
