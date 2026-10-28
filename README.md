# VIS4AI_SHAP_SHIFTERS

Team **SHAP Shifters**, AI and Visualization Lab WS26 (JKU Linz, Visual Data Science Lab).

## Task and model

**Task:** patient-similarity retrieval for clinical-trial cohort matching. Given a diabetes admission, a k-nearest-neighbor model (k = 10) returns the most similar patients. A neighbor set is good when it shares the patient's later complication pathway (kidney, cardiovascular, eye/neuro, or none).

| | Model | Representation |
|---|---|---|
| A | Count / clinical-feature KNN | Diagnosis, medication and lab features plus age |
| B | [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) KNN | 768-dim embedding of each record serialized as text |

**Dataset:** [UCI Diabetes 130-US Hospitals 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), public, no data use agreement. The original MIMIC-IV + CLMBR-T-base plan was dropped; see [docs/decision_log.md](docs/decision_log.md).

## Research questions

<!-- Agree on these at the start of each phase. Answer every one of them explicitly in the results. -->

**Phase I – How do the models behave?** (black box closed)
1. Do the retrieved neighbors share the query patient's later complication pathway, and where in patient space does this fail?
2. Are neighbors chosen by demographics (sex, race, age) rather than clinical state?
3. How sensitive are neighbors to small clinical vs. demographic edits?

**Phase II – What did the models learn?** (black box open)
- To be fixed after the midpoint, each RQ explaining one Phase I finding.

## Structure

The two pairs switch roles after Phase I, as the assignment requires.

| Pair | Members | Phase I | Phase II |
|---|---|---|---|
| A | Julian Bakan, Ahmad Hasan | AI-aided | human-only |
| B | Alfiia Ziganshina, Aqeel Jajja | human-only | AI-aided |

```
shared/                      common starting point agreed by all four members
  01_prepare_baseline_data.py  cohort, Model A/B representations, KNN, counterfactual flips -> shared/outputs/
  02_phase1_shared_baseline.ipynb
phase1/                      Phase I: How do the models behave? (plan: phase1/README.md)
  pair_A_ai_aided/
  pair_B_human_only/
phase2/                      Phase II: What did the models learn?
  pair_A_human_only/
  pair_B_ai_aided/
ai4vis/                      AI interaction records and the comparison of the two pairs
  phase1_ai_log.md
  phase2_ai_log.md
  reflection.md
presentation/
  midpoint/                  progress video (<= 5 min) slides
  final/                     final slide deck (10-min recorded presentation)
  presentation_video.json    video links
docs/                        decisions, reviews, course notes, original proposal
environment.yml
team-info.json               required by the course
workload.json                required by the course
```

Data (`data/`, `data_cache/`, `outputs/`) is never committed.

## Deliverables

| Due | Deliverable | Where |
|---|---|---|
| 28.10.2026 | Task selection on Moodle | — |
| 11.11.2026 | Code from both pairs, Phase I AI record, video ≤ 5 min | `phase1/`, `ai4vis/phase1_ai_log.md`, `presentation/midpoint/` |
| 09.12.2026 | All code, AI records of both phases, 10-min recorded presentation + slides | `phase2/`, `ai4vis/`, `presentation/final/` |

Feedback slots: 05./06.11 and 26.11.–04.12. Live presentations: 10.–17.12.2026 (attend at least one). Full schedule: [docs/course_notes.md](docs/course_notes.md).

## Setup

```bash
conda env create -f environment.yml
conda activate shap_shifters
cd shared
python 01_prepare_baseline_data.py   # downloads UCI into shared/data_cache/, writes shared/outputs/
jupyter lab
```

Pair notebooks read the shared outputs from `../../shared/outputs/`. Known issues in the baseline: [docs/baseline_review.md](docs/baseline_review.md).
