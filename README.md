# VIS4AI_SHAP_SHIFTERS

Group project for **AI and Visualization Lab WS26** (JKU Linz, Visual Data Science Lab).

**Topic:** Explaining patient-similarity models. Given a diabetes patient, a k-nearest-neighbor model (k = 10) returns the most similar patients; a good neighbor set shares the patient's later complication pathway (kidney, cardiovascular, eye/neuro, or none).

| | Model | Representation |
|---|---|---|
| A | Count / clinical-feature KNN | Vector of diagnosis, medication and lab features plus age |
| B | Bio_ClinicalBERT KNN (optional: MedCPT) | 768-dim embedding of each record serialized as text |

**Dataset (current plan):** [UCI Diabetes 130-US Hospitals 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), public, no data use agreement.
The original plan used MIMIC-IV + CLMBR-T-base; it was dropped because of access approvals and setup cost (see [docs/decision_log.md](docs/decision_log.md)).

## Assignment phases

- **Phase I – How does the model behave?** Black box closed: investigative visualizations of inputs vs. neighbors (scatter plot with one variable per channel, reordered confusion matrices, input-feature UMAP with drill-down, admission paths, counterfactual neighbor flips).
- **Phase II – What did the model learn?** Black box open: embedding projections, probing classifiers, distance decomposition, embedding trajectories. Each view explains one Phase I observation.
- **AI4VIS:** two members design with generative AI, two without, then swap; prompts and accepted/rejected suggestions are logged in [docs/ai4vis_log.md](docs/ai4vis_log.md).

## Timeline

| Date | Milestone |
|---|---|
| 21.10.2026 | Group formation |
| 28.10.2026 | Task & model registration on Moodle |
| 11.11.2026 | Midpoint (Phase I) |
| 09.12.2026 | Final deadline, then mandatory presentations |

## Repository layout

```
docs/
  course_notes.md        summary of lectures and lab sessions relevant to the assignment
  decision_log.md        why we moved from MIMIC-IV/CLMBR-T-base to UCI + Bio_ClinicalBERT
  baseline_review.md     review of the shared Phase I baseline and checks to run
  ai4vis_log.md          AI4VIS prompt / decision log
  proposal_v1_mimic/     original MIMIC-IV proposal (document + slides), kept for reference
notebooks/               analysis notebooks
figures/                 aggregated figures only
data/                    local data, git-ignored
team-info.json           required by the course
workload.json            required by the course
```

## Setup

Download the UCI dataset into `data/` (it is git-ignored). Notebooks and requirements will be added with the shared baseline.
