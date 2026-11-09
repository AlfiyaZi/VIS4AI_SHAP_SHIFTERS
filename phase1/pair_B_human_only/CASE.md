# Pair B case – Phase I (human-only)

Alfiia Ziganshina, Aqeel Jajja. Same problem as Pair A, own analysis, no generative AI for analysis, code or visualization design.

## Problem (shared with Pair A)
Two k = 10 cosine KNN models retrieve similar patients for a diabetes stay: Model A (127-D clinical features) and Model B (Bio_ClinicalBERT on the record as text). Neighbors are good when they share the next-admission pathway. Data: `shared/data/` (30,248 stays, 16,773 patients). Black box closed: inputs, neighbors, votes and edit flips only.

## What the tutor asked for (05.11, see [../../docs/feedback1_notes.md](../../docs/feedback1_notes.md))
- Few, informative charts, each interpreted; no pure performance charts.
- Show how inputs influence the output and how the output changes when an attribute changes.
- Overview first (e.g. a reordered confusion matrix), then drill down.

## Already covered by Pair A (avoid duplicating; compare instead)
From `phase1/pair_A_ai_aided/` on branch `phase1/pair-a-ai-aided`: input UMAP colored by the strongest drivers, reordered confusion matrices, vote-share calibration, A-vs-B disagreement by age and pathway, subgroup accuracy vs. base rate, cohort profiler, mean neighbor flips per edit.

## Our angle (to be decided by Pair B)
- Question(s):
- Why this differs from Pair A:
- Overview view:
- Drill-down view(s):
- What would confirm / refute it:

## Deliverables by 11.11
- [ ] `pairB_phase1.ipynb`: every finding backed by a figure or number produced there
- [ ] 1–2 key figures in `figures/`
- [ ] `work_log.md`: time spent and what was dropped, for the pair comparison
- [ ] Pair B part of the midpoint video, as part of one team story
