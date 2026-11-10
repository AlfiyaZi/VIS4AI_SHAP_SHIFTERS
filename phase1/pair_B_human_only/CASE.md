# Pair B case – Phase I (human-only)

Alfiia Ziganshina, Aqeel Jajja. Same problem as Pair A, own analysis, no generative AI for analysis, code or visualization design.

#

## Our angle
- **Question(s):** What does each model match on (RQ3), how stable are its neighbours per stay –
  for whom, and does it change the prediction (RQ4) – and why are the rare pathways never
  predicted (rare-pathway part of RQ1, RQ5)?
- **Why this differs from Pair A:** Pair A shows *what* happens, as averages. We drill down
  *per stay*: for whom and under which inputs it happens.
- **Overview view:** matching profile – lift over random neighbours per input field (Fig. 1).
- **Drill-down view(s):** neighbours replaced per stay after one edit (Fig. 2); crowding vs.
  flips (Fig. 3); fragile stays in the input UMAP and fragile vs. stable fields (Fig. 4a/4b);
  prediction change (Fig. 5); same-pathway neighbours for rare pathways (Fig. 6).
- **What would confirm / refute it:** confirmed if fragility concentrates in an input-defined
  group (it does for Model B: emergency admissions with unknown specialty) and the rare-pathway
  signal is present but outvoted (it is). Refuted if flips were uniform across stays or the
  neighbours of rare stays were at random level (true for kidney, not for eye/neuro in Model A).

## Narrowed and dropped
| Idea | Decision                                                                  | Why |
|---|---------------------------------------------------------------------------|---|
| Raw match shares (B1) | narrowed to lift over random                                              | raw shares hide chance agreement |
| "Model B ignores the numbers" (B2) | narrowed (one example card → all stays (Fig. 1b) "ignores" → "weighs less" | one case is not evidence |
| A–B overlap by specialty (B5) | dropped                                                                   | duplicates Pair A's A-vs-B disagreement |
| Same patient across stays (B6) | dropped                                                                   | answers no team RQ |
| Anti-hubs | dropped, one line kept in section 7                                       | no pattern |
| Hubs (B7) | kept as negative result                                                   | negative results are reported |

#