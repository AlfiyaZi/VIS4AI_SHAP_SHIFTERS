# Phase I plan – How do the models behave?

Black box closed: we look only at what goes in and what comes out.

- **Input:** one patient admission (age, sex, race, HbA1c, glucose, length of stay, lab / medication / diagnosis counts, insulin, …).
- **Output:** the k = 10 neighbors, plus what we derive from them: predicted pathway (majority vote), neighbor agreement (the model's "confidence"), the neighbors' demographics, and how many neighbors change when we edit the input.
- **Not allowed in Phase I:** embeddings, layers, attention, weights (`shared/data_cache/phase2_model_internals.npz` stays closed until Phase II).

## Step 0 – fix the shared pipeline (before any analysis)

See [../docs/baseline_review.md](../docs/baseline_review.md) for details.

- [ ] Remove the generated "clinical presentation" sentence; serialize only real UCI fields.
- [ ] Missing `torch` / `transformers` or a failed download raise an error (no fallback encoder, no synthetic cohort); write the encoder name and data source into the outputs.
- [ ] Use all repeat-admission patients, not a 384-patient sample; report class sizes per pathway.
- [ ] Tighten the ICD-9 pathway mapping; label = real next admission only.
- [ ] Add baselines: majority class and random neighbors.

## Research questions

| # | Question | Pattern type (Lab 02) | Visualization | In-depth follow-up |
|---|---|---|---|---|
| 1 | Do neighbors share the patient's next-admission pathway, and where in patient space does this fail? | Errors | Input-feature UMAP colored by agreement, A vs. B side by side; confusion matrices reordered by error similarity for A, B and baseline | Select the failing region and compare its features with successful patients (small multiples). What distinguishes it? Does it have sub-groups? |
| 2 | Is the model confidently wrong? | Confidence | Agreement vs. correct / incorrect, per pathway | Which patients get a confident but wrong vote, and what do they share? |
| 3 | Are neighbors chosen by demographics rather than clinical state? | Bias | Share of same-sex / same-race neighbors vs. the share expected by chance, per group | Does it differ between A and B? Is one group systematically worse matched? |
| 4 | How sensitive are neighbors to edits of the same record? | Perturbations, counterfactuals | Neighbor-flip distributions for clinical edits (HbA1c, medications) vs. demographic edits (sex, age, race); for Model B also a reworded template with identical content | If Model B flips on sex or wording but not on HbA1c, that is the key finding to explain in Phase II. |
| 5 | *(optional)* Does the rare Eye/Neuro pathway fail? | Training data as hypothesis source | RQ 1–2 restricted to rare vs. common pathways | Hypothesis H2 of the proposal |

Rules: every finding cites a figure or number the notebook actually produces; show the data behind each pattern; put the explanation on the figure; answer every question, including negative results.

## Who does what

- **All four (one meeting):** agree on the questions, data and starting visualization; assign Pair A and Pair B in the [README](../README.md).
- **Pair A – AI-aided** ([pair_A_ai_aided/](pair_A_ai_aided/)): the same questions with richer interaction (e.g. a linked dashboard: select patients in the UMAP and see their neighbors and features). Log prompts and accepted / rejected suggestions in [../ai4vis/phase1_ai_log.md](../ai4vis/phase1_ai_log.md).
- **Pair B – human-only** ([pair_B_human_only/](pair_B_human_only/)): own code from the shared outputs, not a copy of Pair A's notebook.
- **End of phase:** compare insights, visualization choices, speed vs. verification effort and confidence in [../ai4vis/reflection.md](../ai4vis/reflection.md).

## Timeline to the midpoint

| When | What |
|---|---|
| 28.10 23:59 | Moodle task-selection entry |
| 29.10 – 01.11 | Fix the pipeline; group meeting; assign pairs |
| 02.11 – 06.11 | Pairs work separately; book a Feedback 1 slot (05./06.11) |
| 07.11 – 10.11 | In-depth follow-ups; AI log; pair comparison; record the video |
| **11.11 23:59** | Submit code of both pairs, Phase I AI log, video ≤ 5 min (problem → what each pair did and found → key figures → Phase II plan) |
