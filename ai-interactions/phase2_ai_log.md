# Phase II – AI interaction log

Phase II roles: **Pair A human-only** (Julian Bakan, Ahmad Hasan), **Pair B AI-aided** (Alfiia Ziganshina, Aqeel Jajja).

## Pair B (AI-aided): Model B explained from the inside

**Tool:** Claude Code (Claude Opus), used as a collaborator for planning, code, figures and text. Branch `feature/phase2-pair-b-ai-aided`. All numbers in the notebook come from executed code; captions were checked against the printed outputs.

> **Draft, to be reviewed by Pair B.** The "Human decision" column must reflect what the pair actually decided.

### Important prompts

| # | Prompt (Pair B) | What the AI did |
|---|---|---|
| P1 | "Build the solution for Phase II in a feature branch." | Read the course slides (Lab 02, Lab 03, Lectures 03–05) and the repo; found no Phase II material (no embeddings, no model weights). Proposed a plan: one Phase II question per Phase I finding, with techniques from Lectures 04/05. |
| P2 | "Do Model B first." / "We explained A and B in Phase I; overlap is meant, compare at the end." | Scoped to Model B; chose a field-level Shapley game that also works for Model A, so the final comparison uses the same method. |
| P3 | "Phase II: visualise how the model works inside, how it classifies." | Added two views: per-layer projection (Lecture 03) and a one-stay walk-through record → tokens → vector → neighbours → vote (explaining by example, Lecture 05). |
| P4 | "Look at last year's lectures and methods like SHAP and others we can use to see what happens inside the model's layers and how it classifies." | Read 2025 Lectures 04–07 and 2026 Lecture 04; mapped each method (PDP/ICE, LIME, SHAP, saliency, Grad-CAM, examples, prototypes, network dissection) to what was already used; added a classifier per layer, Grad-CAM for text and ICE/PDP. Skipped LIME (overlaps with Shapley). |


### Suggestions: accepted, refined, rejected

| Step | AI suggestion | Outcome | Why |
|---|---|---|---|
| Reproduction | Re-embed all 30,248 stays and check Phase I is reproduced before explaining anything | Accepted | 9.97 of 10 neighbours reproduced, median d10 0.00060 vs. 0.00060 |
| Device | Run Bio_ClinicalBERT on the Apple GPU (MPS) | **Rejected (AI error)** | Crashed on macOS 14; switched to CPU, as in Phase I |
| Field spans | Regex to find each field's value in the text | **Refined (AI error)** | First version silently missed "Max Glucose" (label "A1C" contains a digit); caught by checking all 30,248 texts |
| Concept probes | Linear probe per layer for each field and the outcome | **Refined** | Every input field scored ~1.0 in every layer: trivially true, because the fields are in the text. Added variance share per field to measure how strongly a field shapes the distances |
| Shapley game | Mask fields by deleting them | **Rejected by the AI itself** | Deleted fields give unrealistic texts; used an interventional game instead (take field values from another real stay), so every coalition is a valid record. Efficiency check: error ≤ 4 × 10⁻⁷ |
| Walk-through vote | Majority vote with alphabetical tie-break | **Refined (AI error)** | Differed from Phase I's rule (tie → best-ranked neighbour); after the fix the prediction matches Phase I |
| Layer projection | Colour the outcome by drawing classes in order | **Refined twice (AI error)** | First version hid the rare pathways under the majority class, the second exaggerated them with larger dots; final version draws points in random order so the visible mix matches the real proportions |
| Fig. 2 caption | "Below a ratio of 1 few neighbours change, above it nearly all" | **Corrected (AI over-claim)** | The plot shows a steady rise, not a jump; caption rewritten to match the figure |
| Fig. 8 caption | "The layers squeeze the space (d10 0.003 → 0.0006)" | **Corrected (AI over-simplification)** | The plot shows a fall to 0.0003 at layer 11 and a rise in layer 12; caption now describes both |
| Grad-CAM per field | Rank fields by Grad-CAM share | **Refined** | Mass grows with the number of tokens (age = 5 tokens); stated as a caveat instead of a ranking |
| Fig. 3 caption | Causal explanations for two outliers (admission type, insulin) | **Removed (AI speculation)** | Not checked against data; replaced by the observation only |

### Failure patterns observed

1. **Plausible but unverified explanations.** The AI wrote causal reasons into captions that no computation supported (Fig. 3) and described a plot more cleanly than it looked (Fig. 2).
2. **Silent technical failures.** A parser that dropped one of 15 fields without an error; a tie-break rule that differed from Phase I. Both only surfaced through explicit checks.
3. **Visual encodings that mislead.** Over-plotting decided which class was visible; the first two fixes swung from hiding to exaggerating the rare classes.
4. **Methods that are "correct" but uninformative.** Probes for input fields are trivially perfect; the AI had to re-frame the question (how strongly, not whether).

### Best practices that worked

* Reproduce the model and the Phase I numbers **before** explaining it.
* Build checks into every technique (Shapley efficiency, IG completeness, Phase I cross-check of the walk-through).
* Look at every figure; write captions only from printed numbers.
* Use the same explanation game for Model A and Model B so the comparison is fair.

## Pair A (human-only)

No generative AI except code completion and documentation lookup (to be filled in by Pair A).
