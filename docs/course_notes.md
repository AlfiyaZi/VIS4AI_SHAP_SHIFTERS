# Course notes (AI and Visualization WS26)

Summary of the lecture and lab material relevant to the assignment. Slides themselves are on Moodle.

## Organization (from the Moodle course page, course 364.036 ff., 2026W)

- Lecture (VL, 1.5 ECTS) weekly, Thursday 10:15–11:45, hybrid (HS 6 + online). Written on-site Moodle exam (Linz, Vienna or Bregenz).
- Lab (UE, 1.5 ECTS) held irregularly, hybrid (S3 055 + online); you may join any slot.
- Group assignment, 4 students split into two pairs. Lab attendance is optional, except that you must attend **at least one** final presentation session.
- Grading key: <50 → 5, <63 → 4, <75 → 3, <88 → 2, ≥88 → 1.
- Contact: Moodle forum, or ai-and-vis-course@jku.at (not individual team members).
- No deadline extensions.

Lectures: 01 Introduction · 02 Fundamentals & Explaining Algorithms · 03 Explaining Through Projections · 04 Visual Explanation Techniques I · 05 Visual Explanation Techniques II · 06 GenAI & VIS · 07 Q&A, Recap · 08 Exam.

### Key dates

| Date | Event |
|---|---|
| Wed 21.10.2026 23:59 | Group formation |
| Wed 28.10.2026 23:59 | Task selection (task, model, input dataset) |
| Thu 29.10.2026 13:45 | Lab: Practical Examples 2 |
| Thu 05.11 / Fri 06.11.2026 | Feedback 1 (book a 10-min slot, S3 055) |
| Wed 11.11.2026 23:59 | **Midpoint submission (end of Phase I)** |
| Thu 12.11.2026 13:45 | Lab: Practical Examples 3 |
| 26.11 – 04.12.2026 | Feedback 2 (six slots) |
| Wed 09.12.2026 23:59 | **Final submission (end of Phase II)** |
| 10.12 / 11.12 / 17.12.2026 | Live presentation sessions (attend at least one) |
| Thu 17.12.2026 | Written exam |
| Thu 18.02.2027 | Retry exam |

Note: the Lecture 01 slides list older exam dates (11.12.2025 / 19.02.2026); the Moodle dates above are authoritative.

## Assignment requirements (official Moodle description)

**Task and model.** A model trained on a specific task: pre-trained, trained or fine-tuned; discriminative or generative. One model in depth or several compared. All four members choose together.

**Phase I – How does the model behave?** Black box: inputs and outputs only.
- Input data: official test set, established benchmark, or a synthetic dataset for edge cases / hypotheses. Training data may be explored as a source of hypotheses.
- The dataset must be **large enough that reported patterns are systematic**, not cherry-picked.
- Behavior = error patterns, biases, hallucinations, shifts in diversity / confidence / style, …
- **Not a performance report**; a table of benchmark scores is not sufficient. Visualizations must expose behavior that aggregate metrics hide.
- **Identifying a pattern is not enough; understand it in depth.** E.g. for an error cluster in a projection: what distinguishes it, why is it dense or sparse, does it have sub-clusters with different behavior?

**Phase II – What did the model learn?** Open the black box (weights, representations, attention, gradients), ideally continuing from Phase I findings.
- Compare how input groups are organized in representation space, across layers, or across training checkpoints.
- Techniques: saliency maps, attention maps, LIME, SHAP, KNN, others. **Each pair must use at least two techniques and compare or combine their insights.**
- Goal: link regions of representation space to behaviors or input groups (hypothesis-driven, e.g. compare saliency of correct vs. incorrect examples), not visualize arbitrary examples.

**AI4Vis.**
- Two pairs: **AI-aided** (GenAI actively used to brainstorm, interpret, suggest/generate visualizations and code) and **human-only** (own reasoning and code; AI only for code completion and documentation lookup). **Pairs switch roles after Phase I.**
- Both pairs work on the same model and task. At the start of each phase, agree on a common direction: main questions, data, initial visualization, which Phase I findings to pursue in Phase II.
- Then investigate separately; communicate, but do not make the pipelines identical, so the differences can be discussed.
- At the end of each phase compare: insights one pair missed, visualization choices, breadth vs. depth, speed vs. verification overhead, confidence in conclusions and the evidence behind it.
- The AI-aided pair is expected to build **richer interaction or broader analysis**, and to keep a record: important prompts, how suggestions affected decisions, what was accepted / modified / refined and why, plus failure patterns and best practices.

**Deliverables.**
- *Midpoint (11.11):* code from both pairs; AI interaction record of the Phase I AI-aided pair; **progress video ≤ 5 min** (problem, what each pair did and found, key visualizations, Phase II plan). Overtime costs points.
- *Final (09.12):* complete code from both pairs; AI records from both phases; **10-min recorded presentation + slide deck** (problem and motivation, both pairs' methods and findings for Phase I then II, AI4Vis reflection, conclusion). A random subset of groups presents live instead; the whole group or one member may present.

**Grading.** Open-ended: ambition and originality of the problem count, but analysis quality matters most. A challenging problem done well can earn more than basic goals reached on a simple problem.

**Repository.** Public GitHub repo; submit its URL. `team-info.json` and `workload.json` (templates on Moodle) in the root, filled in before submission.

## Phase I investigative tools (Lab 02, practical examples)

- **Scatter plots:** each variable gets its own visual channel (x, y, small multiples, shape, color). Channels are limited, so for high-dimensional data use a projection and drill down → understand → summarize → annotate (Projection Space Explorer).
- **Rich outputs:** project the outputs instead of (or as well as) the inputs; use a separate encoder (e.g. SBERT) for items without attributes.
- **Derived interpretable attributes:** NLP features, captioning/object detection, LLM-as-a-judge criteria, diversity or alignment metrics. Map them onto the projection or project with them.
- **Sequences:** connect consecutive states into paths (Time Curves, Bach et al. 2015; Projection Path Explorer, Hinterreiter et al. 2021).
- **Confusion matrices:** equal accuracy can hide different error structure. Reorder rows/columns by error similarity to reveal blocks (Alsallakh et al. 2017). See also ConfusionFlow, InstanceFlow, Abstraction Alignment (Boggust et al. 2025).
- Inspiration: VISxAI workshop (https://visxai.io), TrustMLVis survey (https://trustmlvis.lnu.se).

## Lecture 2 – Fundamentals and explaining algorithms

- Humans see patterns algorithms miss (Anscombe's quartet); algorithms reveal patterns humans miss.
- Visualization (Munzner): visual representations that help people carry out tasks more effectively, "not pretty pictures". Human in the loop is needed for ill-defined problems such as model building and validation.
- XAI should be designed for its audience (Miller 2019): ML experts, data scientists, decision-makers (doctors), affected people (patients), regulators.
- Storytelling: author- vs. reader-driven (martini glass, interactive slideshow, drill-down story).
- Gestalt grouping principles; connection is the strongest and can override others.
- "Eyes beat memory": prefer static small multiples or key frames over animation.

## Lecture 3 – Explaining through projections

- PCA: cheap, can project new points, but linear.
- t-SNE: strong local clusters, but global structure lost, non-deterministic, hyperparameter-sensitive. From *How to Use t-SNE Effectively*: hyperparameters matter, cluster sizes and inter-cluster distances may mean nothing, noise can look structured.
- UMAP: faster than t-SNE, tunable local/global trade-off, still non-deterministic.
- Encode categorical attributes with hue/shape and ordered ones with size/brightness. No unjustified 3D.
- Embedding trajectories: time curves, Rubik's cube solving, chess games, NN training (confusion-matrix distances), CIME / CIME4R.
- Next: intrinsic vs. post-hoc, global vs. local, model-agnostic vs. model-specific explanations.
