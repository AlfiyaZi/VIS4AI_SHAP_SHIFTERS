# Course notes (AI and Visualization WS26)

Summary of the lecture and lab material relevant to the assignment. Slides themselves are on Moodle.

## Organization

- Lecture (VL, 1.5 ECTS) Thursday 10:15–11:45, HS 5; lab (UE, 1.5 ECTS) Thursday/Friday, S3 055.
- Group assignment, 4 students. Attendance is compulsory only at the final presentation (on-site or Zoom with video and full name).
- Grading key: <50 → 5, <63 → 4, <75 → 3, <88 → 2, ≥88 → 1.
- Contact: Moodle forum, or ai-and-vis-course@jku.at for individual requests.
- No deadline extensions.

## Assignment requirements (Lab 02)

**Phase I – How does the model behave?** Keep the black box closed; look only at inputs and outputs.
- Visualization must be an *investigative tool*, not a chart of a performance metric.
- Patterns to look for: errors/confidence changes, biases, hallucination, safety violations, shifts in output characteristics.
- Input data: official test set, established benchmark, synthetic data (edge cases, perturbations, counterfactuals), or the training data as a source of hypotheses (e.g. under-represented cases).

**Phase II – What did the model learn?** Open the black box; explore internal representations to *explain* Phase I behavior (e.g. Phase I: more mistakes on certain backgrounds → Phase II: does the model rely on background features?). A gallery of saliency maps alone is not enough.

**AI4VIS.** Explore how generative AI supports and influences visualization design and analysis. Record important prompts and which suggestions were accepted, rejected or refined, and why. Report failure patterns and emerging best practices.

**Repository.** Public GitHub repo with `team-info.json` and `workload.json` in the root.

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
