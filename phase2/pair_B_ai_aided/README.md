# Phase II – Pair B (AI-aided)

Pair B after the role switch. Record AI use in [../../ai-interactions/phase2_ai_log.md](../../ai-interactions/phase2_ai_log.md). Use at least two explanation techniques and compare or combine them.

## Model B: what did it learn? (`pairB_phase2_modelB.ipynb`)

Run order (Python 3.11 with `torch`, `transformers`, `scikit-learn`, `umap-learn`; CPU is fine):

| Script | Technique | Output (gitignored) |
|---|---|---|
| `scripts/01_embed_layers.py` | Re-embed all stays, keep all 13 layers; checks Phase I is reproduced (~7 min) | `shared/data_cache/phase2_model_b_layers.npy` |
| `scripts/02_geometry_edits.py` | Embedding geometry; edit shift vs. neighbourhood radius | `outputs/anisotropy.csv`, `outputs/edit_shift.parquet` |
| `scripts/03_field_shapley.py` | Shapley values over the 15 fields (pairwise similarity game) | `outputs/field_shapley_B.parquet` |
| `scripts/04_token_saliency.py` | Integrated Gradients over tokens (template vs. value) | `outputs/token_ig_B.parquet` |
| `scripts/05_layer_probes.py`, `05b_field_variance.py` | Concept probes per layer; variance share per field | `outputs/layer_probes_B.csv`, `outputs/field_variance_B.csv` |
| `scripts/05c_outcome_probe.py` | Stronger outcome probe per layer, all train stays (corrects step 5 for the outcome; needs step 14) | `outputs/outcome_probe_B.csv` |
| `scripts/06_layer_projection.py` | UMAP of the space after each layer | `figures/fig_layer_projection_B.png` |
| `scripts/07_walkthrough.py` | One stay classified step by step, before/after the HbA1c edit | `figures/fig_walkthrough_B.png` |
| `scripts/08_layer_knn.py` | Model B's classifier (KNN + vote) run on every layer | `outputs/layer_knn_B.csv` |
| `scripts/09_layer_gradcam.py` | Grad-CAM for text: tokens x layers | `figures/fig_layer_gradcam_B.png` |
| `scripts/10_ice_pdp.py` | ICE / partial dependence for age and HbA1c | `figures/fig_ice_pdp_B.png` |
| `scripts/11_lime.py` | LIME for text (words incl. template), fidelity and stability checks | `figures/fig_lime_B.png` |
| `scripts/12_finetune.py` | Fine-tunes a copy (Model B') on the next pathway, measures every checkpoint (~1 h on CPU; `--plot-only` re-draws) | `figures/fig_finetune_B.png` |
| `scripts/13_finetune_projection.py` | UMAP of the evaluation stays per checkpoint | `figures/fig_finetune_projection_B.png` |

## Model A vs Model B (`pairB_phase2_modelA_vs_B.ipynb`)

The same techniques applied to Model A (z-scored numbers + one-hot categories, cosine KNN), on the same pairs and stays as for Model B. Runs in about 2 minutes on CPU; needs the Model B outputs above.

| Script | Technique | Output (gitignored unless a figure) |
|---|---|---|
| `scripts/14_model_a_geometry.py` | Reproduce Model A; geometry; exact per-field decomposition of the cosine; variance share; edit shift vs. radius | `outputs/model_a_x.npy`, `geometry_A.json`, `model_a_decomposition.csv`, `field_variance_A.csv`, `edit_shift_A.parquet` |
| `scripts/15_model_a_shapley.py` | Same Shapley game as step 3, same 300 pairs | `outputs/field_shapley_A.parquet` |
| `scripts/16_model_a_ice.py` | ICE / PDP, same 300 stays as step 10 | `figures/fig_ice_pdp_AB.png` |
| `scripts/17_supervised_reference.py` | Classifiers trained on the outcome (reference); permutation importance per field | `outputs/supervised_reference.csv`, `field_importance_outcome.csv` |
| `scripts/18_compare_ab.py` | Weights vs. outcome, the two spaces, UMAP of the same stays, groups | `figures/fig_compare_*_AB.png` |

## Interactive dashboard (`dashboard/`)

Dash app with five linked views around one selected stay: (1) map of all 30,248 stays in Model A's or Model B's space (layers 0 / 6 / 12), coloured by outcome, specialty, age, who is right, or fragility; (2) the stay's record, its neighbours' shared fields and their vote in both models; (3) why they are neighbours: Model A's exact per-field split, and a live field-by-field comparison of Model A and Model B for one neighbour; (4) what-if: edit fields and re-run both models (the stay moves on the map); (5) what each model weighs vs. what predicts the outcome, and groups.

```bash
pip install dash plotly                    # in the Phase II environment
python dashboard/precompute.py             # once, ~5 min: Model B maps for all stays (gitignored)
python dashboard/app.py                    # http://127.0.0.1:8051
python -m pytest dashboard/test_dashboard.py -q   # live models reproduce Phase I; all views render
```

All figures use Pair A's Phase I colours and look (`apply_style()` and the palettes in `scripts/common.py`).

The notebook only reads these outputs.
