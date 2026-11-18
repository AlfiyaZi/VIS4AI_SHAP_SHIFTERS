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
| `scripts/06_layer_projection.py` | UMAP of the space after each layer | `figures/fig_layer_projection_B.png` |
| `scripts/07_walkthrough.py` | One stay classified step by step, before/after the HbA1c edit | `figures/fig_walkthrough_B.png` |
| `scripts/08_layer_knn.py` | Model B's classifier (KNN + vote) run on every layer | `outputs/layer_knn_B.csv` |
| `scripts/09_layer_gradcam.py` | Grad-CAM for text: tokens x layers | `figures/fig_layer_gradcam_B.png` |
| `scripts/10_ice_pdp.py` | ICE / partial dependence for age and HbA1c | `figures/fig_ice_pdp_B.png` |
| `scripts/11_lime.py` | LIME for text (words incl. template), fidelity and stability checks | `figures/fig_lime_B.png` |
| `scripts/12_finetune.py` | Fine-tunes a copy (Model B') on the next pathway, measures every checkpoint (~1 h on CPU; `--plot-only` re-draws) | `figures/fig_finetune_B.png` |
| `scripts/13_finetune_projection.py` | UMAP of the evaluation stays per checkpoint | `figures/fig_finetune_projection_B.png` |

All figures use Pair A's Phase I colours and look (`apply_style()` and the palettes in `scripts/common.py`).

The notebook only reads these outputs.
