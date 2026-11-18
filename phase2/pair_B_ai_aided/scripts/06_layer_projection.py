"""Step 6 (technique: projecting hidden layers, Lecture 03): how does the space change inside Model B?

UMAP of the mean-pooled embedding after selected layers (0 = token embeddings ... 12 = Model B's space),
for one fixed sample of stays. Each layer's UMAP is initialised with the previous layer's layout, so
the panels stay comparable from left to right. Colour = specialty group, ER + unknown specialty,
and the true next pathway (the outcome Model B votes on).

Writes (gitignored) outputs/layer_umap_B.parquet and figures/fig_layer_projection_B.png.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from common import CACHE, FIG, LIGHT, OUT, PATHWAY_COLORS, QUALITATIVE, VALUE_C, apply_style, l2n, load_cohort

apply_style()

LAYERS = [0, 2, 4, 6, 8, 10, 12]
df = load_cohort()
layers = np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")
rng = np.random.default_rng(4)
sub = np.sort(rng.choice(len(df), 6000, replace=False))
d = df.iloc[sub].reset_index(drop=True)

cache = OUT / "layer_umap_B.parquet"
if cache.exists():
    saved = pd.read_parquet(cache)
    assert (saved["encounter_id"].values == d["encounter_id"].values).all()
    coords = {l: saved[[f"L{l}_x", f"L{l}_y"]].to_numpy() for l in LAYERS}
else:
    coords, init = {}, "spectral"
    for layer in LAYERS:
        x = l2n(np.asarray(layers[sub, layer, :], dtype=np.float32))
        x = x - x.mean(0)                    # remove the shared direction so local structure is visible
        xy = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", init=init, random_state=0).fit_transform(x)
        coords[layer], init = xy, xy
        print("layer", layer, "done")
    pd.DataFrame({f"L{l}_{a}": coords[l][:, k] for l in LAYERS for k, a in enumerate("xy")}
                 ).assign(encounter_id=d["encounter_id"].values).to_parquet(cache, index=False)

spec = d["medical_specialty"].where(d["medical_specialty"].isin(
    d["medical_specialty"].value_counts().index[:5]), "other")
er_unknown = d["admission_source"].eq("Emergency Room") & d["medical_specialty"].eq("Unknown")
ROWS = [
    ("specialty (top 5)", spec),
    ("ER + unknown specialty", er_unknown.map({True: "ER + unknown", False: "other"})),
    ("next pathway (outcome)", d["true_next_pathway"]),
]
PALETTE = QUALITATIVE
PATHWAY_COL = PATHWAY_COLORS
fig, axes = plt.subplots(len(ROWS), len(LAYERS), figsize=(2.6 * len(LAYERS), 2.6 * len(ROWS)))
for r, (title, lab) in enumerate(ROWS):
    cats = [c for c in lab.value_counts().index if c != "other"] + (["other"] if (lab == "other").any() else [])
    colors = {c: (PALETTE[i] if c != "other" else LIGHT) for i, c in enumerate(cats)}
    if title.startswith("next pathway"):
        colors = PATHWAY_COL
    # draw the largest group first so small groups (rare pathways) stay visible on top
    draw = sorted(cats, key=lambda c: -(lab == c).sum())
    shuffle = np.random.default_rng(0).permutation(len(lab))
    for c_i, layer in enumerate(LAYERS):
        ax = axes[r, c_i]
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if c_i == 0:
            ax.set_ylabel(title, fontsize=10)
        if title.startswith("next pathway"):
            # random draw order: in dense clusters the visible colour mix then matches the real proportions
            ax.scatter(*coords[layer][shuffle].T, s=1.5, c=lab.iloc[shuffle].map(colors).tolist(), rasterized=True)
            for c in draw:
                ax.scatter([], [], s=1.5, color=colors[c], label=f"{c} ({(lab == c).mean():.0%})")
            continue
        for c in draw:
            m = (lab == c).to_numpy()
            ax.scatter(*coords[layer][m].T, s=1.5, color=colors[c], label=c, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"after layer {layer}" if layer else "token embeddings", fontsize=10)
        if c_i == 0:
            ax.set_ylabel(title, fontsize=10)
    axes[r, -1].legend(markerscale=6, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
fig.suptitle("Inside Model B: the stay space after each layer (UMAP, n = 6,000 stays)", y=1.0)
plt.tight_layout()
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "fig_layer_projection_B.png", dpi=150, bbox_inches="tight")
print("saved", FIG / "fig_layer_projection_B.png")
