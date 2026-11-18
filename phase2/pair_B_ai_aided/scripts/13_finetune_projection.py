"""Step 13: how the stay representations change across fine-tuning checkpoints (assignment: "visualize how
input representations change across checkpoints").

Uses the evaluation embeddings saved by 12_finetune.py (same 6,000 stays at every checkpoint). UMAP per
checkpoint, each initialised with the previous layout so panels stay comparable; colour = specialty
and next pathway (drawn in random order so the colour mix reflects real proportions).

Writes figures/fig_finetune_projection_B.png.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from common import CACHE, FIG, LIGHT, OUT, PATHWAY_COLORS, QUALITATIVE, apply_style, l2n, load_cohort

apply_style()
STEPS = [0, 150, 300, 450, 600, 761]
df = load_cohort()
rng = np.random.default_rng(11)                       # identical split / sample as 12_finetune.py
pats = df["patient_nbr"].unique()
test_p = set(rng.choice(pats, int(0.2 * len(pats)), replace=False))
is_test = df["patient_nbr"].isin(test_p).to_numpy()
train_idx, test_idx = np.where(~is_test)[0], np.where(is_test)[0]
eval_index = rng.choice(train_idx, 4800, replace=False)
eval_query = rng.choice(test_idx, 1200, replace=False)
sel = np.concatenate([eval_index, eval_query])
d = df.iloc[sel].reset_index(drop=True)

coords, init = {}, "spectral"
for s in STEPS:
    x = l2n(np.load(CACHE / f"ft_emb_step{s}.npy").astype(np.float32))
    x = x - x.mean(0)
    coords[s] = init = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", init=init,
                                 random_state=0).fit_transform(x)
    print("step", s, "done")

spec = d["medical_specialty"].where(d["medical_specialty"].isin(d["medical_specialty"].value_counts().index[:5]), "other")
metrics = pd.read_csv(OUT / "finetune_metrics.csv").set_index("step")
shuffle = np.random.default_rng(0).permutation(len(d))
fig, axes = plt.subplots(2, len(STEPS), figsize=(3 * len(STEPS), 6.4))
for c, s in enumerate(STEPS):
    for r, (lab, palette) in enumerate([(spec, None), (d["true_next_pathway"], PATHWAY_COLORS)]):
        ax = axes[r, c]
        if palette is None:
            cats = [k for k in lab.value_counts().index if k != "other"]
            palette = {k: QUALITATIVE[i] for i, k in enumerate(cats)} | {"other": LIGHT}
        ax.scatter(*coords[s][shuffle].T, s=1.2, c=lab.iloc[shuffle].map(palette).tolist(), rasterized=True)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if r == 0:
            m = metrics.loc[s]
            ax.set_title(f"{'Model B (step 0)' if s == 0 else f'step {s}'}\nspecialty {m.var_specialty:.0%} · "
                         f"outcome {m.var_outcome:.1%}", fontsize=10)
        if c == len(STEPS) - 1:
            for k, col in palette.items():
                ax.scatter([], [], s=12, color=col, label=k)
            ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5), markerscale=2)
    axes[0, 0].set_ylabel("specialty (top 5)"); axes[1, 0].set_ylabel("next pathway (outcome)")
fig.suptitle("Fine-tuning on the next pathway: the stay space across checkpoints (6,000 evaluation stays; "
             "titles = share of the space explained)", x=0.01, ha="left")
plt.tight_layout()
fig.savefig(FIG / "fig_finetune_projection_B.png", dpi=150, bbox_inches="tight")
print("saved", FIG / "fig_finetune_projection_B.png")
