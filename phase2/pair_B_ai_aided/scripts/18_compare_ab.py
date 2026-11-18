"""Step 18: Model A vs Model B, side by side (the Phase I overlap, now explained from the inside).

Fig. AB-1  what each model weighs vs what the outcome depends on (per field):
           Shapley mismatch cost on the same random pairs (steps 3 / 15), normalised to the largest,
           next to permutation importance for the outcome (step 17).
Fig. AB-2  the two spaces: variance share per field (steps 5b / 14), edit shift / radius and flips.
Fig. AB-3  the same 6,000 stays projected with UMAP in both spaces, coloured by specialty, age, outcome.
Fig. AB-4  groups: where the models agree / disagree, crowded ER + unknown-specialty stays,
           fragile (>= 8 flips) vs stable stays, correct vs wrong.

Writes figures/fig_compare_*_AB.png and (gitignored) outputs/compare_groups_AB.csv, compare_umap_A.parquet.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from common import (FIELDS, FIG, LIGHT, MODEL_COLORS, MUTED, NAVY, OUT, PATHWAY_COLORS, QUALITATIVE,
                    apply_style, l2n, load_cohort)

apply_style()
CA, CB = MODEL_COLORS["model_a"], MODEL_COLORS["model_b"]
df = load_cohort()
labels = list(FIELDS)
EDITS = {"sex_swap": "Sex swap", "age_shift": "Age +20 y", "hba1c_elevation": "HbA1c -> >8"}

# ---- Fig AB-1: weights vs outcome -------------------------------------------------------------
cost = {}
for m in "AB":
    s = pd.read_parquet(OUT / f"field_shapley_{m}.parquet")
    s = s[s.differs & (s.kind == "random")]
    cost[m] = -s.groupby("field").phi.mean()
w = pd.DataFrame({"A": cost["A"] / cost["A"].max(), "B": cost["B"] / cost["B"].max()}).reindex(labels)
imp = pd.read_csv(OUT / "field_importance_outcome.csv").set_index("field").reindex(labels)
w["outcome"] = imp.logloss_increase.clip(lower=0) / imp.logloss_increase.max()
w = w.sort_values("outcome")
rho = {m: float(w[m].corr(w.outcome, method="spearman")) for m in "AB"}
print("Spearman(field weight, outcome importance):", {k: round(v, 2) for k, v in rho.items()})
print("max/min mismatch cost ratio:", {m: round(float(cost[m].max() / cost[m].min()), 1) for m in "AB"})

fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
yy = np.arange(len(w))
for ax, col, color, title in [
        (axes[0], "A", CA, f"Model A: similarity lost when the field differs\n(Spearman with outcome: {rho['A']:.2f})"),
        (axes[1], "B", CB, f"Model B: similarity lost when the field differs\n(Spearman with outcome: {rho['B']:.2f})"),
        (axes[2], "outcome", NAVY, "What the outcome depends on\n(permutation importance, trained model)")]:
    ax.barh(yy, w[col], color=color)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(0, 1.08); ax.set_xlabel("relative to the largest field")
axes[0].set_yticks(yy, w.index)
fig.suptitle("What each model weighs vs what predicts the next pathway (Shapley, same 150 random pairs for both models)",
             x=0.01, ha="left")
plt.tight_layout()
fig.savefig(FIG / "fig_compare_weights_AB.png", dpi=150, bbox_inches="tight")

# ---- Fig AB-2: spaces -----------------------------------------------------------------------
va = pd.read_csv(OUT / "field_variance_A.csv").set_index("field").r2
vb = pd.read_csv(OUT / "field_variance_B.csv").query("layer == 12").set_index("field").r2
geo_a = json.load(open(OUT / "geometry_A.json"))
ea, eb = pd.read_parquet(OUT / "edit_shift_A.parquet"), pd.read_parquet(OUT / "edit_shift.parquet")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
order = vb.sort_values().index
yy = np.arange(len(order))
axes[0].barh(yy - 0.2, va.reindex(order), 0.4, color=CA, label="Model A")
axes[0].barh(yy + 0.2, vb.reindex(order), 0.4, color=CB, label="Model B (layer 12)")
axes[0].set_yticks(yy, order); axes[0].legend(loc="lower right")
axes[0].set_title("Share of the space explained by each field alone (R²)", fontsize=10.5)
xx = np.arange(len(EDITS))
ra = [np.nanmedian(ea[f"ratio_{e}"]) for e in EDITS]
rb = [np.nanmedian(eb[f"ratio_{e}"]) for e in EDITS]
axes[1].bar(xx - 0.2, ra, 0.4, color=CA, label="Model A"); axes[1].bar(xx + 0.2, rb, 0.4, color=CB, label="Model B")
axes[1].axhline(1, color=MUTED, ls=":", lw=1)
axes[1].set_xticks(xx, EDITS.values()); axes[1].set_title("Edit shift / radius of the 10 neighbours (median)", fontsize=10.5)
axes[1].legend()
fa = [df[f"model_a_flips_{e}"].astype(float).mean() for e in EDITS]
fb = [df[f"model_b_flips_{e}"].astype(float).mean() for e in EDITS]
axes[2].bar(xx - 0.2, fa, 0.4, color=CA, label="Model A"); axes[2].bar(xx + 0.2, fb, 0.4, color=CB, label="Model B")
axes[2].set_xticks(xx, EDITS.values()); axes[2].set_ylim(0, 10)
axes[2].set_title("Neighbours replaced by the edit (of 10, Phase I)", fontsize=10.5)
for ax, vals in [(axes[1], (ra, rb)), (axes[2], (fa, fb))]:
    for k, v in enumerate(vals[0]):
        ax.text(k - 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for k, v in enumerate(vals[1]):
        ax.text(k + 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
fig.suptitle(f"Two different spaces: Model A random-pair cosine {geo_a['random_pair_cosine']:.2f}, "
             f"Model B 0.995 (one narrow cone)", x=0.01, ha="left")
plt.tight_layout()
fig.savefig(FIG / "fig_compare_space_AB.png", dpi=150, bbox_inches="tight")
for e, a_, b_, fa_, fb_ in zip(EDITS, ra, rb, fa, fb):
    print(f"{e:16s} ratio A {a_:.2f} B {b_:.2f} | flips A {fa_:.2f} B {fb_:.2f}")

# ---- Fig AB-3: the same stays in both spaces ------------------------------------------------
rng = np.random.default_rng(4)                                      # identical sample to 06_layer_projection.py
sub = np.sort(rng.choice(len(df), 6000, replace=False))
d = df.iloc[sub].reset_index(drop=True)
ub = pd.read_parquet(OUT / "layer_umap_B.parquet")
assert (ub.encounter_id.values == d.encounter_id.values).all()
cache = OUT / "compare_umap_A.parquet"
if cache.exists():
    ua = pd.read_parquet(cache)[["x", "y"]].to_numpy()
else:
    xa = np.load(OUT / "model_a_x.npy")[sub]
    ua = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=0).fit_transform(xa)
    pd.DataFrame({"encounter_id": d.encounter_id, "x": ua[:, 0], "y": ua[:, 1]}).to_parquet(cache, index=False)
coords = {"Model A": ua, "Model B (layer 12)": ub[["L12_x", "L12_y"]].to_numpy()}
spec = d.medical_specialty.where(d.medical_specialty.isin(d.medical_specialty.value_counts().index[:5]), "other")
cats = [k for k in spec.value_counts().index if k != "other"]
spec_pal = {k: QUALITATIVE[i] for i, k in enumerate(cats)} | {"other": LIGHT}
shuffle = np.random.default_rng(0).permutation(len(d))
fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
for r, (name, xy) in enumerate(coords.items()):
    xy = xy[shuffle]
    dd = d.iloc[shuffle]
    axes[r, 0].scatter(*xy.T, s=1.5, c=spec.iloc[shuffle].map(spec_pal).tolist(), rasterized=True)
    sc = axes[r, 1].scatter(*xy.T, s=1.5, c=dd.age_mid, cmap="viridis", rasterized=True)
    axes[r, 2].scatter(*xy.T, s=1.5, c=dd.true_next_pathway.map(PATHWAY_COLORS).tolist(), rasterized=True)
    for c in range(3):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
    axes[r, 0].set_ylabel(name, fontsize=12, color=CA if r == 0 else CB, fontweight="bold")
for c, t in enumerate(["specialty (top 5)", "age", "next pathway (outcome)"]):
    axes[0, c].set_title(t)
for k, col in spec_pal.items():
    axes[1, 0].scatter([], [], s=14, color=col, label=k)
axes[1, 0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
for k, col in PATHWAY_COLORS.items():
    axes[1, 2].scatter([], [], s=14, color=col, label=k)
axes[1, 2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
fig.colorbar(sc, ax=axes[:, 1], fraction=0.03, label="age (bracket middle)")
fig.suptitle("The same 6,000 stays in Model A's and Model B's space (UMAP, cosine)", x=0.01, ha="left")
fig.savefig(FIG / "fig_compare_umap_AB.png", dpi=150, bbox_inches="tight")

# ---- Fig AB-4: groups ---------------------------------------------------------------------
df["group_er_unknown"] = (df.admission_source == "Emergency Room") & (df.medical_specialty == "Unknown")  # as in step 5
d10a = pd.read_parquet(OUT / "edit_shift_A.parquet").d10.to_numpy()
d10b = np.stack(df.model_b_distances.to_numpy())[:, -1]
groups = {
    "all stays": np.ones(len(df), bool),
    "ER + unknown specialty": df.group_er_unknown.to_numpy(),
    "other stays": ~df.group_er_unknown.to_numpy(),
    "both correct": (df.model_a_correct & df.model_b_correct).to_numpy(),
    "only A correct": (df.model_a_correct & ~df.model_b_correct).to_numpy(),
    "only B correct": (~df.model_a_correct & df.model_b_correct).to_numpy(),
    "both wrong": (~df.model_a_correct & ~df.model_b_correct).to_numpy(),
}
rows = []
for g, m in groups.items():
    rows.append({"group": g, "n": int(m.sum()), "share": float(m.mean()),
                 "A accuracy": float(df.model_a_correct[m].mean()), "B accuracy": float(df.model_b_correct[m].mean()),
                 "A-B neighbour overlap": float(df.ab_neighbor_overlap[m].mean()),
                 "A HbA1c flips": float(df.model_a_flips_hba1c_elevation[m].astype(float).mean()),
                 "B HbA1c flips": float(df.model_b_flips_hba1c_elevation[m].astype(float).mean()),
                 "A median d10": float(np.median(d10a[m])), "B median d10": float(np.median(d10b[m])),
                 "same prediction A,B": float((df.model_a_pred_pathway == df.model_b_pred_pathway)[m].mean())})
grp = pd.DataFrame(rows)
grp.to_csv(OUT / "compare_groups_AB.csv", index=False)
print(grp.round(3).to_string(index=False))
# fragile in one model = fragile in the other?
fa_ = df.model_a_flips_hba1c_elevation.astype(float); fb_ = df.model_b_flips_hba1c_elevation.astype(float)
ok = fa_.notna() & fb_.notna()
print("Spearman flips A vs B (HbA1c):", round(fa_[ok].corr(fb_[ok], method="spearman"), 3),
      "| fragile (>=8) A:", round((fa_[ok] >= 8).mean(), 3), "B:", round((fb_[ok] >= 8).mean(), 3),
      "both:", round(((fa_[ok] >= 8) & (fb_[ok] >= 8)).mean(), 3))
print("prediction agreement A vs B:", round((df.model_a_pred_pathway == df.model_b_pred_pathway).mean(), 3),
      "| mean neighbour overlap:", round(df.ab_neighbor_overlap.mean(), 3))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax = axes[0]
cells = grp.set_index("group").loc[["both correct", "only A correct", "only B correct", "both wrong"]]
ax.bar(range(4), cells.share, color=[NAVY, CA, CB, LIGHT])
for k, v in enumerate(cells.share):
    ax.text(k, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(4), cells.index, rotation=15); ax.set_title("Who is right? (share of 30,248 stays)", fontsize=10.5)
ax = axes[1]
ov = df.ab_neighbor_overlap.astype(float)
ax.hist(ov, bins=np.arange(-0.5, 11.5, 1) / 10 if ov.max() <= 1 else np.arange(-0.5, 11.5, 1), color=NAVY)
ax.set_title(f"Neighbours shared by A and B (mean {ov.mean():.2f})", fontsize=10.5)
ax.set_xlabel("overlap of the two neighbour sets")
ax = axes[2]
sel = grp.set_index("group").loc[["ER + unknown specialty", "other stays"]]
xx = np.arange(2)
ax.bar(xx - 0.2, sel["A HbA1c flips"], 0.4, color=CA, label="Model A")
ax.bar(xx + 0.2, sel["B HbA1c flips"], 0.4, color=CB, label="Model B")
for k, (a_, b_) in enumerate(zip(sel["A HbA1c flips"], sel["B HbA1c flips"])):
    ax.text(k - 0.2, a_, f"{a_:.1f}", ha="center", va="bottom", fontsize=9)
    ax.text(k + 0.2, b_, f"{b_:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xx, [f"{g}\n(n = {n:,})" for g, n in zip(sel.index, sel.n)]); ax.set_ylim(0, 10)
ax.set_title("HbA1c edit: neighbours replaced, by group", fontsize=10.5); ax.legend()
fig.suptitle("Model A vs Model B by group: same accuracy, different stays, different fragility", x=0.01, ha="left")
plt.tight_layout()
fig.savefig(FIG / "fig_compare_groups_AB.png", dpi=150, bbox_inches="tight")
print("saved 4 comparison figures")
