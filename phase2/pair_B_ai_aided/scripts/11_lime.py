"""Step 11 (technique: LIME for text, Lecture 04 / Molnar ch. 14): which WORDS does Model B's
classification depend on locally?

LIME: switch random subsets of the record's words off (remove them from the text), re-run Model B,
and fit a weighted linear surrogate on the on/off pattern; kernel = exp(-d^2 / width^2) with d the
cosine distance between the on/off vector and the full text (as in the lime package for text).
Features = every whitespace word of the record, INCLUDING template words ("Specialty:", "Labs:"),
so LIME can test the template finding of Integrated Gradients / Grad-CAM model-agnostically.

Two targets per perturbed text:
  vote   - share of the 10 nearest other-patient stays (real index, layer 12) that carry the
           pathway Model B predicted for the original text (= the classification)
  sim1   - cosine similarity to the stay's original 1st neighbour (comparable with IG / Grad-CAM)

LIME's known weaknesses (Lecture 04) are measured: fidelity = weighted R^2 of the surrogate,
stability = Spearman correlation of the word weights between two runs with different seeds.

Writes (gitignored) outputs/lime_B.parquet, outputs/lime_fit_B.csv and figures/fig_lime_B.png.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import CACHE, FIG, K, NAVY, OUT, TEMPLATE_C, VALUE_C, BertEncoder, apply_style, field_spans, l2n, load_cohort

apply_style()

p = argparse.ArgumentParser()
p.add_argument("--stays", type=int, default=40)
p.add_argument("--samples", type=int, default=500)
p.add_argument("--stability", type=int, default=8, help="stays re-run with a second seed")
args = p.parse_args()

df = load_cohort()
pos = {e: i for i, e in enumerate(df["encounter_id"])}
y = df["true_next_pathway"].to_numpy()
patient = df["patient_nbr"].to_numpy()
X = l2n(np.asarray(np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")[:, -1, :], dtype=np.float32))
enc = BertEncoder()


def words_of(text):
    """Whitespace words with (field, part): value if the word overlaps the field's value span."""
    spans, out, start = field_spans(text), [], 0
    for w in text.split(" "):
        s, e = start, start + len(w)
        start = e + 1
        field, part = "[template]", "template"
        for label, (fs, fe) in spans.items():
            if s < fe and e > fs:
                field, part = label, "value"
                break
        if part == "template":
            for label, (fs, fe) in spans.items():
                if e <= fs and s >= fs - len(label) - 2:
                    field = label
        out.append((w, field, part))
    return out


def predict(q, embs):
    d = 1.0 - embs @ X.T
    d[:, patient == patient[q]] = np.inf
    nb = np.argpartition(d, K, axis=1)[:, :K]
    return y[nb]


def vote_pred(labels):
    vals, cnt = np.unique(labels, return_counts=True)
    tied = set(vals[cnt == cnt.max()])
    return next(v for v in labels if v in tied)


def lime_one(q, seed, width=0.25):
    text = df["model_text"].iloc[q]
    words = words_of(text)
    n = len(words)
    rng = np.random.default_rng(seed)
    Z = np.ones((args.samples, n), dtype=np.int8)
    for i in range(1, args.samples):                          # row 0 = original text
        k = rng.integers(1, n)                                # LIME: remove a random number of words
        Z[i, rng.choice(n, k, replace=False)] = 0
    texts = [" ".join(w for (w, _, _), z in zip(words, row) if z) for row in Z]
    E = l2n(enc.encode(texts))
    orig_nb = np.argsort(np.where(patient == patient[q], np.inf, 1.0 - X @ X[q]), kind="stable")[:K]
    pred = vote_pred(y[orig_nb])
    targets = {"vote": (predict(q, E) == pred).mean(1), "sim1": E @ X[orig_nb[0]]}
    dist = 1.0 - Z.sum(1) / np.sqrt(Z.sum(1) * n)             # cosine distance to the all-ones vector
    w = np.exp(-(dist ** 2) / width ** 2)
    rows, fits = [], []
    A = np.hstack([np.ones((len(Z), 1)), Z.astype(float)])
    for tname, t in targets.items():
        Wsq = np.sqrt(w)[:, None]
        coef = np.linalg.lstsq(A * Wsq, t * Wsq[:, 0], rcond=None)[0]   # weighted least squares
        resid = t - A @ coef
        r2 = 1 - (w * resid ** 2).sum() / (w * (t - np.average(t, weights=w)) ** 2).sum()
        fits.append({"q": int(df["encounter_id"].iloc[q]), "target": tname, "seed": seed, "r2": float(r2),
                     "pred": pred, "true": y[q], "target_std": float(t.std())})
        for (word, field, part), c in zip(words, coef[1:]):
            rows.append({"q": int(df["encounter_id"].iloc[q]), "target": tname, "seed": seed,
                         "word": word, "field": field, "part": part, "weight": float(c)})
    return rows, fits


rng = np.random.default_rng(7)
sample = list(rng.choice(len(df), args.stays, replace=False))
walk = df.index[df["admission_source"].eq("Emergency Room") & df["medical_specialty"].eq("Unknown")
                & df["A1Cresult"].eq("None") & df["model_b_flips_hba1c_elevation"].eq(10)
                & df["model_b_pred_changed_hba1c_elevation"]][0]
sample = [int(walk)] + [int(s) for s in sample if s != walk]
rows, fits = [], []
cached = (OUT / "lime_B.parquet").exists()
for i, q in enumerate([] if cached else sample):
    r, f = lime_one(q, seed=100 + i)
    rows += r; fits += f
    if i < args.stability:                                    # LIME instability check (Lecture 04)
        r, f = lime_one(q, seed=900 + i)
        rows += r; fits += f
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{len(sample)} stays")
if cached:
    res, fit = pd.read_parquet(OUT / "lime_B.parquet"), pd.read_csv(OUT / "lime_fit_B.csv")
else:
    res, fit = pd.DataFrame(rows), pd.DataFrame(fits)
    res.to_parquet(OUT / "lime_B.parquet", index=False)
    fit.to_csv(OUT / "lime_fit_B.csv", index=False)

first = res[res.seed < 900]
print("surrogate fidelity (weighted R^2), median:", fit[fit.seed < 900].groupby("target").r2.median().round(3).to_dict())
share = (first.assign(a=first.weight.abs()).groupby(["target", "part"]).a.sum()
         / first.assign(a=first.weight.abs()).groupby("target").a.sum()).unstack().round(3)
print("share of |LIME weight| by part:\n", share.to_string())
print("share of words by part:", first[first.target == "vote"].part.value_counts(normalize=True).round(3).to_dict())
stab = []
for (q, t), g in res.groupby(["q", "target"]):
    if g.seed.nunique() == 2:
        a, b = (g[g.seed == s].weight.to_numpy() for s in sorted(g.seed.unique()))
        stab.append({"q": q, "target": t, "spearman": spearmanr(a, b)[0]})
stab = pd.DataFrame(stab)
stab.to_csv(OUT / "lime_stability_B.csv", index=False)
print("stability (Spearman of word weights, two seeds), median:", stab.groupby("target").spearman.median().round(3).to_dict())
print("top fields by mean |weight| (vote):",
      first[first.target == "vote"].assign(a=lambda t: t.weight.abs()).groupby("field").a.mean()
      .sort_values(ascending=False).head(8).round(4).to_dict())

# ---- figure: walk-through stay, top words for both targets + summary --------------------------
eid = int(df["encounter_id"].iloc[walk])
fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
for ax, t, title in [(axes[0], "vote", "vote share for the predicted pathway"),
                     (axes[1], "sim1", "similarity to the 1st neighbour")]:
    g = first[(first.q == eid) & (first.target == t)].copy()
    g = g.reindex(g.weight.abs().sort_values(ascending=False).index).head(15)[::-1]
    ax.barh(range(len(g)), g.weight, color=np.where(g.part == "value", VALUE_C, TEMPLATE_C))
    ax.set_yticks(range(len(g)), [f"{w}  ({f})" for w, f in zip(g.word, g.field)], fontsize=8)
    ax.axvline(0, color=NAVY, lw=0.8)
    r2 = fit[(fit.q == eid) & (fit.target == t) & (fit.seed < 900)].r2.iloc[0]
    ax.set_title(f"stay {eid}: {title}\n(top 15 words, surrogate R² = {r2:.2f}; red = value, grey = template)", fontsize=9)
lab = {"vote": "vote share\n(classification)", "sim1": "similarity to\n1st neighbour"}
share[["template", "value"]].rename(index=lab).plot.barh(ax=axes[2], stacked=True, color=[TEMPLATE_C, VALUE_C], width=0.6)
axes[2].set_ylabel(""); axes[2].legend(["template words", "value words"], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
for k, t in enumerate(share.rename(index=lab).index):
    f = fit[(fit.target == {v: k_ for k_, v in lab.items()}[t]) & (fit.seed < 900)].r2.median()
    axes[2].text(1.02, k, f"R² {f:.2f}", va="center", fontsize=9, transform=axes[2].get_yaxis_transform())
axes[2].set_title(f"all {first.q.nunique()} stays: share of |LIME weight|\non template vs. value words", fontsize=9)
axes[2].set_xlim(0, 1)
plt.tight_layout()
fig.savefig(FIG / "fig_lime_B.png", dpi=150, bbox_inches="tight")
print("saved", FIG / "fig_lime_B.png")
