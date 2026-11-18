"""Step 10 (technique: ICE / partial dependence, Lecture 04): how does Model B's vote react when one
field is varied, stay by stay?

For 300 random stays, Age is set to each of the 10 brackets and A1C to each of its 4 values; every
variant is re-encoded, its 10 nearest other-patient stays are retrieved from the real index, and the
vote share for "Cardiovascular" (the 2nd most common pathway) is recorded. ICE = one line per stay,
PDP = mean. For comparison: the real share of Cardiovascular next stays per bracket in the data. A
model that learned clinical risk would follow the data line; one that matches wording would not.

Writes (gitignored) outputs/ice_B.parquet and figures/fig_ice_pdp_B.png.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CACHE, FIG, K, NAVY, OUT, PATHWAY_COLORS, BertEncoder, apply_style, l2n, load_cohort,
                    record_values, render)

apply_style()

AGES = [f"[{10 * i}-{10 * (i + 1)})" for i in range(10)]
A1C = ["None", "Norm", ">7", ">8"]
TARGET = "Cardiovascular"
df = load_cohort()
y = df["true_next_pathway"].to_numpy()
patient = df["patient_nbr"].to_numpy()
X = l2n(np.asarray(np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")[:, -1, :], dtype=np.float32))
rng = np.random.default_rng(6)
sample = rng.choice(len(df), 300, replace=False)

variants = []
for q in sample:
    v = record_values(df.iloc[q])
    variants += [(q, "Age", a, render({**v, "Age": a})) for a in AGES]
    variants += [(q, "A1C", a, render({**v, "A1C": a})) for a in A1C]
cache = OUT / "ice_B.parquet"
emb = [] if cache.exists() else l2n(BertEncoder().encode([t for *_, t in variants]))

rows = []
orig_nb = {q: set(np.argsort(1 - X @ X[q] + np.where(patient == patient[q], np.inf, 0), kind="stable")[:K]) for q in sample}
for (q, field, value, _), e in zip(variants, emb):
    d = 1 - X @ e
    d[patient == patient[q]] = np.inf
    nb = np.argsort(d, kind="stable")[:K]
    rows.append({"q": int(q), "field": field, "value": value, "is_original": value == df[{"Age": "age", "A1C": "A1Cresult"}[field]].iloc[q],
                 "vote_share_cardio": float((y[nb] == TARGET).mean()), "kept_neighbours": len(orig_nb[q] & set(nb))})
if cache.exists():
    res = pd.read_parquet(cache)
else:
    res = pd.DataFrame(rows)
    res.to_parquet(cache, index=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey="row")
for c, (field, values, col) in enumerate([("Age", AGES, "age"), ("A1C", A1C, "A1Cresult")]):
    sub = res[res.field == field]
    piv = sub.pivot(index="q", columns="value", values="vote_share_cardio")[values]
    ax = axes[0, c]
    for _, row in piv.sample(80, random_state=0).iterrows():
        ax.plot(range(len(values)), row.values, color=PATHWAY_COLORS[TARGET], alpha=0.12, lw=0.8)
    ax.plot(range(len(values)), piv.mean().values, "o-", color=PATHWAY_COLORS[TARGET], lw=2.5, label="Model B: PDP (mean of ICE)")
    real = df.groupby(col)["true_next_pathway"].apply(lambda s: (s == TARGET).mean()).reindex(values)
    ax.plot(range(len(values)), real.values, "s--", color=NAVY, label="real share in the data")
    ax.set_xticks(range(len(values)), values, rotation=30)
    ax.set_title(f"{field}: vote share for '{TARGET}'", fontsize=10)
    kept = sub.pivot(index="q", columns="value", values="kept_neighbours")[values]
    axes[1, c].plot(range(len(values)), kept.mean().values, "o-", color="#E63946")
    axes[1, c].set_xticks(range(len(values)), values, rotation=30)
    axes[1, c].set_title(f"{field}: original neighbours kept (of 10), mean", fontsize=10)
    print(field, "PDP:", piv.mean().round(3).to_dict())
    print(field, "real:", real.round(3).to_dict())
    print(field, "kept:", kept.mean().round(2).to_dict())
axes[0, 0].set_ylabel("vote share"); axes[1, 0].set_ylabel("neighbours kept")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("ICE / PDP: varying one field of 300 stays (thin lines = single stays)", y=1.0)
plt.tight_layout()
fig.savefig(FIG / "fig_ice_pdp_B.png", dpi=150, bbox_inches="tight")
print("saved", FIG / "fig_ice_pdp_B.png")
