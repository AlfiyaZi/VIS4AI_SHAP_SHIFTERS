"""Step 16 (Model A vs B, technique: ICE / partial dependence, Lecture 04).

Same 300 stays, same variants and same read-out as 10_ice_pdp.py, run through Model A:
Age set to each bracket, A1C to each value; vote share for Cardiovascular among the 10 nearest
other-patient stays, and how many of the original 10 neighbours survive. Plotted next to Model B.

Writes (gitignored) outputs/ice_A.parquet and figures/fig_ice_pdp_AB.png.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FIG, K, MODEL_COLORS, NAVY, OUT, ModelA, apply_style, l2n, load_cohort, record_values)

apply_style()
AGES = [f"[{10 * i}-{10 * (i + 1)})" for i in range(10)]
A1C = ["None", "Norm", ">7", ">8"]
TARGET = "Cardiovascular"
df = load_cohort()
y = df["true_next_pathway"].to_numpy()
patient = df["patient_nbr"].to_numpy()
ma = ModelA(df)
X = l2n(np.load(OUT / "model_a_x.npy"))
rng = np.random.default_rng(6)                                   # identical sample to 10_ice_pdp.py
sample = rng.choice(len(df), 300, replace=False)


def knn(e, q):
    d = 1 - X @ e
    d[patient == patient[q]] = np.inf
    return np.lexsort((np.arange(len(d)), d))[:K]               # ties by index, as in Phase I


rows = []
for q in sample:
    v = record_values(df.iloc[q])
    orig = set(knn(X[q], q))
    for field, vals, col in [("Age", AGES, "age"), ("A1C", A1C, "A1Cresult")]:
        E = l2n(ma.encode([{**v, field: a} for a in vals]))
        for a, e in zip(vals, E):
            nb = knn(e, q)
            rows.append({"q": int(q), "field": field, "value": a, "is_original": a == df[col].iloc[q],
                         "vote_share_cardio": float((y[nb] == TARGET).mean()), "kept_neighbours": len(orig & set(nb))})
res = pd.DataFrame(rows)
res.to_parquet(OUT / "ice_A.parquet", index=False)
res_b = pd.read_parquet(OUT / "ice_B.parquet")

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey="row")
for c, (field, vals, col) in enumerate([("Age", AGES, "age"), ("A1C", A1C, "A1Cresult")]):
    for model, r in [("model_a", res), ("model_b", res_b)]:
        sub = r[r.field == field]
        piv = sub.pivot(index="q", columns="value", values="vote_share_cardio")[vals]
        kept = sub.pivot(index="q", columns="value", values="kept_neighbours")[vals]
        name = "Model A" if model == "model_a" else "Model B"
        axes[0, c].plot(range(len(vals)), piv.mean().values, "o-", color=MODEL_COLORS[model], lw=2.2, label=f"{name} PDP")
        axes[1, c].plot(range(len(vals)), kept.mean().values, "o-", color=MODEL_COLORS[model], lw=2.2, label=name)
        print(field, name, "PDP", piv.mean().round(3).to_dict())
        print(field, name, "kept", kept.mean().round(2).to_dict())
    real = df.groupby(col)["true_next_pathway"].apply(lambda s: (s == TARGET).mean()).reindex(vals)
    axes[0, c].plot(range(len(vals)), real.values, "s--", color=NAVY, label="real share in the data")
    for r_ in (0, 1):
        axes[r_, c].set_xticks(range(len(vals)), vals, rotation=30)
    axes[0, c].set_title(f"{field}: vote share for '{TARGET}' (PDP)", fontsize=10)
    axes[1, c].set_title(f"{field}: original neighbours kept (of 10), mean", fontsize=10)
axes[0, 0].set_ylabel("vote share"); axes[1, 0].set_ylabel("neighbours kept")
axes[0, 0].legend(fontsize=8); axes[1, 0].legend(fontsize=8)
fig.suptitle("ICE / PDP, Model A vs Model B: the same 300 stays, one field varied at a time", x=0.01, ha="left")
plt.tight_layout()
fig.savefig(FIG / "fig_ice_pdp_AB.png", dpi=150, bbox_inches="tight")
print("saved", FIG / "fig_ice_pdp_AB.png")
