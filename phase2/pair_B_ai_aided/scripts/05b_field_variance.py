"""Step 5b: how strongly does each field SHAPE Model B's space (not just: is it encoded)?

Probes in 05 show that every input field is linearly decodable from every layer - expected, the
fields are written in the text. What matters for retrieval is how much of the geometry each field
accounts for. For every layer and field: R^2 of a linear regression from the field (one-hot, numeric
fields binned in quartiles) to the centered, L2-normalised embedding = share of embedding variance
explained by that field alone. The outcome (true next pathway) is included for comparison.

Writes (gitignored) outputs/field_variance_B.csv.
"""
import numpy as np
import pandas as pd

from common import CACHE, FIELDS, OUT, l2n, load_cohort

df = load_cohort()
layers = np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")
rng = np.random.default_rng(5)
sub = np.sort(rng.choice(len(df), 12000, replace=False))
d = df.iloc[sub]

cols = dict(FIELDS) | {"next pathway (outcome)": "true_next_pathway"}
numeric = {"time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses"}
design = {}
for label, col in cols.items():
    v = d[col]
    if col in numeric:
        v = pd.qcut(v.rank(method="first"), 4, labels=False)
    design[label] = pd.get_dummies(v.astype(str)).to_numpy(float)

rows = []
for layer in range(layers.shape[1]):
    x = l2n(np.asarray(layers[sub, layer, :], dtype=np.float32))
    x = x - x.mean(0)
    total = (x ** 2).sum()
    for label, X in design.items():
        coef, *_ = np.linalg.lstsq(X, x, rcond=None)          # one-hot -> group means
        rows.append({"layer": layer, "field": label, "r2": float(((X @ coef) ** 2).sum() / total)})
res = pd.DataFrame(rows)
res.to_csv(OUT / "field_variance_B.csv", index=False)
print(res.pivot(index="field", columns="layer", values="r2").round(3)
      .sort_values(12, ascending=False).to_string())
