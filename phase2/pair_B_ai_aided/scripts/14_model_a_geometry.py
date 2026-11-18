"""Step 14 (Model A, same questions as steps 1, 2 and 5b for Model B).

Model A is transparent: 5 z-scored numbers + 10 one-hot fields -> cosine KNN. So "looking inside" is
exact here: cos(q, r) = sum over the 15 fields of term_f / (|q| |r|), with
term_f = z_q * z_r for a numeric field and 1 / 0 (same value / different value) for a categorical one.

1. Reproduction: rebuild Model A from the records and compare with Phase I's neighbours.
2. Geometry: random-pair cosine (Model B: 0.995 = everything in one narrow cone).
3. Exact decomposition of the similarity to the 10 neighbours, per field.
4. Variance share per field (same design as 05b) -> directly comparable with Model B's layer 12.
5. Edit shift / neighbourhood radius for sex, age +20, HbA1c (same edits as Phase I) and its
   correlation with the Phase I neighbour flips.

Writes (gitignored) outputs/model_a_x.npy, model_a_decomposition.csv, field_variance_A.csv,
edit_shift_A.parquet, geometry_A.json.
"""
import json

import numpy as np
import pandas as pd

from common import FIELDS, OUT, ModelA, knn_other_patients, l2n, load_cohort, record_values

AGE_BRACKETS = [f"[{10 * i}-{10 * (i + 1)})" for i in range(10)]
EDITS = {
    "sex_swap": lambda v: {**v, "Sex": {"Male": "Female", "Female": "Male"}[v["Sex"]]},
    "age_shift": lambda v: {**v, "Age": AGE_BRACKETS[min(AGE_BRACKETS.index(v["Age"]) + 2, 9)]},
    "hba1c_elevation": lambda v: {**v, "A1C": ">8"} if v["A1C"] in ("None", "Norm") else v,
}

df = load_cohort()
values = [record_values(r) for r in df.to_dict("records")]
patient = df["patient_nbr"].to_numpy()
ma = ModelA(df)
x = ma.encode(values)
np.save(OUT / "model_a_x.npy", x)
xn = l2n(x)
print(f"Model A: {x.shape[1]} columns ({len(ma.NUMERIC)} numeric + one-hot)")

# 1. reproduction --------------------------------------------------------------------------
nbr, dist = knn_other_patients(x, patient)
pos = {e: i for i, e in enumerate(df["encounter_id"])}
phase1 = np.array([[pos[e] for e in row] for row in df["model_a_neighbors"]])
same = (nbr[:, :, None] == phase1[:, None, :]).any(2).sum(1)
d10_p1 = np.stack(df["model_a_distances"].to_numpy())[:, -1]
geo = {"reproduction_mean_shared_neighbours": float(same.mean()),
       "reproduction_identical_sets": float((same == 10).mean()),
       "median_d10": float(np.median(dist[:, -1])), "median_d10_phase1": float(np.median(d10_p1))}
# where sets differ: is it only a different pick among stays at exactly the 10th distance (a tie)?
diff = same < 10
geo["differing_sets_explained_by_ties"] = float(np.isclose(dist[diff, -1], d10_p1[diff], atol=1e-5).mean()) if diff.any() else 1.0
print({k: round(v, 4) for k, v in geo.items()})

# 2. geometry ------------------------------------------------------------------------------
rng = np.random.default_rng(0)
i, j = rng.integers(0, len(df), 5000), rng.integers(0, len(df), 5000)
xc = l2n(x - x.mean(0))
geo["random_pair_cosine"] = float((xn[i] * xn[j]).sum(1).mean())
geo["random_pair_cosine_centered"] = float((xc[i] * xc[j]).sum(1).mean())
geo["share_identical_record_neighbour1"] = float((dist[:, 0] < 1e-6).mean())
geo["share_d10_zero"] = float((dist[:, -1] < 1e-6).mean())
print("random-pair cosine", round(geo["random_pair_cosine"], 3), "centered", round(geo["random_pair_cosine_centered"], 3))

# 3. exact decomposition of the similarity to the neighbours --------------------------------
q = np.repeat(np.arange(len(df)), 10)
r = nbr.ravel()
norms = np.linalg.norm(x, axis=1)
terms = ma.field_terms(x[q], x[r]) / (norms[q] * norms[r])[:, None]
assert np.allclose(terms.sum(1), (xn[q] * xn[r]).sum(1), atol=1e-4), "decomposition must be exact"
# the same for random pairs, to see which fields neighbours share MORE than random stays do
tr = ma.field_terms(x[i], x[j]) / (norms[i] * norms[j])[:, None]
dec = pd.DataFrame({"field": list(FIELDS), "neighbour_term": terms.mean(0), "random_term": tr.mean(0)})
dec["lift"] = dec.neighbour_term - dec.random_term
dec["share_of_similarity"] = dec.neighbour_term / dec.neighbour_term.sum()
dec.to_csv(OUT / "model_a_decomposition.csv", index=False)
print(dec.round(4).sort_values("neighbour_term", ascending=False).to_string(index=False))

# 4. variance share per field (identical design to 05b_field_variance.py) --------------------
rng5 = np.random.default_rng(5)
sub = np.sort(rng5.choice(len(df), 12000, replace=False))
d = df.iloc[sub]
cols = dict(FIELDS) | {"next pathway (outcome)": "true_next_pathway"}
numeric = {"time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses"}
xs = l2n(x[sub]); xs = xs - xs.mean(0)
total = (xs ** 2).sum()
rows = []
for label, col in cols.items():
    v = d[col]
    if col in numeric:
        v = pd.qcut(v.rank(method="first"), 4, labels=False)
    X = pd.get_dummies(v.astype(str)).to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, xs, rcond=None)
    rows.append({"field": label, "r2": float(((X @ coef) ** 2).sum() / total)})
var = pd.DataFrame(rows)
var.to_csv(OUT / "field_variance_A.csv", index=False)
print(var.sort_values("r2", ascending=False).round(3).to_string(index=False))

# 5. edits: shift vs neighbourhood radius ---------------------------------------------------
res = {"encounter_id": df["encounter_id"].to_numpy(), "d10": dist[:, -1]}
for name, fn in EDITS.items():
    applied = df[f"{name}_applied"].to_numpy()
    e = l2n(ma.encode([fn(v) for v, a in zip(values, applied) if a]))
    shift = np.full(len(df), np.nan, dtype=np.float32)
    shift[applied] = 1.0 - (e * xn[applied]).sum(1)
    ratio = shift / np.maximum(dist[:, -1], 1e-9)
    res[f"shift_{name}"], res[f"ratio_{name}"] = shift, ratio
    flips = df[f"model_a_flips_{name}"].astype(float).where(applied)
    rho = pd.Series(ratio).corr(flips, method="spearman")
    geo[f"median_ratio_{name}"] = float(np.nanmedian(ratio))
    geo[f"mean_flips_{name}"] = float(np.nanmean(flips))
    geo[f"spearman_ratio_flips_{name}"] = float(rho)
    print(f"{name:16s} median shift {np.nanmedian(shift):.4f}  shift/d10 {np.nanmedian(ratio):.2f}  "
          f"flips {np.nanmean(flips):.2f}  Spearman {rho:.2f}")
pd.DataFrame(res).to_parquet(OUT / "edit_shift_A.parquet", index=False)
json.dump(geo, open(OUT / "geometry_A.json", "w"), indent=1)
