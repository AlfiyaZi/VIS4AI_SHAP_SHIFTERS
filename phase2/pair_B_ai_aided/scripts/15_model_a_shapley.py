"""Step 15 (Model A vs B, technique: Shapley values over input fields, Lecture 04).

The same pairwise game as 03_field_shapley.py, played with Model A's vector on the SAME 300 pairs:
  v(S) = cos( A(record of r with the fields in S taken from q), A(r) ),  sum_f phi_f = cos(q, r) - 1.
Model A is cheap, so 200 permutations per pair instead of 20.
The 150 "neighbour" pairs are Model B's query + 1st neighbour, so this also asks: how does Model A
see the pairs Model B considers closest?

Writes (gitignored) outputs/field_shapley_A.parquet.
"""
import numpy as np
import pandas as pd

from common import FIELDS, OUT, ModelA, l2n, load_cohort, record_values

PERMS = 200
df = load_cohort()
values = [record_values(r) for r in df.to_dict("records")]
pos = {e: i for i, e in enumerate(df["encounter_id"])}
ma = ModelA(df)
pairs_b = pd.read_parquet(OUT / "field_shapley_B.parquet")[["q", "r", "kind"]].drop_duplicates()
rng = np.random.default_rng(1)
labels = list(FIELDS)

rows = []
for q_id, r_id, kind in pairs_b.itertuples(index=False):
    q, r = pos[q_id], pos[r_id]
    diff = [f for f in labels if values[q][f] != values[r][f]]
    ref = l2n(ma.encode([values[r]]))[0]
    phi = dict.fromkeys(labels, 0.0)
    for _ in range(PERMS if diff else 0):
        perm = list(rng.permutation(diff))
        recs, cur = [dict(values[r])], dict(values[r])
        for f in perm:
            cur[f] = values[q][f]
            recs.append(dict(cur))
        sims = l2n(ma.encode(recs)) @ ref
        for f, delta in zip(perm, np.diff(sims)):
            phi[f] += float(delta) / PERMS
    total = float(l2n(ma.encode([values[q]]))[0] @ ref) - 1.0
    for f in labels:
        rows.append({"q": q_id, "r": r_id, "kind": kind, "field": f, "differs": f in diff,
                     "phi": phi[f], "total": total})
res = pd.DataFrame(rows)
res.to_parquet(OUT / "field_shapley_A.parquet", index=False)
check = res.groupby(["q", "r"]).agg(s=("phi", "sum"), t=("total", "first"))
print("efficiency check, max |sum phi - (cos-1)|:", float((check.s - check.t).abs().max()))
d = res[res.differs]
summary = (d.groupby("field").agg(mismatch_cost=("phi", "mean"), n=("phi", "size"))
           .assign(share_of_loss=d.groupby("field").phi.sum() / d.phi.sum()).sort_values("mismatch_cost"))
print(summary.round(4).to_string())
print("mean cos-1 per kind:", res.groupby("kind").total.mean().round(4).to_dict())
