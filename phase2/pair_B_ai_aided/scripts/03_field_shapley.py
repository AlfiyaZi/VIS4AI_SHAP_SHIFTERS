"""Step 3 (technique: Shapley values over input fields, model-agnostic, Lecture 04).

Question (from Phase I RQ3/RQ4): which fields make two stays (dis)similar for Model B?

Game for a pair of stays (q, r):  players = the 15 fields.
  v(S) = cos( emb(text of r with the fields in S replaced by q's values), emb(r) )
  v(empty) = 1 (r compared with itself), v(all) = cos(q, r).
Every coalition is a real-looking record (each field holds a real value), so no out-of-distribution
"masked" texts. Fields where q and r agree are dummies (phi = 0). phi_f < 0 = similarity lost because
field f differs. Efficiency: sum_f phi_f = cos(q, r) - 1 exactly (holds per sampled permutation).

The same game can be played with Model A's vector, so the two models are directly comparable.
Writes (gitignored) outputs/field_shapley_B.parquet.
"""
import argparse

import numpy as np
import pandas as pd

from common import FIELDS, OUT, BertEncoder, l2n, load_cohort, record_values, render

p = argparse.ArgumentParser()
p.add_argument("--pairs", type=int, default=300)
p.add_argument("--perms", type=int, default=20)
args = p.parse_args()

df = load_cohort()
values = [record_values(r) for r in df.to_dict("records")]
rng = np.random.default_rng(1)
pos = {e: i for i, e in enumerate(df["encounter_id"])}

# half random pairs (what the model weighs in general), half query + its 1st Phase I neighbour
q_rand = rng.choice(len(df), args.pairs // 2, replace=False)
r_rand = rng.choice(len(df), args.pairs // 2, replace=False)
q_nb = rng.choice(len(df), args.pairs // 2, replace=False)
r_nb = np.array([pos[df["model_b_neighbors"].iloc[q][0]] for q in q_nb])
pairs = [(q, r, "random") for q, r in zip(q_rand, r_rand)] + [(q, r, "neighbour") for q, r in zip(q_nb, r_nb)]

# collect every text we need, encode once
plans, needed = [], set()
labels = list(FIELDS)
for q, r, kind in pairs:
    diff = [f for f in labels if values[q][f] != values[r][f]]
    perms = [list(rng.permutation(diff)) for _ in range(args.perms)] if diff else []
    for perm in perms:
        cur = dict(values[r])
        needed.add(render(cur))
        for f in perm:
            cur[f] = values[q][f]
            needed.add(render(cur))
    plans.append((q, r, kind, diff, perms))
needed = sorted(needed)
print(f"{len(pairs)} pairs, {len(needed):,} distinct texts to encode")
emb = dict(zip(needed, l2n(BertEncoder().encode(needed))))

rows = []
for q, r, kind, diff, perms in plans:
    ref = emb[render(values[r])]
    phi = dict.fromkeys(labels, 0.0)
    for perm in perms:
        cur = dict(values[r])
        prev = float(emb[render(cur)] @ ref)
        for f in perm:
            cur[f] = values[q][f]
            now = float(emb[render(cur)] @ ref)
            phi[f] += (now - prev) / len(perms)
            prev = now
    total = float(emb[render(values[q])] @ ref) - 1.0
    for f in labels:
        rows.append({"q": int(df["encounter_id"].iloc[q]), "r": int(df["encounter_id"].iloc[r]), "kind": kind,
                     "field": f, "differs": f in diff, "phi": phi[f], "total": total})
res = pd.DataFrame(rows)
res.to_parquet(OUT / "field_shapley_B.parquet", index=False)

check = res.groupby(["q", "r"]).agg(s=("phi", "sum"), t=("total", "first"))
print("efficiency check, max |sum phi - (cos-1)|:", float((check.s - check.t).abs().max()))
summary = (res[res.differs].groupby("field")
           .agg(mismatch_cost=("phi", "mean"), n=("phi", "size"))
           .assign(share_of_loss=lambda t: res[res.differs].groupby("field").phi.sum() / res[res.differs].phi.sum())
           .sort_values("mismatch_cost"))
print(summary.round(5).to_string())
