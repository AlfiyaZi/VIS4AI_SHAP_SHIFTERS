"""Step 2 (technique: embedding geometry): why does one edit replace almost all of Model B's neighbours?

Phase I (black box): the HbA1c edit replaces 8.2 of 10 neighbours, crowded stays (small distance gap)
lose more, emergency + unknown-specialty stays are the most crowded.
Phase II (inside): compare how far an edit moves the stay's embedding with how close its neighbours are.
If shift >> distance to the 10th neighbour, the whole neighbourhood must be replaced.

Writes (gitignored) outputs/edit_shift.parquet and outputs/anisotropy.csv.
"""
import numpy as np
import pandas as pd

from common import CACHE, OUT, BertEncoder, l2n, load_cohort, record_values, render

AGE_BRACKETS = [f"[{10 * i}-{10 * (i + 1)})" for i in range(10)]
EDITS = {
    "sex_swap": lambda v: {**v, "Sex": {"Male": "Female", "Female": "Male"}[v["Sex"]]},
    "age_shift": lambda v: {**v, "Age": AGE_BRACKETS[min(AGE_BRACKETS.index(v["Age"]) + 2, 9)]},
    "hba1c_elevation": lambda v: {**v, "A1C": ">8"} if v["A1C"] in ("None", "Norm") else v,
}

df = load_cohort()
values = [record_values(r) for r in df.to_dict("records")]
assert all(render(v) == t for v, t in zip(values, df["model_text"])), "template mismatch"

layers = np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")
rng = np.random.default_rng(0)

# --- anisotropy per layer: how similar are two random stays? ------------------------------
i, j = rng.integers(0, len(df), 5000), rng.integers(0, len(df), 5000)
rows = []
for layer in range(layers.shape[1]):
    x = np.asarray(layers[:, layer, :], dtype=np.float32)
    xn = l2n(x)
    xc = l2n(x - x.mean(0))                      # remove the direction shared by all stays
    rows.append({"layer": layer,
                 "random_pair_cosine": float((xn[i] * xn[j]).sum(1).mean()),
                 "random_pair_cosine_centered": float((xc[i] * xc[j]).sum(1).mean()),
                 "mean_direction_share": float(np.linalg.norm(x.mean(0)) ** 2 / (x ** 2).sum(1).mean())})
aniso = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
aniso.to_csv(OUT / "anisotropy.csv", index=False)
print(aniso.round(4).to_string(index=False))

# --- how far does each edit move a stay? -----------------------------------------------
last = l2n(np.asarray(layers[:, -1, :], dtype=np.float32))
d10 = np.stack(df["model_b_distances"].to_numpy())[:, -1]
enc = BertEncoder()
res = {"encounter_id": df["encounter_id"].to_numpy(), "d10": d10}
for name, fn in EDITS.items():
    applied = df[f"{name}_applied"].to_numpy()
    texts = [render(fn(v)) for v, a in zip(values, applied) if a]
    emb = l2n(enc.encode(texts))
    shift = np.full(len(df), np.nan, dtype=np.float32)
    shift[applied] = 1.0 - (emb * last[applied]).sum(1)          # cosine distance original -> edited
    res[f"shift_{name}"] = shift
    res[f"ratio_{name}"] = shift / np.maximum(d10, 1e-9)
    flips = df[f"model_b_flips_{name}"].astype(float).where(applied)
    rho = pd.Series(res[f"ratio_{name}"]).corr(flips, method="spearman")
    print(f"{name:16s} n={applied.sum():6,d}  median shift {np.nanmedian(shift):.5f}  "
          f"median shift/d10 {np.nanmedian(res[f'ratio_{name}']):.2f}  Spearman(ratio, flips) {rho:.2f}")
pd.DataFrame(res).to_parquet(OUT / "edit_shift.parquet", index=False)
