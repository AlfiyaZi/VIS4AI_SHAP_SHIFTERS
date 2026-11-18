"""Step 17 (Model A vs B: how much outcome signal is in the 15 fields at all?).

Fine-tuning (step 12) suggested the fields carry little information about the next pathway. Here a
direct check with classifiers TRAINED on the outcome, same patient split as step 12 (80 / 20, seed 11):
  * logistic regression (class-weighted) on Model A's vector
  * gradient boosting (class-weighted) on Model A's vector
  * logistic regression (class-weighted) on Model B's embedding (layer 12)
  * Model A / Model B as used (KNN k = 10, vote) on the same test stays, index = train stays
Metric: macro recall (chance = 0.25) and per-pathway recall.
Then: which fields does the outcome actually depend on? Permutation importance per FIELD (all of a
field's columns shuffled together) for the boosting model, measured as the rise in test log-loss.
This is "what the data rewards", to set next to "what each model weighs" (Shapley, steps 3 / 15).

Writes (gitignored) outputs/supervised_reference.csv and outputs/field_importance_outcome.csv.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_sample_weight

from common import CACHE, FIELDS, K, OUT, PATHWAY_ORDER, ModelA, l2n, load_cohort

df = load_cohort()
y = df["true_next_pathway"].to_numpy()
rng = np.random.default_rng(11)                                   # identical split to 12_finetune.py
pats = df["patient_nbr"].unique()
test_p = set(rng.choice(pats, int(0.2 * len(pats)), replace=False))
te = df["patient_nbr"].isin(test_p).to_numpy()
tr = ~te
xa = np.load(OUT / "model_a_x.npy")
xb = np.asarray(np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")[:, -1, :], dtype=np.float32)
ma = ModelA(df)


def recalls(pred, yt):
    return {c: float((pred[yt == c] == c).mean()) for c in PATHWAY_ORDER}


def knn_vote(x):
    ix, qx = l2n(x[tr]), l2n(x[te])
    out = []
    for s in range(0, len(qx), 2048):
        nb = np.argsort(1 - qx[s:s + 2048] @ ix.T, axis=1, kind="stable")[:, :K]
        for row in y[tr][nb]:
            vals, cnt = np.unique(row, return_counts=True)
            tied = set(vals[cnt == cnt.max()])
            out.append(next(v for v in row if v in tied))
    return np.array(out)


rows = []
sw = compute_sample_weight("balanced", y[tr])
models = {
    "LogReg on Model A vector": (LogisticRegression(max_iter=3000, class_weight="balanced"), xa),
    "Gradient boosting on Model A vector": (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                           random_state=0), xa),
    "LogReg on Model B embedding": (LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5), xb),
}
fitted = {}
for name, (m, x) in models.items():
    m.fit(x[tr], y[tr], **({"sample_weight": sw} if "boosting" in name else {}))
    pred = m.predict(x[te])
    fitted[name] = m
    rec = recalls(pred, y[te])
    rows.append({"model": name, "trained_on_outcome": True, "accuracy": float((pred == y[te]).mean()),
                 "macro_recall": float(np.mean(list(rec.values()))), **rec})
for name, x in [("Model A (KNN vote)", xa), ("Model B (KNN vote)", xb)]:
    pred = knn_vote(x)
    rec = recalls(pred, y[te])
    rows.append({"model": name, "trained_on_outcome": False, "accuracy": float((pred == y[te]).mean()),
                 "macro_recall": float(np.mean(list(rec.values()))), **rec})
res = pd.DataFrame(rows)
res.to_csv(OUT / "supervised_reference.csv", index=False)
print(res.round(3).to_string(index=False))

# permutation importance per field (boosting on Model A vector)
gb = fitted["Gradient boosting on Model A vector"]
sw_te = compute_sample_weight("balanced", y[te])
base = log_loss(y[te], gb.predict_proba(xa[te]), labels=gb.classes_, sample_weight=sw_te)
r = np.random.default_rng(0)
imp = []
for f, s in ma.slices.items():
    gains = []
    for _ in range(5):
        xp = xa[te].copy()
        xp[:, s] = xp[r.permutation(len(xp))][:, s]
        gains.append(log_loss(y[te], gb.predict_proba(xp), labels=gb.classes_, sample_weight=sw_te) - base)
    imp.append({"field": f, "logloss_increase": float(np.mean(gains)), "sd": float(np.std(gains))})
imp = pd.DataFrame(imp).sort_values("logloss_increase", ascending=False)
imp.to_csv(OUT / "field_importance_outcome.csv", index=False)
print(f"balanced test log-loss {base:.4f} (uniform guess: {np.log(4):.4f})")
print(imp.round(4).to_string(index=False))
