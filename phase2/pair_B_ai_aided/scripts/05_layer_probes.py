"""Step 5 (technique: concept probing per layer - an adaptation of network dissection, Lecture 05).

Question (from Phase I RQ3/RQ5): what does Model B's embedding encode, and where does it appear?
For every layer (0 = token embeddings ... 12 = the layer Model B uses) a linear probe (logistic
regression) predicts one concept from the mean-pooled embedding. Train/test split by patient.
Score = balanced accuracy, with chance = 1 / number of classes. Concepts = input fields that Phase I
found matched (specialty, payer, admission source, sex, HbA1c, ...) and the outcome (next pathway),
which Phase I found NOT matched (lift +0.01).

Writes (gitignored) outputs/layer_probes_B.csv.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from common import CACHE, OUT, load_cohort

df = load_cohort()
layers = np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")


def top(col, n):
    keep = df[col].value_counts().index[:n]
    return df[col].where(df[col].isin(keep), "other")


CONCEPTS = {
    "sex": df["gender"],
    "age bracket": df["age"],
    "admission source (top 4)": top("admission_source", 4),
    "specialty (top 8)": top("medical_specialty", 8),
    "payer (top 6)": top("payer_code", 6),
    "HbA1c": df["A1Cresult"],
    "insulin": df["insulin"],
    "medication change": df["change"],
    "ER + unknown specialty": (df["admission_source"].eq("Emergency Room") & df["medical_specialty"].eq("Unknown")).astype(str),
    "next pathway (outcome)": df["true_next_pathway"],
}

rng = np.random.default_rng(3)
sub = np.sort(rng.choice(len(df), 12000, replace=False))
groups = df["patient_nbr"].to_numpy()[sub]
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(sub, groups=groups))

rows = []
for layer in range(layers.shape[1]):
    x = np.asarray(layers[sub, layer, :], dtype=np.float32)
    sc = StandardScaler().fit(x[tr])
    xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
    for name, y in CONCEPTS.items():
        y = y.to_numpy()[sub]
        clf = LogisticRegression(max_iter=300, C=0.5, class_weight="balanced").fit(xtr, y[tr])
        bacc = balanced_accuracy_score(y[te], clf.predict(xte))
        k = len(np.unique(y))
        rows.append({"layer": layer, "concept": name, "balanced_acc": bacc, "chance": 1 / k,
                     "above_chance": (bacc - 1 / k) / (1 - 1 / k)})
    print(f"layer {layer:2d} done")
res = pd.DataFrame(rows)
res.to_csv(OUT / "layer_probes_B.csv", index=False)
print(res.pivot(index="concept", columns="layer", values="above_chance").round(2).to_string())
