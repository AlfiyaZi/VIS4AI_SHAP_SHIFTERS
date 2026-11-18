"""Step 5c: a stronger probe for the OUTCOME per layer (correction to step 5).

Step 5 trained each probe on 8,400 stays with standardised features and 300 iterations. For the input
fields that is plenty (they are written in the text), but for the weak outcome signal it under-reads:
it suggested the outcome "fades with depth" (0.11 -> 0.04 above chance). A check with all train stays
(same patient split as steps 12 / 17: 80 / 20, seed 11), raw features, C = 0.5, up to 3,000
iterations, gives ~0.36-0.38 balanced accuracy at every layer, and the same for Model A's vector.

Writes (gitignored) outputs/outcome_probe_B.csv (with Model A's vector as layer = -1).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from common import CACHE, OUT, load_cohort

warnings.filterwarnings("ignore")
df = load_cohort()
y = df["true_next_pathway"].to_numpy()
rng = np.random.default_rng(11)
pats = df["patient_nbr"].unique()
test_p = set(rng.choice(pats, int(0.2 * len(pats)), replace=False))
te = df["patient_nbr"].isin(test_p).to_numpy()
tr = ~te
layers = np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")
inputs = [(-1, np.load(OUT / "model_a_x.npy"))] + [(l, None) for l in range(layers.shape[1])]
rows = []
for layer, x in inputs:
    x = np.asarray(layers[:, layer, :], dtype=np.float32) if x is None else x
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(x[tr], y[tr])
    b = balanced_accuracy_score(y[te], m.predict(x[te]))
    rows.append({"layer": layer, "balanced_acc": b, "above_chance": (b - 0.25) / 0.75})
    print(f"layer {layer:2d}  balanced acc {b:.3f}  above chance {(b - 0.25) / 0.75:.3f}", flush=True)
pd.DataFrame(rows).to_csv(OUT / "outcome_probe_B.csv", index=False)
