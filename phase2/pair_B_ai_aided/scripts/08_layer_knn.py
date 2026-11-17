"""Step 8: how does the classification form across the layers? ("logit lens" for a KNN classifier)

Model B classifies with the vector after layer 12. Here the SAME classifier (cosine KNN, k = 10,
same-patient stays excluded, majority vote with Phase I's tie rule) is run on the mean-pooled vector
after EVERY layer 0..12. Per layer:
  accuracy vs. the majority baseline, recall of each pathway,
  overlap of the neighbours with the final (layer 12) neighbours,
  how often the neighbours share the patient's specialty / admission source / next pathway.

Writes (gitignored) outputs/layer_knn_B.csv.
"""
import numpy as np
import pandas as pd

from common import CACHE, OUT, knn_other_patients, load_cohort

df = load_cohort()
layers = np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")
patient = df["patient_nbr"].to_numpy()
y = df["true_next_pathway"].to_numpy()
classes = sorted(np.unique(y))
spec = df["medical_specialty"].to_numpy()
src = df["admission_source"].to_numpy()


def vote(nb):
    lab = y[nb]
    pred = np.empty(len(nb), dtype=object)
    for i, row in enumerate(lab):
        vals, cnt = np.unique(row, return_counts=True)
        tied = set(vals[cnt == cnt.max()])
        pred[i] = next(v for v in row if v in tied)          # Phase I rule: tie -> best-ranked neighbour
    return pred


final_nb, _ = knn_other_patients(np.asarray(layers[:, -1, :], dtype=np.float32), patient)
rows = []
for layer in range(layers.shape[1]):
    nb, dist = knn_other_patients(np.asarray(layers[:, layer, :], dtype=np.float32), patient)
    pred = vote(nb)
    row = {"layer": layer, "accuracy": float((pred == y).mean()),
           "overlap_with_final": float(np.mean([len(set(a) & set(b)) for a, b in zip(nb, final_nb)])) / 10,
           "same_specialty": float((spec[nb] == spec[:, None]).mean()),
           "same_source": float((src[nb] == src[:, None]).mean()),
           "same_pathway": float((y[nb] == y[:, None]).mean()),
           "median_d10": float(np.median(dist[:, -1]))}
    for c in classes:
        row[f"recall {c}"] = float((pred[y == c] == c).mean())
    rows.append(row)
    print(f"layer {layer:2d}: acc {row['accuracy']:.3f}  overlap {row['overlap_with_final']:.2f}  "
          f"same pathway {row['same_pathway']:.3f}  same specialty {row['same_specialty']:.3f}")
res = pd.DataFrame(rows)
res["majority_baseline"] = float((y == pd.Series(y).mode()[0]).mean())
res.to_csv(OUT / "layer_knn_B.csv", index=False)
print(res.round(3).to_string(index=False))
