"""Step 1: re-create Model B's embeddings, keep all 13 layers, and check that Phase I is reproduced.

Writes (gitignored):
  shared/data_cache/phase2_model_b_layers.npy   [n, 13, 768] float16, mean-pooled per layer
Prints how many Phase I neighbours (retrieval_results.model_b_neighbors) are reproduced.
"""
import time

import numpy as np

from common import CACHE, BertEncoder, knn_other_patients, load_cohort

df = load_cohort()
texts = df["model_text"].tolist()
patient = df["patient_nbr"].to_numpy()
pos = {e: i for i, e in enumerate(df["encounter_id"])}

target = CACHE / "phase2_model_b_layers.npy"
if target.exists():
    layers = np.load(target, mmap_mode="r")
    print("cached:", target, layers.shape)
else:
    t0 = time.time()
    enc = BertEncoder()
    print("device:", enc.device)
    layers = enc.encode(texts, layers="all")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(target, layers)
    print(f"encoded {len(texts):,} texts in {time.time() - t0:.0f} s ->", target, layers.shape)

last = np.asarray(layers[:, -1, :], dtype=np.float32)
nbr, dist = knn_other_patients(last, patient)
phase1 = np.stack([[pos[e] for e in row] for row in df["model_b_neighbors"]])
overlap = np.mean([len(set(a) & set(b)) for a, b in zip(nbr, phase1)])
same_set = np.mean([set(a) == set(b) for a, b in zip(nbr, phase1)])
print(f"Phase I neighbours reproduced: mean overlap {overlap:.2f} / 10, identical sets {same_set:.1%}")
print("median d10 now / Phase I:", float(np.median(dist[:, -1])),
      float(np.median(np.stack(df["model_b_distances"].to_numpy())[:, -1])))
