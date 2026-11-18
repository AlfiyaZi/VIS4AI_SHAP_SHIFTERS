"""One-off precompute for the Phase II dashboard (run once, ~5 min on CPU; output is gitignored).

Model A's map is Phase I's UMAP of all 30,248 stays (cohort columns umap_x / umap_y).
Model B gets the same kind of map at layers 0, 6 and 12 (all stays, cosine, mean-centred,
each layer initialised with the previous layout so the panels stay comparable).

Needs: shared/data_cache/phase2_model_b_layers.npy (scripts/01_embed_layers.py).
Writes: outputs/dashboard_umap_B.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import CACHE, OUT, l2n, load_cohort  # noqa: E402

LAYERS = [0, 6, 12]
df = load_cohort()
layers = np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")
res, init = {"encounter_id": df["encounter_id"].to_numpy()}, "spectral"
for layer in LAYERS:
    x = l2n(np.asarray(layers[:, layer, :], dtype=np.float32))
    x = x - x.mean(0)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", init=init, random_state=0).fit_transform(x)
    res[f"L{layer}_x"], res[f"L{layer}_y"] = xy[:, 0], xy[:, 1]
    init = xy
    print("layer", layer, "done", flush=True)
pd.DataFrame(res).to_parquet(OUT / "dashboard_umap_B.parquet", index=False)
print("wrote", OUT / "dashboard_umap_B.parquet")
