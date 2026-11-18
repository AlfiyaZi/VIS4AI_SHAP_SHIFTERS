"""Data layer for the Phase II dashboard (Pair B, AI-aided): Model A and Model B from the inside.

Reads the shared Phase I contract (shared/data) and the Phase II outputs (outputs/, gitignored),
and runs both models live for the selected stay:
  * Model A: the exact Phase I encoding (scripts/common.py: ModelA), cheap, exact
  * Model B: Bio_ClinicalBERT, loaded lazily on the first request that needs it (~10 s on CPU)
Neighbours are always 10 nearest stays of OTHER patients (cosine), vote = majority with ties going to
the class of the best-ranked neighbour (Phase I rule).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import (CACHE, FIELDS, K, OUT, PATHWAY_ORDER, ModelA, l2n, load_cohort,  # noqa: E402
                    record_values, render)

WALKTHROUGH_STAY = 80742510          # the stay of the Model B notebook, Section 7
EDITABLE = ["Age", "Sex", "A1C", "Specialty", "Source", "Admission Type", "Payer", "Insulin", "Max Glucose",
            "Med Change", "Diabetes Med"]
AGES = [f"[{10 * i}-{10 * (i + 1)})" for i in range(10)]
GROUPS = {
    "all": "All stays",
    "er_unknown": "Emergency Room + unknown specialty",
    "both_correct": "Both models correct",
    "only_a": "Only Model A correct",
    "only_b": "Only Model B correct",
    "both_wrong": "Both models wrong",
    "fragile_b": "Fragile in B (HbA1c edit replaces >= 8)",
    "fragile_a": "Fragile in A (HbA1c edit replaces >= 8)",
}


def vote(labels) -> str:
    vals, cnt = np.unique(labels, return_counts=True)
    tied = set(vals[cnt == cnt.max()])
    return next(v for v in labels if v in tied)


@dataclass
class DashboardData:
    df: pd.DataFrame
    values: list
    xa: np.ndarray                    # Model A vectors, L2-normalised [n, 127]
    xb: np.ndarray                    # Model B layer-12 vectors, L2-normalised [n, 768]
    ma: ModelA
    maps: dict                        # name -> [n, 2]
    weights: pd.DataFrame             # per field: Shapley cost A, B; outcome importance; variance share A, B
    groups: dict                      # key -> bool mask
    pos: dict                         # encounter_id -> row
    categories: dict                  # editable field -> allowed values
    _bert: object = field(default=None, repr=False)

    # ---- models -----------------------------------------------------------------------------
    def bert(self):
        if self._bert is None:
            from common import BertEncoder
            self._bert = BertEncoder()
        return self._bert

    def encode_a(self, records: list[dict]) -> np.ndarray:
        return l2n(self.ma.encode(records))

    def encode_b(self, records: list[dict]) -> np.ndarray:
        return l2n(self.bert().encode([render(r) for r in records]))

    def knn(self, model: str, vec: np.ndarray, row: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.xa if model == "A" else self.xb
        d = 1.0 - x @ vec
        d[self.df["patient_nbr"].to_numpy() == self.df["patient_nbr"].iat[row]] = np.inf
        nb = np.lexsort((np.arange(len(d)), d))[:K]
        return nb, d[nb]

    # ---- views ------------------------------------------------------------------------------
    def phase1_neighbours(self, model: str, row: int) -> tuple[np.ndarray, np.ndarray]:
        col = "model_a" if model == "A" else "model_b"
        nb = np.array([self.pos[e] for e in self.df[f"{col}_neighbors"].iat[row]])
        return nb, np.asarray(self.df[f"{col}_distances"].iat[row], dtype=float)

    def match_matrix(self, row: int, nb: np.ndarray) -> np.ndarray:
        """[10, 15] 1 where the neighbour has the same value as the stay, per field."""
        me = self.values[row]
        return np.array([[float(self.values[j][f] == me[f]) for f in FIELDS] for j in nb])

    def decomposition_a(self, row: int, nb: np.ndarray) -> pd.DataFrame:
        """Exact per-field terms of Model A's cosine between the stay and each neighbour."""
        raw = self.ma.encode([self.values[row]] + [self.values[j] for j in nb])
        q, r = raw[:1].repeat(len(nb), 0), raw[1:]
        terms = self.ma.field_terms(q, r) / (np.linalg.norm(q, axis=1) * np.linalg.norm(r, axis=1))[:, None]
        return pd.DataFrame(terms, columns=list(FIELDS))

    def field_swap(self, model: str, row: int, other: int) -> pd.DataFrame:
        """For each field where the two stays differ: similarity gained when the neighbour takes the
        stay's value for that field alone (how much that one difference costs)."""
        me, nbv = self.values[row], self.values[other]
        diff = [f for f in FIELDS if me[f] != nbv[f]]
        recs = [me, nbv] + [{**nbv, f: me[f]} for f in diff]
        e = self.encode_a(recs) if model == "A" else self.encode_b(recs)
        base = float(e[0] @ e[1])
        gains = [float(e[0] @ e[2 + i]) - base for i in range(len(diff))]
        return pd.DataFrame({"field": diff, "cost": gains}).sort_values("cost", ascending=False)

    def what_if(self, row: int, edits: dict) -> dict:
        """Re-run both models on an edited copy of the stay. Original neighbours are recomputed with
        the same vectors, so 'kept' counts only real changes."""
        rec = {**self.values[row], **{k: v for k, v in edits.items() if v is not None}}
        out = {"record": rec, "changed": [f for f in FIELDS if rec[f] != self.values[row][f]]}
        y = self.df["true_next_pathway"].to_numpy()
        for m in "AB":
            x = self.xa if m == "A" else self.xb
            orig, _ = self.knn(m, x[row], row)
            new_vec = (self.encode_a if m == "A" else self.encode_b)([rec])[0]
            new, dist = self.knn(m, new_vec, row)
            out[m] = {"orig": orig.tolist(), "new": new.tolist(), "dist": dist.tolist(),
                      "kept": int(len(set(orig) & set(new))),
                      "vote_before": vote(y[orig]), "vote_after": vote(y[new]),
                      "shares_before": {c: float((y[orig] == c).mean()) for c in PATHWAY_ORDER},
                      "shares_after": {c: float((y[new] == c).mean()) for c in PATHWAY_ORDER},
                      "shift": float(1 - x[row] @ new_vec)}
        return out


def load() -> DashboardData:
    df = load_cohort().reset_index(drop=True)
    values = [record_values(r) for r in df.to_dict("records")]
    ma = ModelA(df)
    xa = l2n(ma.encode(values))
    xb = l2n(np.asarray(np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")[:, -1, :], dtype=np.float32))

    maps = {"Model A (Phase I map)": df[["umap_x", "umap_y"]].to_numpy()}
    ub = pd.read_parquet(OUT / "dashboard_umap_B.parquet")
    assert (ub["encounter_id"].to_numpy() == df["encounter_id"].to_numpy()).all()
    for layer in (12, 6, 0):
        maps[f"Model B, layer {layer}"] = ub[[f"L{layer}_x", f"L{layer}_y"]].to_numpy()

    # per-field weights from the notebooks' outputs
    w = {}
    for m in "AB":
        s = pd.read_parquet(OUT / f"field_shapley_{m}.parquet")
        s = s[s.differs & (s.kind == "random")]
        w[f"shapley_{m}"] = -s.groupby("field").phi.mean()
    imp = pd.read_csv(OUT / "field_importance_outcome.csv").set_index("field").logloss_increase.clip(lower=0)
    va = pd.read_csv(OUT / "field_variance_A.csv").set_index("field").r2
    vb = pd.read_csv(OUT / "field_variance_B.csv").query("layer == 12").set_index("field").r2
    weights = pd.DataFrame({**w, "outcome": imp, "variance_A": va, "variance_B": vb}).reindex(list(FIELDS))

    ca, cb = df["model_a_correct"].to_numpy(bool), df["model_b_correct"].to_numpy(bool)
    fa = df["model_a_flips_hba1c_elevation"].astype(float).to_numpy()
    fb = df["model_b_flips_hba1c_elevation"].astype(float).to_numpy()
    groups = {
        "all": np.ones(len(df), bool),
        "er_unknown": ((df.admission_source == "Emergency Room") & (df.medical_specialty == "Unknown")).to_numpy(),
        "both_correct": ca & cb, "only_a": ca & ~cb, "only_b": ~ca & cb, "both_wrong": ~ca & ~cb,
        "fragile_b": np.nan_to_num(fb) >= 8, "fragile_a": np.nan_to_num(fa) >= 8,
    }
    categories = {f: sorted({v[f] for v in values}) for f in EDITABLE}
    categories["Age"] = AGES
    pos = {e: i for i, e in enumerate(df["encounter_id"])}
    return DashboardData(df, values, xa, xb, ma, maps, weights, groups, pos, categories)
