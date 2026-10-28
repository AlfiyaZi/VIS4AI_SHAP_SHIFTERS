"""
Data layer for the Pair A (AI-aided) Phase I dashboard.

Reads ONLY the static shared contract in ``shared/data/`` and derives the analysis
structures the four views need:

- one merged encounter table (cohort + retrieval + counterfactual flips)
- confusion matrices for Model A, Model B and the majority baseline, with a class order
  obtained by hierarchical clustering of the models' pairwise confusion
- HDBSCAN clusters on the input-space UMAP plus the input feature that best separates them
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "shared" / "data"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

PATHWAYS = ["Kidney / Renal", "Cardiovascular", "Eye / Neurological", "Uncomplicated / Other"]
PATHWAY_SHORT = {
    "Kidney / Renal": "Kidney",
    "Cardiovascular": "Cardio",
    "Eye / Neurological": "Eye/Neuro",
    "Uncomplicated / Other": "Other",
}
PATHWAY_COLORS = {
    "Kidney / Renal": "#8E6CBF",
    "Cardiovascular": "#D55E00",
    "Eye / Neurological": "#0072B2",
    "Uncomplicated / Other": "#A0A0A0",
}
A1C_ORDER = ["None", "Norm", ">7", ">8"]
A1C_COLORS = {"None": "#BDBDBD", "Norm": "#009E73", ">7": "#E69F00", ">8": "#B2182B"}
SEX_SYMBOLS = {"Female": "circle", "Male": "diamond"}

MODELS = {
    "model_a": "Model A (clinical features)",
    "model_b": "Model B (Bio_ClinicalBERT)",
    "majority": "Majority baseline",
}
EDITS = {"sex_swap": "Sex swap", "age_shift": "Age +20 y", "hba1c_elevation": "HbA1c to >8"}

# Input fields the UMAP was computed from (candidates for "what separates the clusters").
CATEGORICAL_DRIVERS = [
    "admission_type", "admission_source", "medical_specialty", "payer_code", "gender",
    "age", "max_glu_serum", "A1Cresult", "insulin", "change", "diabetesMed",
]
NUMERIC_DRIVERS = [
    "time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses", "age_mid",
]
INSPECTOR_FIELDS = [
    "gender", "age", "race", "time_in_hospital", "admission_type", "admission_source",
    "medical_specialty", "payer_code", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses", "max_glu_serum", "A1Cresult", "insulin",
    "change", "diabetesMed", "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "current_pathway", "readmitted",
]


@dataclass
class DashboardData:
    df: pd.DataFrame                      # one row per encounter, index = encounter_id
    metrics: dict
    order: list[str]                      # clustered pathway order for View 2
    confusion: dict[str, pd.DataFrame]    # model -> counts (true x pred), PATHWAYS order
    clusters: pd.Series                   # HDBSCAN label per encounter (-1 = noise)
    driver_ranking: pd.DataFrame          # feature, kind, association
    cluster_summary: pd.DataFrame         # cluster, x, y, n, label text
    pos: dict = field(default_factory=dict)   # encounter_id -> row position


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python shared/01_prepare_baseline_data.py` first."
        )
    return path


def load_tables() -> tuple[pd.DataFrame, dict]:
    cohort = pd.read_parquet(_require(DATA_DIR / "cohort_encounters.parquet"))
    retrieval = pd.read_parquet(_require(DATA_DIR / "retrieval_results.parquet"))
    flips = pd.read_parquet(_require(DATA_DIR / "counterfactual_flips.parquet"))
    with open(_require(DATA_DIR / "baseline_metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)

    df = cohort.merge(
        retrieval.drop(columns=["patient_nbr", "true_next_pathway"]), on="encounter_id",
        validate="one_to_one",
    ).merge(flips.drop(columns=["patient_nbr"]), on="encounter_id", validate="one_to_one")
    for col in ("gender", "A1Cresult", "true_next_pathway"):
        df[col] = df[col].astype(str)

    # Deterministic jitter for the discrete scatter axes.
    rng = np.random.default_rng(7)
    df["jitter_x"] = rng.uniform(-0.035, 0.035, len(df))
    df["jitter_y"] = rng.uniform(-0.38, 0.38, len(df))
    return df.set_index("encounter_id", drop=False), metrics


# ---------------------------------------------------------------------------
# View 2: confusion matrices + clustered class order
# ---------------------------------------------------------------------------
def confusion_counts(true: pd.Series, pred: pd.Series) -> pd.DataFrame:
    ct = pd.crosstab(pd.Categorical(true, PATHWAYS), pd.Categorical(pred, PATHWAYS),
                     dropna=False)
    ct.index, ct.columns = PATHWAYS, PATHWAYS
    return ct


def clustered_order(confusions: list[pd.DataFrame]) -> tuple[list[str], np.ndarray]:
    """Order classes so that mutually confused classes sit next to each other.

    Each class gets an error profile: its row-normalized confusion with every other class,
    in both directions, pooled over the given models. Classes are clustered with average
    linkage on 1 - symmetric confusion and the leaves are optimally ordered
    (Alsallakh et al. 2017 style).
    """
    sym = np.zeros((len(PATHWAYS), len(PATHWAYS)))
    for c in confusions:
        r = c.to_numpy(float)
        r = r / np.maximum(r.sum(axis=1, keepdims=True), 1)
        np.fill_diagonal(r, 0)
        sym += r + r.T
    sym /= max(sym.max(), 1e-12)
    dist = 1.0 - sym
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    z = optimal_leaf_ordering(linkage(condensed, method="average"), condensed)
    return [PATHWAYS[i] for i in leaves_list(z)], sym


# ---------------------------------------------------------------------------
# View 3: clusters on the UMAP and their driving feature
# ---------------------------------------------------------------------------
def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y).to_numpy(float)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    n = ct.sum()
    expected = ct.sum(1, keepdims=True) @ ct.sum(0, keepdims=True) / n
    chi2 = ((ct - expected) ** 2 / np.maximum(expected, 1e-12)).sum()
    r, k = ct.shape
    # Bergsma bias correction.
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def _correlation_ratio(labels: pd.Series, values: pd.Series) -> float:
    v = values.astype(float).to_numpy()
    total = ((v - v.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    between = sum(len(g) * (g.mean() - v.mean()) ** 2
                  for _, g in pd.Series(v).groupby(labels.to_numpy()))
    return float(np.sqrt(between / total))


def umap_clusters(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import HDBSCAN

    xy = df[["umap_x", "umap_y"]].to_numpy(np.float64)
    key = hashlib.sha256(xy.tobytes()).hexdigest()[:16]
    cache = CACHE_DIR / f"umap_clusters_{key}.npy"
    if cache.exists():
        labels = np.load(cache)
    else:
        min_size = max(100, len(df) // 60)
        labels = HDBSCAN(min_cluster_size=min_size, min_samples=25).fit_predict(xy)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, labels)
    labels = pd.Series(labels, index=df.index, name="cluster")

    inside = labels >= 0
    rows = []
    for col in CATEGORICAL_DRIVERS:
        rows.append((col, "categorical (Cramer's V)",
                     _cramers_v(df.loc[inside, col].astype(str), labels[inside])))
    for col in NUMERIC_DRIVERS:
        rows.append((col, "numeric (correlation ratio)",
                     _correlation_ratio(labels[inside], df.loc[inside, col])))
    ranking = (pd.DataFrame(rows, columns=["feature", "kind", "association"])
               .sort_values("association", ascending=False).reset_index(drop=True))

    driver = ranking.iloc[0]["feature"]
    summary = []
    for c, g in df[inside].groupby(labels[inside]):
        if driver in NUMERIC_DRIVERS:
            text = f"{driver} median {g[driver].median():.0f}"
        else:
            vc = g[driver].astype(str).value_counts(normalize=True)
            text = f"{vc.index[0]} ({vc.iloc[0]:.0%})"
        summary.append({
            "cluster": int(c), "x": g["umap_x"].median(), "y": g["umap_y"].median(),
            "n": len(g), "label": text,
            "disparity": g["agreement_disparity_b_minus_a"].mean(),
        })
    return labels, ranking, pd.DataFrame(summary)


@lru_cache(maxsize=1)
def load() -> DashboardData:
    df, metrics = load_tables()
    true = df["true_next_pathway"]
    confusion = {
        "model_a": confusion_counts(true, df["model_a_pred_pathway"]),
        "model_b": confusion_counts(true, df["model_b_pred_pathway"]),
        "majority": confusion_counts(true, df["majority_pred_pathway"]),
    }
    order, _ = clustered_order([confusion["model_a"], confusion["model_b"]])
    labels, ranking, summary = umap_clusters(df)
    df["umap_cluster"] = labels
    data = DashboardData(df=df, metrics=metrics, order=order, confusion=confusion,
                         clusters=labels, driver_ranking=ranking, cluster_summary=summary)
    data.pos = {eid: i for i, eid in enumerate(df.index)}
    return data
