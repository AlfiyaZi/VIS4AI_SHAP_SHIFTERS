"""
01_prepare_baseline_data.py
===========================
Shared Phase I baseline for team SHAP_SHIFTERS (JKU AI and Visualization Lab WS26).

Run ONCE by whoever has a working PyTorch setup. Everything it produces is a static,
self-contained table in ``shared/data/``. Both pairs (AI-aided and human-only) analyse
these tables; nobody downstream has to run a model, a GPU or this script.

Pipeline
--------
1. Data: UCI Diabetes 130-US Hospitals 1999-2008 (UCI ID 296). Downloaded once into
   ``shared/data_cache/`` (official UCI zip via ``urllib``; ``ucimlrepo`` as second source).
2. Cohort: patients with >= 2 encounters (``patient_nbr``). Visits are ordered by
   ``encounter_id``. The label of visit V_t is the primary diagnosis (``diag_1``) of the
   NEXT visit V_t+1, mapped to 4 pathways. The final visit of every patient is dropped
   (it has no next admission to evaluate).
3. Inputs: 15 authentic UCI encounter fields of V_t only. No generated or invented text.
   Model B reads them through one fixed serialization template (``TEXT_TEMPLATE``);
   Model A reads exactly the same fields as a numeric / one-hot vector.
4. Retrieval (k = 10, cosine, neighbors always from OTHER patients):
   - Model A: StandardScaler (numeric) + one-hot (categorical) -> cosine KNN.
   - Model B: ``emilyalsentzer/Bio_ClinicalBERT``, mean-pooled last layer (768-D) -> cosine KNN.
   - Baselines: majority class and random neighbors.
5. Counterfactual edits of the query (index unchanged): sex swap, age +20 years,
   HbA1c None/Norm -> ">8". Neighbor flips = k - |original neighbors & edited neighbors|.
6. 2-D UMAP of the input features (Model A space; no model output involved).

There are NO silent fallbacks: a failed download, a missing torch / transformers install or a
model that does not produce 768-D embeddings raises an error.

Outputs (``shared/data/``, Parquet + CSV; list columns are JSON strings in the CSV files)
------------------------------------------------------------------------------------------
- cohort_encounters.parquet     raw UCI fields, derived fields, true next pathway, UMAP x/y
- retrieval_results.parquet     neighbors, agreement and predictions of A, B and baselines
- counterfactual_flips.parquet  neighbor flips per encounter for 3 edits x 2 models
- baseline_metrics.json         cohort statistics, recall, agreement, baselines, provenance

Phase II material (embeddings, feature matrices) goes to
``shared/data_cache/phase2_model_internals.npz`` and stays closed during Phase I.

Usage
-----
    python shared/01_prepare_baseline_data.py                  # full run (GPU if available)
    python shared/01_prepare_baseline_data.py --device cpu     # explicit CPU batcher
    python shared/01_prepare_baseline_data.py --download-only  # just fetch the UCI data
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UCI_DATASET_ID = 296
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
UCI_EXPECTED_ROWS = 101_766
MODEL_B_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_B_DIM = 768

KIDNEY = "Kidney / Renal"
CARDIO = "Cardiovascular"
EYE_NEURO = "Eye / Neurological"
OTHER = "Uncomplicated / Other"
PATHWAYS = [KIDNEY, CARDIO, EYE_NEURO, OTHER]

# Official labels from the dataset's IDs_mapping.csv. The text model reads these labels
# instead of bare integers; the raw IDs are kept in the cohort table.
ADMISSION_TYPE_LABELS = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 5: "Not Available",
    6: "NULL", 7: "Trauma Center", 8: "Not Mapped",
}
ADMISSION_SOURCE_LABELS = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Not Available", 10: "Transfer from critial access hospital",
    11: "Normal Delivery", 12: "Premature Delivery", 13: "Sick Baby", 14: "Extramural Birth",
    15: "Not Available", 17: "NULL", 18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency", 20: "Not Mapped", 21: "Unknown/Invalid",
    22: "Transfer from hospital inpt/same fac reslt in a sep claim",
    23: "Born inside this hospital", 24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center", 26: "Transfer from Hospice",
}

AGE_BRACKETS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
AGE_MIDPOINT = {b: 10 * i + 5 for i, b in enumerate(AGE_BRACKETS)}

# The 15 authentic UCI fields both models see. ``Sex`` is prepended to the agreed template
# because the sex-swap counterfactual would otherwise be a no-op for Model B.
TEXT_TEMPLATE = (
    "Sex: {gender}, Age: {age}, Hospital Stay: {time_in_hospital} days, "
    "Admission Type: {admission_type}, Source: {admission_source}, "
    "Specialty: {medical_specialty}, Payer: {payer_code}, Labs: {num_lab_procedures}, "
    "Medications: {num_medications}, Diagnoses count: {number_diagnoses}, "
    "Max Glucose: {max_glu_serum}, A1C: {A1Cresult}, Insulin: {insulin}, "
    "Med Change: {change}, Diabetes Med: {diabetesMed}."
)
MODEL_A_NUMERIC = [
    "age_mid", "time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses",
]
MODEL_A_CATEGORICAL = [
    "gender", "admission_type", "admission_source", "medical_specialty", "payer_code",
    "max_glu_serum", "A1Cresult", "insulin", "change", "diabetesMed",
]
EDITS = ["sex_swap", "age_shift", "hba1c_elevation"]


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Ground truth: ICD-9 -> pathway
# ---------------------------------------------------------------------------
def map_icd9_to_pathway(code: object) -> str:
    """Map one ICD-9 code (UCI format, e.g. '250.41', '428', 'V57', '?') to a pathway.

    Kidney / Renal      250.4x, 580-589
    Eye / Neurological  250.5x, 250.6x, 357.x, 362.x
    Cardiovascular      250.7x, 390-459
    Uncomplicated/Other everything else (incl. V/E codes, missing)
    250.x sub-codes are resolved first.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return OTHER
    s = str(code).strip()
    if not s or s == "?" or s[0] in "VvEe":
        return OTHER
    try:
        major = int(float(s))
    except ValueError:
        return OTHER
    if major == 250:
        sub = s.split(".", 1)[1][:1] if "." in s else ""
        if sub == "4":
            return KIDNEY
        if sub in ("5", "6"):
            return EYE_NEURO
        if sub == "7":
            return CARDIO
        return OTHER
    if 580 <= major <= 589:
        return KIDNEY
    if major in (357, 362):
        return EYE_NEURO
    if 390 <= major <= 459:
        return CARDIO
    return OTHER


# ---------------------------------------------------------------------------
# 2. Data download / loading
# ---------------------------------------------------------------------------
MANUAL_FETCH_HELP = f"""
Could not obtain the UCI Diabetes 130-US dataset automatically. Fetch it manually, then rerun:

  PowerShell:
    Invoke-WebRequest -Uri "{UCI_ZIP_URL}" -OutFile shared/data_cache/uci296.zip
    Expand-Archive shared/data_cache/uci296.zip -DestinationPath shared/data_cache/uci296
  bash:
    curl -L -o shared/data_cache/uci296.zip "{UCI_ZIP_URL}"
    unzip shared/data_cache/uci296.zip -d shared/data_cache/uci296

  then:  python shared/01_prepare_baseline_data.py --uci-csv shared/data_cache/uci296/<...>/diabetic_data.csv
"""


def _extract_csv_from_zip(zip_path: Path, target_csv: Path) -> None:
    with zipfile.ZipFile(zip_path) as outer:
        names = outer.namelist()
        csv_names = [n for n in names if n.endswith("diabetic_data.csv")]
        if csv_names:
            with outer.open(csv_names[0]) as src, open(target_csv, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return
        # Some UCI mirrors nest a second zip (dataset_diabetes.zip) inside the archive.
        inner_zips = [n for n in names if n.endswith(".zip")]
        for inner in inner_zips:
            inner_path = target_csv.parent / Path(inner).name
            with outer.open(inner) as src, open(inner_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            try:
                _extract_csv_from_zip(inner_path, target_csv)
                return
            finally:
                inner_path.unlink(missing_ok=True)
    raise FileNotFoundError(f"diabetic_data.csv not found inside {zip_path}")


def download_uci(cache_dir: Path) -> Path:
    """Download the official UCI archive (urllib), falling back to ``ucimlrepo``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_csv = cache_dir / "diabetic_data.csv"
    errors = []

    zip_path = cache_dir / "uci296.zip"
    try:
        log(f"  downloading {UCI_ZIP_URL}")
        req = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "shap-shifters/2.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        _extract_csv_from_zip(zip_path, target_csv)
        log(f"  saved {target_csv}")
        return target_csv
    except Exception as exc:  # noqa: BLE001 - collected and re-raised below
        errors.append(f"urllib: {exc!r}")
    finally:
        zip_path.unlink(missing_ok=True)

    try:
        log("  trying ucimlrepo.fetch_ucirepo(id=296)")
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=UCI_DATASET_ID)
        df = pd.concat([ds.data.ids, ds.data.features, ds.data.targets], axis=1)
        # ucimlrepo parses the literal string "None" (= not measured) as NaN; restore it.
        for col in ("max_glu_serum", "A1Cresult"):
            df[col] = df[col].fillna("None")
        df = df.fillna("?")
        df.to_csv(target_csv, index=False)
        log(f"  saved {target_csv}")
        return target_csv
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ucimlrepo: {exc!r}")

    raise RuntimeError("UCI download failed:\n  " + "\n  ".join(errors) + "\n" + MANUAL_FETCH_HELP)


def load_uci_raw(csv_path: Path) -> pd.DataFrame:
    # keep_default_na=False: the literal "None" in A1Cresult / max_glu_serum means
    # "not measured" and must not become NaN. "?" is UCI's missing-value marker.
    df = pd.read_csv(
        csv_path,
        na_values=["?"],
        keep_default_na=False,
        dtype={"diag_1": str, "diag_2": str, "diag_3": str, "payer_code": str},
        low_memory=False,
    )
    required = {"encounter_id", "patient_nbr", "gender", "age", "diag_1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is not the UCI Diabetes 130-US file (missing {missing}).")
    if len(df) != UCI_EXPECTED_ROWS:
        log(f"  [warning] expected {UCI_EXPECTED_ROWS} rows, found {len(df)}")
    return df


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 3. Cohort
# ---------------------------------------------------------------------------
def serialize_record(row: pd.Series | dict) -> str:
    return TEXT_TEMPLATE.format(**{k: row[k] for k in (
        "gender", "age", "time_in_hospital", "admission_type", "admission_source",
        "medical_specialty", "payer_code", "num_lab_procedures", "num_medications",
        "number_diagnoses", "max_glu_serum", "A1Cresult", "insulin", "change", "diabetesMed",
    )})


def add_model_input_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Derived, input-side fields (all from the current visit V_t)."""
    df = df.copy()
    df["age_mid"] = df["age"].map(AGE_MIDPOINT).astype(int)
    df["admission_type"] = df["admission_type_id"].map(ADMISSION_TYPE_LABELS).fillna("Unknown")
    df["admission_source"] = df["admission_source_id"].map(ADMISSION_SOURCE_LABELS).fillna("Unknown")
    df["model_text"] = [serialize_record(r) for r in df.to_dict("records")]
    return df


def build_cohort(raw: pd.DataFrame, limit_patients: int | None, seed: int) -> pd.DataFrame:
    df = raw[raw["gender"].isin(["Male", "Female"])].copy()
    for col in ("race", "medical_specialty", "payer_code"):
        df[col] = df[col].fillna("Unknown")
    for col in ("max_glu_serum", "A1Cresult"):
        df[col] = df[col].replace("", "None").fillna("None")

    # encounter_id is used as the time order of a patient's admissions.
    df = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
    g = df.groupby("patient_nbr", sort=False)
    df["visit_index"] = g.cumcount() + 1
    df["n_visits_patient"] = g["encounter_id"].transform("size")
    df = df[df["n_visits_patient"] >= 2].copy()

    g = df.groupby("patient_nbr", sort=False)
    df["next_encounter_id"] = g["encounter_id"].shift(-1)
    df["next_diag_1"] = g["diag_1"].shift(-1)
    df = df[df["next_encounter_id"].notna()].copy()  # drop final visit of every patient
    df["next_encounter_id"] = df["next_encounter_id"].astype("int64")

    df["current_pathway"] = df["diag_1"].map(map_icd9_to_pathway)
    df["true_next_pathway"] = df["next_diag_1"].map(map_icd9_to_pathway)

    if limit_patients:
        rng = np.random.default_rng(seed)
        pids = df["patient_nbr"].unique()
        keep = rng.choice(pids, size=min(limit_patients, len(pids)), replace=False)
        df = df[df["patient_nbr"].isin(set(keep))].copy()
        log(f"  [debug] --limit-patients {limit_patients}: kept {len(df)} encounters")

    df = add_model_input_fields(df)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Representations
# ---------------------------------------------------------------------------
class ModelAEncoder:
    """Transparent clinical-feature vector.

    Numeric fields -> StandardScaler (unit variance). Categorical fields -> one-hot, left
    unscaled, so every categorical field contributes exactly one unit of squared norm, the
    same as a z-scored numeric field. (Z-scoring one-hot columns would weight rare
    categories by ~1/sqrt(p) and let e.g. a rare specialty dominate the cosine distance.)
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, df: pd.DataFrame) -> "ModelAEncoder":
        self.scaler.fit(df[MODEL_A_NUMERIC].astype(float).values)
        self.onehot.fit(df[MODEL_A_CATEGORICAL].astype(str).values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self.scaler.transform(df[MODEL_A_NUMERIC].astype(float).values)
        cat = self.onehot.transform(df[MODEL_A_CATEGORICAL].astype(str).values)
        return np.hstack([num, cat]).astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        cats = [
            f"{field}={value}"
            for field, values in zip(MODEL_A_CATEGORICAL, self.onehot.categories_)
            for value in values
        ]
        return MODEL_A_NUMERIC + cats


class BioClinicalBertEncoder:
    """Mean-pooled last-layer Bio_ClinicalBERT embeddings.

    Deterministic batching: texts are de-duplicated, sorted by token length and encoded in
    fixed-size batches (fp32, eval mode, no dropout). On CPU the batcher additionally pins
    the thread count and enables ``torch.use_deterministic_algorithms``. There is no
    fallback encoder: any import / download / shape problem raises.
    """

    def __init__(self, model_name: str, device: str, batch_size: int | None,
                 max_length: int, cpu_threads: int | None, cache_dir: Path) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Model B needs `torch` and `transformers` (see environment.yml). "
                "There is deliberately no fallback encoder."
            ) from exc
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False.")
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size or (256 if device == "cuda" else 32)
        self.cache_dir = cache_dir
        self.model_name = model_name

        torch.manual_seed(0)
        if device == "cpu":
            torch.set_num_threads(cpu_threads or max(1, (os.cpu_count() or 2) // 2))
            torch.use_deterministic_algorithms(True)
            self.device_name = f"cpu ({torch.get_num_threads()} threads)"
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.device_name = torch.cuda.get_device_name(0)

        log(f"  loading {model_name} on {self.device_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        hidden = int(self.model.config.hidden_size)
        if hidden != MODEL_B_DIM:
            raise RuntimeError(f"{model_name} has hidden size {hidden}, expected {MODEL_B_DIM}.")
        self.n_truncated = 0
        self.max_tokens_seen = 0

    def _cache_path(self, texts: list[str]) -> Path:
        h = hashlib.sha256()
        h.update(f"{self.model_name}|{self.max_length}|mean|v1".encode())
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\x00")
        return self.cache_dir / f"emb_{h.hexdigest()[:20]}.npy"

    def encode(self, texts: list[str], desc: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, MODEL_B_DIM), dtype=np.float32)
        uniq, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
        uniq = [str(t) for t in uniq]
        cache = self._cache_path(uniq)
        if cache.exists():
            log(f"  [{desc}] {len(texts)} texts: cached embeddings {cache.name}")
            return np.load(cache)[inverse]

        torch = self.torch
        lengths = [len(ids) for ids in self.tokenizer(uniq, add_special_tokens=True)["input_ids"]]
        self.max_tokens_seen = max(self.max_tokens_seen, max(lengths))
        self.n_truncated += int(sum(n > self.max_length for n in lengths))
        order = np.argsort(lengths, kind="stable")
        out = np.zeros((len(uniq), MODEL_B_DIM), dtype=np.float32)

        t0 = time.time()
        n_batches = (len(uniq) + self.batch_size - 1) // self.batch_size
        with torch.inference_mode():
            for b in range(n_batches):
                idx = order[b * self.batch_size:(b + 1) * self.batch_size]
                enc = self.tokenizer(
                    [uniq[i] for i in idx], padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out[idx] = pooled.float().cpu().numpy()
                if (b + 1) % 50 == 0 or b + 1 == n_batches:
                    rate = (b + 1) * self.batch_size / max(time.time() - t0, 1e-6)
                    log(f"  [{desc}] batch {b + 1}/{n_batches} ({rate:,.0f} texts/s)")
        if not np.isfinite(out).all():
            raise RuntimeError("Non-finite values in Model B embeddings.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
        return out[inverse]


# ---------------------------------------------------------------------------
# 5. Retrieval
# ---------------------------------------------------------------------------
def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def cosine_knn_other_patients(
    index_x: np.ndarray, index_patient: np.ndarray,
    query_x: np.ndarray, query_patient: np.ndarray,
    k: int, chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine KNN. Neighbors of the same patient (incl. the query itself) are skipped.

    Ties are broken by index position, so results are deterministic.
    Returns (neighbor_positions [n, k], cosine_distances [n, k]).
    """
    max_same = int(pd.Series(index_patient).value_counts().max())
    n_cand = min(k + max_same, len(index_x))
    ix = _l2_normalize(index_x)
    qx = _l2_normalize(query_x)
    nbr = np.empty((len(qx), k), dtype=np.int64)
    dist = np.empty((len(qx), k), dtype=np.float32)
    for start in range(0, len(qx), chunk):
        stop = min(start + chunk, len(qx))
        d = 1.0 - qx[start:stop] @ ix.T
        cand = np.argpartition(d, n_cand - 1, axis=1)[:, :n_cand]
        cd = np.take_along_axis(d, cand, axis=1)
        o = np.lexsort((cand, cd), axis=1)  # primary: distance, secondary: index
        cand = np.take_along_axis(cand, o, axis=1)
        cd = np.take_along_axis(cd, o, axis=1)
        same = index_patient[cand] == query_patient[start:stop, None]
        keep = np.argsort(same, axis=1, kind="stable")[:, :k]
        if np.take_along_axis(same, keep, axis=1).any():
            raise RuntimeError("Not enough other-patient candidates; increase n_cand.")
        nbr[start:stop] = np.take_along_axis(cand, keep, axis=1)
        dist[start:stop] = np.take_along_axis(cd, keep, axis=1)
    return nbr, np.clip(dist, 0.0, 2.0)


def vote(neighbor_labels: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Majority vote over [n, k] integer labels; ties go to the class seen at the best rank."""
    n, k = neighbor_labels.shape
    counts = np.zeros((n, n_classes), dtype=np.int64)
    first = np.full((n, n_classes), k, dtype=np.int64)
    rows = np.arange(n)
    for r in range(k - 1, -1, -1):
        lab = neighbor_labels[:, r]
        counts[rows, lab] += 1
        first[rows, lab] = r
    score = counts * (k + 1) - first
    pred = score.argmax(axis=1)
    return pred, counts[rows, pred] / k


def random_neighbors(patient: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(patient)
    max_same = int(pd.Series(patient).value_counts().max())
    out = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        cand = rng.choice(n, size=min(n, k + max_same), replace=False)
        cand = cand[patient[cand] != patient[i]][:k]
        out[i] = cand
    return out


def summarize(y_true: np.ndarray, y_pred: np.ndarray, agreement: np.ndarray,
              hit: np.ndarray | None) -> dict:
    recall = {}
    for c, name in enumerate(PATHWAYS):
        m = y_true == c
        recall[name] = round(float((y_pred[m] == c).mean()), 4) if m.any() else None
    agree_by = {
        name: round(float(agreement[y_true == c].mean()), 4) if (y_true == c).any() else None
        for c, name in enumerate(PATHWAYS)
    }
    out = {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_recall": round(float(np.mean([v for v in recall.values() if v is not None])), 4),
        "per_pathway_recall": recall,
        "mean_agreement": round(float(agreement.mean()), 4),
        "per_pathway_mean_agreement": agree_by,
        "predicted_pathway_counts": {
            name: int((y_pred == c).sum()) for c, name in enumerate(PATHWAYS)
        },
    }
    if hit is not None:
        out["hit_at_k"] = round(float(hit.mean()), 4)
    return out


# ---------------------------------------------------------------------------
# 6. Counterfactual edits (query side only)
# ---------------------------------------------------------------------------
def apply_edit(df: pd.DataFrame, edit: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (edited copy, mask of rows where the edit actually changes the record)."""
    e = df.copy()
    if edit == "sex_swap":
        e["gender"] = e["gender"].map({"Male": "Female", "Female": "Male"})
        applied = np.ones(len(e), dtype=bool)
    elif edit == "age_shift":
        pos = e["age"].map({b: i for i, b in enumerate(AGE_BRACKETS)}).to_numpy()
        new_pos = np.minimum(pos + 2, len(AGE_BRACKETS) - 1)  # +20 years, capped at [90-100)
        e["age"] = [AGE_BRACKETS[p] for p in new_pos]
        applied = new_pos != pos
    elif edit == "hba1c_elevation":
        applied = e["A1Cresult"].isin(["None", "Norm"]).to_numpy()
        e.loc[applied, "A1Cresult"] = ">8"
    else:
        raise ValueError(edit)
    e = add_model_input_fields(e.drop(columns=["age_mid", "admission_type",
                                               "admission_source", "model_text"]))
    return e, applied


def neighbor_flips(orig: np.ndarray, new: np.ndarray) -> np.ndarray:
    k = orig.shape[1]
    overlap = (orig[:, :, None] == new[:, None, :]).any(axis=2).sum(axis=1)
    return k - overlap


# ---------------------------------------------------------------------------
# 7. Projection
# ---------------------------------------------------------------------------
def umap_2d(x: np.ndarray, seed: int, n_neighbors: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("umap-learn is required (see environment.yml).") from exc
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
                        metric="cosine", random_state=seed)
    return reducer.fit_transform(x).astype(np.float32)


# ---------------------------------------------------------------------------
# 8. Export
# ---------------------------------------------------------------------------
def write_table(df: pd.DataFrame, out_dir: Path, name: str, list_cols: list[str]) -> None:
    df.to_parquet(out_dir / f"{name}.parquet", index=False)
    csv = df.copy()
    for col in list_cols:
        csv[col] = csv[col].map(lambda v: json.dumps([x.item() if hasattr(x, "item") else x
                                                       for x in v]))
    csv.to_csv(out_dir / f"{name}.csv", index=False)
    log(f"  wrote {name}.parquet / .csv ({len(df):,} rows, {df.shape[1]} columns)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--batch-size", type=int, default=None, help="default: 256 on GPU, 32 on CPU")
    p.add_argument("--cpu-threads", type=int, default=None)
    p.add_argument("--max-length", type=int, default=128, help="token limit for Model B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--model-name", default=MODEL_B_NAME)
    p.add_argument("--uci-csv", type=Path, default=None, help="use a manually downloaded diabetic_data.csv")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--limit-patients", type=int, default=None,
                   help="DEBUG ONLY: random subset of patients for a quick smoke test")
    args = p.parse_args()

    t_start = time.time()
    base = Path(__file__).resolve().parent
    cache_dir = base / "data_cache"
    out_dir = base / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    k = args.k

    # 1. Data ---------------------------------------------------------------------------
    log("[1/7] UCI Diabetes 130-US Hospitals (ID 296)")
    csv_path = args.uci_csv or cache_dir / "diabetic_data.csv"
    if not csv_path.exists():
        if args.uci_csv:
            raise FileNotFoundError(csv_path)
        csv_path = download_uci(cache_dir)
    log(f"  using {csv_path}")
    if args.download_only:
        return
    raw = load_uci_raw(csv_path)
    log(f"  raw: {len(raw):,} encounters, {raw['patient_nbr'].nunique():,} patients")

    # 2. Cohort -------------------------------------------------------------------------
    log("[2/7] Repeat-admission cohort, label = next admission diag_1 pathway")
    cohort = build_cohort(raw, args.limit_patients, args.seed)
    pid = cohort["patient_nbr"].to_numpy()
    y = cohort["true_next_pathway"].map({c: i for i, c in enumerate(PATHWAYS)}).to_numpy()
    class_counts = {c: int((y == i).sum()) for i, c in enumerate(PATHWAYS)}
    log(f"  {len(cohort):,} query encounters, {cohort['patient_nbr'].nunique():,} patients")
    for c, n in class_counts.items():
        log(f"    {c:<24} {n:>7,}  ({n / len(cohort):.1%})")

    # 3. Representations ----------------------------------------------------------------
    log("[3/7] Model A: StandardScaler(numeric) + one-hot(categorical)")
    enc_a = ModelAEncoder().fit(cohort)
    xa = enc_a.transform(cohort)
    log(f"  Model A matrix {xa.shape}")

    log(f"[4/7] Model B: {args.model_name}, mean-pooled last layer")
    enc_b = BioClinicalBertEncoder(args.model_name, args.device, args.batch_size,
                                   args.max_length, args.cpu_threads, cache_dir / "embeddings")
    xb = enc_b.encode(cohort["model_text"].tolist(), "original")
    log(f"  Model B matrix {xb.shape}; longest text {enc_b.max_tokens_seen} tokens, "
        f"{enc_b.n_truncated} truncated at {args.max_length}")

    # 4. Retrieval ----------------------------------------------------------------------
    log(f"[5/7] Cosine KNN (k={k}, other patients only) + baselines")
    nbr_a, dist_a = cosine_knn_other_patients(xa, pid, xa, pid, k)
    nbr_b, dist_b = cosine_knn_other_patients(xb, pid, xb, pid, k)
    nbr_r = random_neighbors(pid, k, args.seed)

    results = {}
    for name, nbr in (("model_a", nbr_a), ("model_b", nbr_b), ("random", nbr_r)):
        labs = y[nbr]
        pred, share = vote(labs, len(PATHWAYS))
        results[name] = dict(
            nbr=nbr, pred=pred, share=share,
            agreement=(labs == y[:, None]).mean(axis=1),
            hit=(labs == y[:, None]).any(axis=1),
        )
    majority_class = int(np.bincount(y, minlength=len(PATHWAYS)).argmax())
    pred_major = np.full(len(y), majority_class)

    enc_ids = cohort["encounter_id"].to_numpy()
    to_ids = lambda m: [enc_ids[row].tolist() for row in m]  # noqa: E731
    overlap_ab = (nbr_a[:, :, None] == nbr_b[:, None, :]).any(axis=2).sum(axis=1) / k
    retrieval = pd.DataFrame({
        "encounter_id": enc_ids,
        "patient_nbr": pid,
        "true_next_pathway": cohort["true_next_pathway"].to_numpy(),
        "model_a_neighbors": to_ids(nbr_a),
        "model_a_distances": [np.round(r, 5).tolist() for r in dist_a],
        "model_a_agreement": results["model_a"]["agreement"],
        "model_a_pred_pathway": [PATHWAYS[i] for i in results["model_a"]["pred"]],
        "model_a_vote_share": results["model_a"]["share"],
        "model_a_correct": results["model_a"]["pred"] == y,
        "model_b_neighbors": to_ids(nbr_b),
        "model_b_distances": [np.round(r, 5).tolist() for r in dist_b],
        "model_b_agreement": results["model_b"]["agreement"],
        "model_b_pred_pathway": [PATHWAYS[i] for i in results["model_b"]["pred"]],
        "model_b_vote_share": results["model_b"]["share"],
        "model_b_correct": results["model_b"]["pred"] == y,
        "agreement_disparity_b_minus_a": results["model_b"]["agreement"] - results["model_a"]["agreement"],
        "ab_neighbor_overlap": overlap_ab,
        "random_neighbors": to_ids(nbr_r),
        "random_agreement": results["random"]["agreement"],
        "random_pred_pathway": [PATHWAYS[i] for i in results["random"]["pred"]],
        "majority_pred_pathway": PATHWAYS[majority_class],
    })
    for name in ("model_a", "model_b"):
        log(f"  {name}: accuracy {(results[name]['pred'] == y).mean():.3f}, "
            f"mean agreement {results[name]['agreement'].mean():.3f}")
    log(f"  majority ({PATHWAYS[majority_class]}): accuracy {(pred_major == y).mean():.3f}")
    log(f"  random neighbors: mean agreement {results['random']['agreement'].mean():.3f}")

    # 5. Counterfactuals ------------------------------------------------------------------
    log("[6/7] Counterfactual neighbor flips (query edited, index unchanged)")
    flips = pd.DataFrame({"encounter_id": enc_ids, "patient_nbr": pid})
    flip_summary = {"model_a": {}, "model_b": {}, "n_applicable": {}}
    for edit in EDITS:
        edited, applied = apply_edit(cohort, edit)
        flips[f"{edit}_applied"] = applied
        flip_summary["n_applicable"][edit] = int(applied.sum())
        rows = np.flatnonzero(applied)
        sub = edited.iloc[rows]
        feats = {
            "model_a": (enc_a.transform(sub), xa, nbr_a, results["model_a"]["pred"]),
            "model_b": (enc_b.encode(sub["model_text"].tolist(), edit), xb, nbr_b,
                        results["model_b"]["pred"]),
        }
        for model, (qx, ix, orig_nbr, orig_pred) in feats.items():
            new_nbr, _ = cosine_knn_other_patients(ix, pid, qx, pid[rows], k)
            f = np.full(len(cohort), np.nan)
            f[rows] = neighbor_flips(orig_nbr[rows], new_nbr)
            new_pred, _ = vote(y[new_nbr], len(PATHWAYS))
            changed = pd.array([pd.NA] * len(cohort), dtype="boolean")
            changed[rows] = new_pred != orig_pred[rows]
            flips[f"{model}_flips_{edit}"] = pd.array(f, dtype="Int64")
            flips[f"{model}_pred_changed_{edit}"] = changed
            flip_summary[model][edit] = round(float(np.nanmean(f)), 4)
            log(f"  {model} {edit:<16} mean flips {np.nanmean(f):.2f} / {k} "
                f"(n={len(rows):,}), prediction changed {np.mean(new_pred != orig_pred[rows]):.1%}")

    # 6. Projection ---------------------------------------------------------------------
    log("[7/7] UMAP of input features + export")
    xy = umap_2d(xa, args.seed, args.umap_neighbors)
    cohort["umap_x"] = xy[:, 0]
    cohort["umap_y"] = xy[:, 1]

    lead = ["encounter_id", "patient_nbr", "visit_index", "n_visits_patient", "next_encounter_id"]
    tail = ["admission_type", "admission_source", "age_mid", "current_pathway", "next_diag_1",
            "true_next_pathway", "model_text", "umap_x", "umap_y"]
    raw_cols = [c for c in cohort.columns if c not in lead + tail]
    cohort_out = cohort[lead + raw_cols + tail]

    write_table(cohort_out, out_dir, "cohort_encounters", [])
    write_table(retrieval, out_dir, "retrieval_results",
                ["model_a_neighbors", "model_a_distances", "model_b_neighbors",
                 "model_b_distances", "random_neighbors"])
    write_table(flips, out_dir, "counterfactual_flips", [])

    import sklearn
    import torch
    import transformers

    shares = np.bincount(y, minlength=len(PATHWAYS)) / len(y)
    metrics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "runtime_seconds": round(time.time() - t_start, 1),
        "data_source": {
            "name": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
            "uci_id": UCI_DATASET_ID,
            "url": UCI_ZIP_URL,
            "raw_rows": int(len(raw)),
            "raw_patients": int(raw["patient_nbr"].nunique()),
            "csv_sha256": sha256_of(csv_path),
            "debug_limit_patients": args.limit_patients,
        },
        "cohort": {
            "definition": "patients with >= 2 encounters; final visit dropped; label = pathway "
                          "of next admission diag_1; visits ordered by encounter_id",
            "n_encounters": int(len(cohort)),
            "n_patients": int(cohort["patient_nbr"].nunique()),
            "class_counts": class_counts,
            "class_share": {c: round(float(s), 4) for c, s in zip(PATHWAYS, shares)},
        },
        "config": {
            "k": k,
            "seed": args.seed,
            "neighbor_rule": "cosine distance; neighbors from other patients only",
            "vote_tie_break": "class of the highest-ranked neighbor among tied classes",
            "text_template": TEXT_TEMPLATE,
            "model_a": {
                "numeric_standardized": MODEL_A_NUMERIC,
                "categorical_one_hot": MODEL_A_CATEGORICAL,
                "n_features": int(xa.shape[1]),
            },
            "model_b": {
                "name": args.model_name,
                "pooling": "attention-masked mean of last hidden layer",
                "embedding_dim": int(xb.shape[1]),
                "max_length": args.max_length,
                "longest_text_tokens": int(enc_b.max_tokens_seen),
                "n_truncated_texts": int(enc_b.n_truncated),
                "device": enc_b.device_name,
                "batch_size": enc_b.batch_size,
            },
            "umap": {"input": "Model A feature matrix (input features only)", "metric": "cosine",
                     "n_neighbors": args.umap_neighbors, "min_dist": 0.1, "random_state": args.seed},
            "edits": {
                "sex_swap": "Male <-> Female (all encounters)",
                "age_shift": "+20 years = +2 age brackets, capped at [90-100); "
                             "not applicable to [90-100)",
                "hba1c_elevation": "A1Cresult None/Norm -> '>8'; not applicable to '>7'/'>8'",
            },
        },
        "models": {
            "model_a": summarize(y, results["model_a"]["pred"], results["model_a"]["agreement"],
                                 results["model_a"]["hit"]),
            "model_b": summarize(y, results["model_b"]["pred"], results["model_b"]["agreement"],
                                 results["model_b"]["hit"]),
            "majority_baseline": {
                **summarize(y, pred_major, (pred_major == y).astype(float), None),
                "majority_class": PATHWAYS[majority_class],
            },
            "random_neighbor_baseline": {
                **summarize(y, results["random"]["pred"], results["random"]["agreement"],
                            results["random"]["hit"]),
                "expected_agreement_sum_p_squared": round(float((shares ** 2).sum()), 4),
            },
        },
        "model_ab_mean_neighbor_overlap": round(float(overlap_ab.mean()), 4),
        "counterfactual_mean_flips": flip_summary,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    with open(out_dir / "baseline_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log("  wrote baseline_metrics.json")

    np.savez_compressed(
        cache_dir / "phase2_model_internals.npz",
        encounter_id=enc_ids,
        model_a_features=xa,
        model_a_feature_names=np.array(enc_a.feature_names),
        model_b_embeddings=xb,
    )
    log("  wrote data_cache/phase2_model_internals.npz (Phase II only, not committed)")
    log(f"Done in {time.time() - t_start:,.0f} s.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # make the failure explicit, never silently degrade
        print(f"\n[FAILED] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
