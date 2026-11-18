"""Shared helpers for Phase II, Pair B (AI-aided): opening Model B (Bio_ClinicalBERT).

Model B as used in Phase I (shared/01_prepare_baseline_data.py):
record -> text template -> Bio_ClinicalBERT -> mean-pooled last layer (768-D) -> cosine KNN, k = 10,
neighbours of the same patient excluded.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "shared" / "data"
CACHE = ROOT / "shared" / "data_cache"          # gitignored: model internals never go into the repo
FIG = Path(__file__).resolve().parents[1] / "figures"
OUT = Path(__file__).resolve().parents[1] / "outputs"   # gitignored

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 128
K = 10

# Field name in the text template -> column in the cohort table (order = order in the text)
FIELDS = {
    "Sex": "gender", "Age": "age", "Hospital Stay": "time_in_hospital",
    "Admission Type": "admission_type", "Source": "admission_source",
    "Specialty": "medical_specialty", "Payer": "payer_code", "Labs": "num_lab_procedures",
    "Medications": "num_medications", "Diagnoses count": "number_diagnoses",
    "Max Glucose": "max_glu_serum", "A1C": "A1Cresult", "Insulin": "insulin",
    "Med Change": "change", "Diabetes Med": "diabetesMed",
}
TEMPLATE_UNITS = {"Hospital Stay": " days"}

# ---- visual style: the same colours and look as Pair A's Phase I dashboard ---------------------
# (phase1/pair_A_ai_aided/dashboard/data_layer.py and figures.py), so Phase I and II read as one project.
PATHWAY_COLORS = {"Kidney / Renal": "#8E6CBF", "Cardiovascular": "#D55E00",
                  "Eye / Neurological": "#0072B2", "Uncomplicated / Other": "#A0A0A0"}
PATHWAY_ORDER = ["Uncomplicated / Other", "Cardiovascular", "Kidney / Renal", "Eye / Neurological"]
A1C_COLORS = {"None": "#BDBDBD", "Norm": "#009E73", ">7": "#E69F00", ">8": "#B2182B"}
MODEL_COLORS = {"model_a": "#457B9D", "model_b": "#E63946"}
QUALITATIVE = ["#2A9D8F", "#E76F51", "#457B9D", "#F4A261", "#E63946", "#8E6CBF", "#52B788", "#DDA15E", "#BC6C25"]
INK, NAVY, MUTED, LIGHT, GRIDC = "#212529", "#1D3557", "#495057", "#ADB5BD", "#F1F3F5"
VALUE_C, TEMPLATE_C = MODEL_COLORS["model_b"], LIGHT       # patient-value tokens vs. template tokens


def apply_style() -> None:
    """Matplotlib look matching Pair A's Plotly figures: white, light grid, system sans-serif, 12 pt."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "figure.titlesize": 14,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": LIGHT, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 1.0, "axes.axisbelow": True,
        "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(color=QUALITATIVE), "legend.frameon": False,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def load_cohort() -> pd.DataFrame:
    cohort = pd.read_parquet(DATA / "cohort_encounters.parquet")
    retrieval = pd.read_parquet(DATA / "retrieval_results.parquet")
    flips = pd.read_parquet(DATA / "counterfactual_flips.parquet")
    df = (cohort
          .merge(retrieval.drop(columns=["patient_nbr", "true_next_pathway"]), on="encounter_id", validate="one_to_one")
          .merge(flips.drop(columns=["patient_nbr"]), on="encounter_id", validate="one_to_one"))
    return df


def render(values: dict[str, str]) -> str:
    """Build the Model B text from {field label: value}; fields missing from `values` are dropped."""
    parts = [f"{label}: {values[label]}{TEMPLATE_UNITS.get(label, '')}" for label in FIELDS if label in values]
    return ", ".join(parts) + "."


def record_values(row) -> dict[str, str]:
    return {label: str(row[col]) for label, col in FIELDS.items()}


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def knn_other_patients(x: np.ndarray, patient: np.ndarray, k: int = K, chunk: int = 2048):
    """Exact cosine KNN, same-patient stays excluded (as in Phase I)."""
    xn = l2n(x.astype(np.float32))
    n = len(xn)
    max_same = int(pd.Series(patient).value_counts().max())
    n_cand = k + max_same
    nbr = np.empty((n, k), dtype=np.int64)
    dist = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = 1.0 - xn[s:e] @ xn.T
        cand = np.argpartition(d, n_cand - 1, axis=1)[:, :n_cand]
        cd = np.take_along_axis(d, cand, axis=1)
        o = np.lexsort((cand, cd), axis=1)
        cand, cd = np.take_along_axis(cand, o, axis=1), np.take_along_axis(cd, o, axis=1)
        same = patient[cand] == patient[s:e, None]
        keep = np.argsort(same, axis=1, kind="stable")[:, :k]
        nbr[s:e] = np.take_along_axis(cand, keep, axis=1)
        dist[s:e] = np.take_along_axis(cd, keep, axis=1)
    return nbr, dist


class BertEncoder:
    """Bio_ClinicalBERT with mean pooling over all 13 hidden layers (0 = token embeddings, 12 = last)."""

    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        # CPU on purpose: Phase I ran on CPU, and torch's MPS backend crashes on macOS 14.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(self.device).eval()

    def encode(self, texts: list[str], layers: str = "last", batch_size: int = 64) -> np.ndarray:
        """layers='last' -> [n, 768]; layers='all' -> [n, 13, 768] (float16)."""
        torch = self.torch
        lengths = [len(t) for t in texts]
        order = np.argsort(lengths, kind="stable")
        out = None
        with torch.inference_mode():
            for b in range(0, len(texts), batch_size):
                idx = order[b:b + batch_size]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
                hs = self.model(**enc).hidden_states
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = [((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy() for h in hs]
                if layers == "last":
                    arr = pooled[-1]
                    if out is None:
                        out = np.zeros((len(texts), arr.shape[-1]), dtype=np.float32)
                else:
                    arr = np.stack(pooled, axis=1).astype(np.float16)
                    if out is None:
                        out = np.zeros((len(texts), len(hs), arr.shape[-1]), dtype=np.float16)
                out[idx] = arr
        return out


def token_parts(text: str, offsets: list, special: list) -> list[tuple[str, str]]:
    """(field, part) per token: part = 'value' (patient's value), 'template' (field name, ':', ',')
    or 'special' ([CLS]/[SEP]). Template tokens are assigned to the field whose name they spell."""
    spans = field_spans(text)
    out = []
    for (s, e), sp in zip(offsets, special):
        if sp:
            out.append(("[special]", "special"))
            continue
        field, part = "[template]", "template"
        for label, (fs, fe) in spans.items():
            if s >= fs and e <= fe:
                field, part = label, "value"
                break
        if part == "template":
            for label, (fs, fe) in spans.items():
                if e <= fs and s >= fs - len(label) - 2:
                    field = label
        out.append((field, part))
    return out


def field_spans(text: str) -> dict[str, tuple[int, int]]:
    """Character span of each field's value inside a Model B text."""
    spans = {}
    for label in FIELDS:
        m = re.search(re.escape(label) + r": ([^,]*?)(?:, (?=[A-Z][A-Za-z0-9 ]+: )|\.$)", text)
        if m:
            spans[label] = m.span(1)
    return spans
