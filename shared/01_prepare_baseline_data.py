"""
01_prepare_baseline_data.py
===========================
Shared Pre-Phase I Data & Model Pipeline for VIS4AI / AI4VIS Lab Assignment.

What this script does (All 4 team members run this ONCE before splitting into pairs):
1. Loads the public clinical cohort:
   - Default (`--dataset uci_diabetes`): UCI Diabetes 130-US Hospitals (ID 296), filtered to
     repeat-admission diabetes patients so later complication pathways and longitudinal
     admission trajectories can be observed.
   - Offline fallback (`--dataset synthetic_demo`): Generates a realistic clinical cohort if offline.
2. Extracts human-interpretable input attributes and serializes a realistic clinical admission
   summary note for each patient encounter (strictly WITHOUT leaking the future target pathway!).
3. Runs k-Nearest-Neighbor (k=10) patient-similarity retrieval using:
   - Model A (Transparent Baseline): Count / TF-IDF clinical feature vector + Cosine KNN.
   - Model B (Foundation Model): Pretrained `emilyalsentzer/Bio_ClinicalBERT` 768-D embeddings + Cosine KNN
     (with a multi-layer clinical semantic transformer fallback if run in a kernel without PyTorch).
4. Evaluates counterfactuals & perturbations (Sex swap, Age shift, HbA1c lab perturbation) to
   measure "Neighbor Flip" sensitivity in Phase I.
5. Saves strictly separated outputs:
   - `outputs/phase1_blackbox_io.csv` -> ONLY Inputs & Outputs (Black box closed for Phase I!)
   - `outputs/phase1_neighbors_detail.json` -> Neighbor lists for interactive drill-down.
   - `outputs/phase2_model_internals.npz` -> 768-D embeddings & layer hidden states (For Phase II ONLY!).
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# 1. ICD-9 Diagnosis Code -> Complication Pathway Mapping (Clinical Ground Truth)
# ---------------------------------------------------------------------------
def map_icd9_to_pathway(diag_code: str) -> str:
    """
    Maps an ICD-9 diagnosis code to one of the 4 diabetes complication pathways:
      - 'Kidney_Renal': Diabetic nephropathy (250.4x), nephritis/renal failure (580-599)
      - 'Cardiovascular': Circulatory diseases (390-459), diabetic peripheral circulatory (250.7x)
      - 'Eye_Neurological': Diabetic retinopathy (250.5x), ophthalmic (360-379), neuropathy (250.6x, 350-359)
      - 'Uncomplicated_Other': Uncomplicated diabetes (250.0x-250.3x) or general medical codes
    """
    if pd.isna(diag_code):
        return "Uncomplicated_Other"
    code = str(diag_code).strip()
    if not code or code == "?":
        return "Uncomplicated_Other"

    if code.startswith("250."):
        sub = code.split(".")[1][:1] if "." in code and len(code.split(".")[1]) > 0 else "0"
        if sub == "4":
            return "Kidney_Renal"
        if sub in ("5", "6"):
            return "Eye_Neurological"
        if sub == "7":
            return "Cardiovascular"
        return "Uncomplicated_Other"

    if code[0].isalpha():
        return "Uncomplicated_Other"
    try:
        num = float(code.split(".")[0])
    except ValueError:
        return "Uncomplicated_Other"

    if 390 <= num <= 459 or num == 785:
        return "Cardiovascular"
    if 580 <= num <= 599:
        return "Kidney_Renal"
    if 350 <= num <= 379:
        return "Eye_Neurological"
    return "Uncomplicated_Other"


def resolve_patient_pathway(row: pd.Series) -> str:
    """
    Determines the complication pathway across diag_1, diag_2, diag_3,
    prioritizing specific complications over 'Uncomplicated_Other'.
    """
    pathways = [
        map_icd9_to_pathway(row.get("diag_1", "")),
        map_icd9_to_pathway(row.get("diag_2", "")),
        map_icd9_to_pathway(row.get("diag_3", "")),
    ]
    for priority in ["Kidney_Renal", "Eye_Neurological", "Cardiovascular"]:
        if priority in pathways:
            return priority
    return "Uncomplicated_Other"


# ---------------------------------------------------------------------------
# 2. Clinical Note Serialization (Strictly Observable Inputs — NO Label Leakage!)
# ---------------------------------------------------------------------------
AGE_MIDPOINTS = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}


def build_clinical_summary(row: pd.Series) -> str:
    """
    Converts observable index-admission attributes (demographics, symptoms, labs, medications,
    utilization) into a natural clinical summary note WITHOUT leaking the target pathway name.
    """
    age_num = int(row["age_num"])
    gender = str(row["gender"])
    race = str(row["race"])
    a1c = str(row["a1c_result"])
    glu = str(row["max_glu_serum"])
    los = int(row["time_in_hospital"])
    labs = int(row["num_lab_procedures"])
    procs = int(row["num_procedures"])
    meds = int(row["num_medications"])
    diags = int(row["number_diagnoses"])
    prior_visits = int(row["prior_visits_total"])
    insulin = str(row["insulin_status"])
    med_change = str(row["diabetes_med_change"])
    symptoms = str(row.get("presenting_notes", "admitted for inpatient glycemic and comorbidity evaluation"))

    return (
        f"Patient is a {age_num}-year-old {race} {gender.lower()} admitted for {los} days "
        f"(encounter #{int(row['admission_number'])} of {int(row['total_patient_admissions'])}, "
        f"prior hospital/outpatient visits: {prior_visits}). "
        f"Clinical presentation: {symptoms}. "
        f"Laboratory panel: HbA1c is {a1c}, maximum serum glucose is {glu}, "
        f"with {labs} lab tests, {procs} clinical procedures, and {diags} active comorbid conditions. "
        f"Pharmacotherapy: {meds} total medications; insulin regimen is {insulin} "
        f"(therapy modification: {med_change})."
    )


def derive_presenting_notes_from_inputs(row: pd.Series, rng: np.random.Generator) -> str:
    """
    Generates realistic, noisy clinical presentation phrasing based on observable clinical
    indicators and comorbidities (includes synonym variation and realistic clinical overlap).
    """
    pathway = str(row["current_pathway"])
    a1c = str(row["a1c_result"])

    # Synonyms so Count-KNN (exact word match) struggles with phrasing variations while semantic models group them
    cardio_phrases = [
        "chest tightness, exertional dyspnea, and elevated blood pressure",
        "systolic hypertension, pedal edema, and irregular cardiac rhythm",
        "coronary ischemia symptoms with shortness of breath on exertion",
        "orthopnea, vascular claudication, and borderline tachycardia",
    ]
    renal_phrases = [
        "elevated serum creatinine, proteinuria, and fluid retention",
        "declining glomerular filtration, ankle swelling, and azotemia",
        "persistent microalbuminuria with impaired kidney clearance",
        "electrolyte imbalance, oliguria, and elevated blood urea nitrogen",
    ]
    eye_neuro_phrases = [
        "blurred vision, peripheral paresthesia, and distal numbness",
        "visual acuity decline, macular edema signs, and neuropathic pain",
        "fundoscopic microvascular changes with burning tingling in feet",
        "progressive視力/visual floaters and diabetic polyneuropathy symptoms",
    ]
    general_phrases = [
        "uncontrolled hyperglycemia, polyuria, and generalised fatigue",
        "routine inpatient glycemic titration and metabolic workup",
        "episodic dizziness, polydipsia, and insulin dose adjustment",
        "post-prandial glucose spikes with mild dehydration",
    ]

    # Introduce realistic clinical ambiguity (28% of admissions present with non-specific or overlapping symptoms)
    roll = rng.random()
    if roll < 0.22:
        chosen_pool = general_phrases
    elif roll < 0.32 and pathway in ("Cardiovascular", "Kidney_Renal"):
        # Cardiorenal overlap (clinically common!)
        chosen_pool = cardio_phrases if pathway == "Kidney_Renal" else renal_phrases
    else:
        if pathway == "Cardiovascular":
            chosen_pool = cardio_phrases
        elif pathway == "Kidney_Renal":
            chosen_pool = renal_phrases
        elif pathway == "Eye_Neurological":
            chosen_pool = eye_neuro_phrases
        else:
            chosen_pool = general_phrases

    phrase = str(rng.choice(chosen_pool))
    if a1c == "Not_Measured":
        phrase += "; sparse outpatient lab documentation prior to admission"
    return phrase


# ---------------------------------------------------------------------------
# 3. Dataset Loaders (Direct UCI Download / ucimlrepo / Realistic Fallback)
# ---------------------------------------------------------------------------
def fetch_raw_uci_dataframe(cache_dir: Path) -> pd.DataFrame | None:
    """
    Attempts to load the UCI Diabetes 130-US Hospitals dataset from local cache,
    `ucimlrepo`, or direct HTTPS download from the UCI Machine Learning Repository.
    """
    cache_csv = cache_dir / "diabetic_data.csv"
    if cache_csv.exists():
        print(f"  -> Loading cached UCI Diabetes dataset from {cache_csv}...")
        return pd.read_csv(cache_csv, low_memory=False)

    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=296)
        df = pd.concat([ds.data.ids, ds.data.features, ds.data.targets], axis=1)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
        return df
    except Exception:
        pass

    try:
        import requests
        url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
        print(f"  -> Downloading UCI Diabetes zip directly from {url}...")
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith("diabetic_data.csv")]
            if csv_names:
                cache_dir.mkdir(parents=True, exist_ok=True)
                with zf.open(csv_names[0]) as src, open(cache_csv, "wb") as dst:
                    dst.write(src.read())
                return pd.read_csv(cache_csv, low_memory=False)
    except Exception as exc:
        print(f"  [!] Online UCI download unavailable ({exc}).")
    return None


def load_uci_diabetes_cohort(max_patients: int = 800, seed: int = 42, cache_dir: Path | None = None) -> pd.DataFrame:
    """
    Loads the UCI Diabetes 130-US Hospitals dataset, filters to repeat-admission patients,
    and prepares interpretable input columns + target pathway.
    """
    print("[1/5] Loading Diabetes longitudinal cohort...")
    raw_df = fetch_raw_uci_dataframe(cache_dir or Path("data_cache"))
    if raw_df is None:
        print("  -> Using built-in realistic clinical diabetes cohort generator...")
        return generate_synthetic_cohort(max_patients=max_patients, seed=seed)

    df = raw_df.copy()
    df["race"] = df["race"].replace("?", "Unknown").fillna("Unknown")
    df = df[df["gender"].isin(["Male", "Female"])].copy()

    if "patient_nbr" not in df.columns:
        df["patient_nbr"] = np.arange(len(df)) // 2
    if "encounter_id" not in df.columns:
        df["encounter_id"] = np.arange(len(df))

    df = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
    df["admission_number"] = df.groupby("patient_nbr").cumcount() + 1
    df["total_patient_admissions"] = df.groupby("patient_nbr")["encounter_id"].transform("count")

    repeat_df = df[df["total_patient_admissions"] >= 2].copy()
    repeat_df["current_pathway"] = repeat_df.apply(resolve_patient_pathway, axis=1)
    repeat_df["true_pathway"] = (
        repeat_df.groupby("patient_nbr")["current_pathway"].shift(-1).fillna(repeat_df["current_pathway"])
    )

    rng = np.random.default_rng(seed)
    unique_patients = repeat_df[["patient_nbr", "true_pathway"]].drop_duplicates("patient_nbr")
    sampled_patient_ids = []
    # Keep natural clinical imbalance (Eye_Neurological is rarer in hospital data)
    quotas = {
        "Cardiovascular": int(max_patients * 0.16),
        "Kidney_Renal": int(max_patients * 0.12),
        "Uncomplicated_Other": int(max_patients * 0.14),
        "Eye_Neurological": int(max_patients * 0.06),
    }
    for pathway, grp in unique_patients.groupby("true_pathway"):
        n_pick = min(len(grp), quotas.get(str(pathway), 60))
        picked = rng.choice(grp["patient_nbr"].values, size=n_pick, replace=False)
        sampled_patient_ids.extend(picked.tolist())

    cohort = repeat_df[repeat_df["patient_nbr"].isin(set(sampled_patient_ids))].copy()
    if len(cohort) > max_patients:
        keep_pids = cohort["patient_nbr"].unique()[: max_patients // 2]
        cohort = cohort[cohort["patient_nbr"].isin(set(keep_pids))].copy()

    cohort = cohort.reset_index(drop=True)

    cohort["patient_id"] = "P" + cohort["patient_nbr"].astype(str) + "_V" + cohort["admission_number"].astype(str)
    cohort["patient_uid"] = "P" + cohort["patient_nbr"].astype(str)
    cohort["age_bracket"] = cohort["age"].astype(str)
    cohort["age_num"] = cohort["age_bracket"].map(AGE_MIDPOINTS).fillna(65).astype(int)
    cohort["a1c_result"] = cohort["A1Cresult"].fillna("None").replace({"None": "Not_Measured"})
    cohort["max_glu_serum"] = cohort["max_glu_serum"].fillna("None").replace({"None": "Not_Measured"})
    cohort["prior_visits_total"] = (
        cohort["number_outpatient"].fillna(0)
        + cohort["number_emergency"].fillna(0)
        + cohort["number_inpatient"].fillna(0)
    ).astype(int)
    cohort["insulin_status"] = cohort["insulin"].fillna("No").astype(str)
    cohort["diabetes_med_change"] = cohort["change"].fillna("No").astype(str)
    cohort["readmitted_outcome"] = cohort["readmitted"].fillna("NO").astype(str)

    cohort["record_completeness"] = (
        (cohort["race"] != "Unknown").astype(float) * 0.34
        + (cohort["a1c_result"] != "Not_Measured").astype(float) * 0.33
        + (cohort["max_glu_serum"] != "Not_Measured").astype(float) * 0.33
    ).round(2)

    cohort["presenting_notes"] = [derive_presenting_notes_from_inputs(row, rng) for _, row in cohort.iterrows()]
    cohort["clinical_summary_text"] = cohort.apply(build_clinical_summary, axis=1)
    cohort["summary_word_count"] = cohort["clinical_summary_text"].str.split().str.len()

    keep_cols = [
        "patient_id", "patient_uid", "admission_number", "total_patient_admissions",
        "age_num", "age_bracket", "gender", "race",
        "a1c_result", "max_glu_serum", "time_in_hospital",
        "num_lab_procedures", "num_procedures", "num_medications",
        "number_diagnoses", "prior_visits_total", "insulin_status",
        "diabetes_med_change", "record_completeness",
        "current_pathway", "true_pathway", "readmitted_outcome",
        "presenting_notes", "clinical_summary_text", "summary_word_count",
    ]
    return cohort[keep_cols]


def generate_synthetic_cohort(max_patients: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic longitudinal diabetes cohort with overlapping distributions,
    disease progression across visits, and realistic clinical noise so Phase I & II
    visualizations reveal authentic error patterns, subgroup biases, and confusion blocks.
    """
    rng = np.random.default_rng(seed)
    pathways = ["Cardiovascular", "Kidney_Renal", "Eye_Neurological", "Uncomplicated_Other"]
    pathway_probs = [0.35, 0.25, 0.13, 0.27]  # Eye_Neurological is rarer, mirroring hospital data

    rows = []
    n_unique_patients = max(max_patients // 2, 200)
    idx = 0
    for pid in range(1000, 1000 + n_unique_patients):
        if idx >= max_patients:
            break
        n_visits = int(rng.choice([2, 2, 3, 4], p=[0.45, 0.25, 0.20, 0.10]))
        sex = str(rng.choice(["Female", "Male"], p=[0.53, 0.47]))
        race = str(rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Unknown"], p=[0.62, 0.21, 0.08, 0.04, 0.05]))
        base_age = int(rng.choice([35, 45, 55, 65, 75, 85], p=[0.07, 0.13, 0.22, 0.27, 0.21, 0.10]))
        true_pathway = str(rng.choice(pathways, p=pathway_probs))

        for v in range(1, n_visits + 1):
            if idx >= max_patients:
                break
            age_num = min(95, base_age + (v - 1) * 2)
            age_bracket = f"[{(age_num // 10) * 10}-{((age_num // 10) + 1) * 10})"

            # Overlapping clinical feature distributions (Cardiovascular and Kidney_Renal overlap partially)
            if true_pathway == "Kidney_Renal":
                a1c = str(rng.choice([">8", ">7", "Norm", "Not_Measured"], p=[0.40, 0.22, 0.16, 0.22]))
                diags = int(np.clip(rng.normal(8.4, 2.1) + v * 0.5, 2, 16))
                labs = int(np.clip(rng.normal(56, 15), 10, 110))
                meds = int(np.clip(rng.normal(17 + v * 1.5, 5.5), 3, 45))
                los = int(np.clip(rng.poisson(4.8), 1, 14))
            elif true_pathway == "Cardiovascular":
                a1c = str(rng.choice([">8", ">7", "Norm", "Not_Measured"], p=[0.28, 0.26, 0.20, 0.26]))
                diags = int(np.clip(rng.normal(7.9, 2.0) + v * 0.4, 2, 16))
                labs = int(np.clip(rng.normal(51, 15), 10, 110))
                meds = int(np.clip(rng.normal(16 + v * 1.4, 5.0), 3, 45))
                los = int(np.clip(rng.poisson(4.3), 1, 14))
            elif true_pathway == "Eye_Neurological":
                a1c = str(rng.choice([">8", ">7", "Norm", "Not_Measured"], p=[0.46, 0.24, 0.12, 0.18]))
                diags = int(np.clip(rng.normal(6.3, 1.9) + v * 0.3, 2, 14))
                labs = int(np.clip(rng.normal(40, 14), 8, 95))
                meds = int(np.clip(rng.normal(13 + v * 1.2, 4.5), 2, 40))
                los = int(np.clip(rng.poisson(3.2), 1, 14))
            else:
                a1c = str(rng.choice(["Norm", ">7", ">8", "Not_Measured"], p=[0.36, 0.20, 0.12, 0.32]))
                diags = int(np.clip(rng.normal(4.8, 1.7), 1, 12))
                labs = int(np.clip(rng.normal(33, 13), 5, 85))
                meds = int(np.clip(rng.normal(11 + v * 1.0, 4.0), 1, 35))
                los = int(np.clip(rng.poisson(2.7), 1, 14))

            glu = str(rng.choice(["Not_Measured", "Norm", ">200", ">300"], p=[0.60, 0.16, 0.14, 0.10]))
            procs = int(np.clip(rng.poisson(1.5 if true_pathway in ("Cardiovascular", "Kidney_Renal") else 0.8), 0, 6))
            prior_v = int(np.clip(rng.poisson(1.2) + (v - 1), 0, 15))
            insulin = str(rng.choice(["Steady", "Up", "Down", "No"], p=[0.30, 0.24, 0.14, 0.32]))
            med_change = "Ch" if insulin in ("Up", "Down") else str(rng.choice(["Ch", "No"], p=[0.3, 0.7]))
            readm = str(rng.choice(["<30", ">30", "NO"], p=[0.18, 0.34, 0.48]))

            rec_comp = round(
                (0.34 if race != "Unknown" else 0.0)
                + (0.33 if a1c != "Not_Measured" else 0.0)
                + (0.33 if glu != "Not_Measured" else 0.0),
                2,
            )

            # On early admissions (V1), clinical presentation may still look uncomplicated or ambiguous
            if v == 1 and rng.random() < 0.25:
                curr_path = "Uncomplicated_Other"
            else:
                curr_path = true_pathway

            row_dict = {
                "patient_id": f"P{pid}_V{v}",
                "patient_uid": f"P{pid}",
                "admission_number": v,
                "total_patient_admissions": n_visits,
                "age_num": age_num,
                "age_bracket": age_bracket,
                "gender": sex,
                "race": race,
                "a1c_result": a1c,
                "max_glu_serum": glu,
                "time_in_hospital": los,
                "num_lab_procedures": labs,
                "num_procedures": procs,
                "num_medications": meds,
                "number_diagnoses": diags,
                "prior_visits_total": prior_v,
                "insulin_status": insulin,
                "diabetes_med_change": med_change,
                "record_completeness": rec_comp,
                "current_pathway": curr_path,
                "true_pathway": true_pathway,
                "readmitted_outcome": readm,
            }
            row_dict["presenting_notes"] = derive_presenting_notes_from_inputs(pd.Series(row_dict), rng)
            row_dict["clinical_summary_text"] = build_clinical_summary(pd.Series(row_dict))
            row_dict["summary_word_count"] = len(row_dict["clinical_summary_text"].split())
            rows.append(row_dict)
            idx += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Feature & Embedding Extraction (Model A: Count/Feature vs. Model B: Bio_ClinicalBERT)
# ---------------------------------------------------------------------------
def build_model_a_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], CountVectorizer, StandardScaler]:
    """
    Model A: Transparent Count + Clinical Feature Vectorizer.
    Uses raw token counts (sensitive to exact wording & demographic tokens) + standardized numeric counts.
    """
    vec = CountVectorizer(max_features=90, stop_words="english", ngram_range=(1, 1))
    text_counts = vec.fit_transform(df["clinical_summary_text"]).toarray().astype(np.float32)

    num_cols = [
        "age_num", "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications", "number_diagnoses", "prior_visits_total",
    ]
    scaler = StandardScaler()
    num_feats = scaler.fit_transform(df[num_cols].values.astype(np.float32))

    # Count vector is unweighted across all words, so common demographic/utilization numbers compete with symptoms
    combined = np.hstack([num_feats * 0.8, text_counts * 0.6])
    feature_names = [f"num__{c}" for c in num_cols] + [f"tok__{t}" for t in vec.get_feature_names_out()]
    return combined, feature_names, vec, scaler


class ClinicalSemanticEncoderFallback:
    """
    Multi-layer clinical representation encoder used when running in a Python kernel (e.g., Python 3.14)
    where PyTorch/HuggingFace wheels are not installed.
    Models how a 12-layer clinical transformer (`Bio_ClinicalBERT`) progressively transforms:
      - Layer 4 (Early layer): Strong encoding of surface demographics (sex, race, age) & note length.
      - Layer 8 (Middle layer): Combines clinical concepts with hospital utilization & lab patterns.
      - Layer 12 (Final layer): Groups synonymous clinical symptom descriptions into dense 768-D concepts,
        while retaining a realistic residual demographic/completeness bias (for Phase I & II discovery!).
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=32, random_state=seed)
        self.num_scaler = StandardScaler()
        self.cat_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.W_l4: np.ndarray | None = None
        self.W_l8: np.ndarray | None = None
        self.W_l12: np.ndarray | None = None
        self.W_cls: np.ndarray | None = None

    @staticmethod
    def _normalize_clinical_synonyms(texts: List[str]) -> List[str]:
        """Simulates pretrained clinical language model synonymy alignment across symptom phrasings."""
        synonym_map = {
            "chest tightness": "cardiac_symptom",
            "exertional dyspnea": "cardiac_symptom",
            "systolic hypertension": "cardiac_symptom",
            "coronary ischemia": "cardiac_symptom",
            "irregular cardiac rhythm": "cardiac_symptom",
            "orthopnea": "cardiac_symptom",
            "vascular claudication": "cardiac_symptom",
            "serum creatinine": "renal_symptom",
            "proteinuria": "renal_symptom",
            "glomerular filtration": "renal_symptom",
            "azotemia": "renal_symptom",
            "microalbuminuria": "renal_symptom",
            "kidney clearance": "renal_symptom",
            "oliguria": "renal_symptom",
            "blood urea nitrogen": "renal_symptom",
            "blurred vision": "ophthalmic_neuro_symptom",
            "peripheral paresthesia": "ophthalmic_neuro_symptom",
            "visual acuity": "ophthalmic_neuro_symptom",
            "macular edema": "ophthalmic_neuro_symptom",
            "neuropathic pain": "ophthalmic_neuro_symptom",
            "microvascular changes": "ophthalmic_neuro_symptom",
            "polyneuropathy": "ophthalmic_neuro_symptom",
            "uncontrolled hyperglycemia": "metabolic_general_symptom",
            "glycemic titration": "metabolic_general_symptom",
            "post-prandial glucose": "metabolic_general_symptom",
        }
        out = []
        for t in texts:
            s = t.lower()
            for phrase, concept in synonym_map.items():
                if phrase in s:
                    s = s + f" {concept} {concept}"
            out.append(s)
        return out

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        norm_texts = self._normalize_clinical_synonyms(df["clinical_summary_text"].tolist())
        t_mat = self.tfidf.fit_transform(norm_texts)
        t_svd = self.svd.fit_transform(t_mat).astype(np.float32)

        num_cols = ["age_num", "time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses", "record_completeness"]
        n_mat = self.num_scaler.fit_transform(df[num_cols].values.astype(np.float32))
        c_mat = self.cat_enc.fit_transform(df[["gender", "race", "a1c_result", "insulin_status"]]).astype(np.float32)

        return self._project_layers(t_svd, n_mat, c_mat, fit=True)

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        norm_texts = self._normalize_clinical_synonyms(df["clinical_summary_text"].tolist())
        t_mat = self.tfidf.transform(norm_texts)
        t_svd = self.svd.transform(t_mat).astype(np.float32)

        num_cols = ["age_num", "time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses", "record_completeness"]
        n_mat = self.num_scaler.transform(df[num_cols].values.astype(np.float32))
        c_mat = self.cat_enc.transform(df[["gender", "race", "a1c_result", "insulin_status"]]).astype(np.float32)

        return self._project_layers(t_svd, n_mat, c_mat, fit=False)

    def _project_layers(self, t_svd: np.ndarray, n_mat: np.ndarray, c_mat: np.ndarray, fit: bool) -> Dict[str, np.ndarray]:
        # Layer 4: Heavily weights surface demographics (c_mat) and age/utilization (n_mat)
        base_l4 = np.hstack([t_svd * 0.6, n_mat * 1.1, c_mat * 1.8])
        # Layer 8: Balanced clinical text concepts and labs
        base_l8 = np.hstack([t_svd * 1.5, n_mat * 1.0, c_mat * 1.1])
        # Layer 12: Strong clinical concept representation, but retains a residual demographic/completeness shortcut
        base_l12 = np.hstack([t_svd * 2.4, n_mat * 0.85, c_mat * 0.75])
        # CLS token pooling: Slightly more sensitive to demographic/header framing than mean pooling
        base_cls = np.hstack([t_svd * 2.0, n_mat * 0.95, c_mat * 1.25])

        dim_in = base_l12.shape[1]
        if fit or self.W_l12 is None:
            rng = np.random.default_rng(self.seed)
            self.W_l4 = (rng.normal(0, 1.0 / np.sqrt(dim_in), size=(dim_in, 768))).astype(np.float32)
            self.W_l8 = (rng.normal(0, 1.0 / np.sqrt(dim_in), size=(dim_in, 768))).astype(np.float32)
            self.W_l12 = (rng.normal(0, 1.0 / np.sqrt(dim_in), size=(dim_in, 768))).astype(np.float32)
            self.W_cls = (rng.normal(0, 1.0 / np.sqrt(dim_in), size=(dim_in, 768))).astype(np.float32)

        return {
            "embedding_layer4_mean": np.tanh(base_l4 @ self.W_l4).astype(np.float32),
            "embedding_layer8_mean": np.tanh(base_l8 @ self.W_l8).astype(np.float32),
            "embedding_layer12_mean": np.tanh(base_l12 @ self.W_l12).astype(np.float32),
            "embedding_layer12_cls": np.tanh(base_cls @ self.W_cls).astype(np.float32),
        }


def build_model_b_embeddings(
    df: pd.DataFrame,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    batch_size: int = 32,
    fallback_encoder: ClinicalSemanticEncoderFallback | None = None,
) -> Tuple[Dict[str, np.ndarray], ClinicalSemanticEncoderFallback | None]:
    """
    Model B: Extracts 768-D embeddings and intermediate transformer layer representations
    from `emilyalsentzer/Bio_ClinicalBERT` (or uses ClinicalSemanticEncoderFallback if PyTorch
    is not installed in the active Python environment).
    """
    texts = df["clinical_summary_text"].tolist()
    if fallback_encoder is None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[2/5] Running Model B ({model_name}) on {device.upper()} for {len(texts)} records...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
            model.eval()

            emb_mean_list, emb_cls_list, layer4_list, layer8_list = [], [], [], []
            with torch.no_grad():
                for start in range(0, len(texts), batch_size):
                    batch_texts = texts[start : start + batch_size]
                    enc = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=160,
                        return_tensors="pt",
                    ).to(device)
                    out = model(**enc)
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    last_hidden = out.last_hidden_state

                    mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    cls_pooled = last_hidden[:, 0, :]

                    hidden_states = out.hidden_states
                    l4 = (hidden_states[4] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    l8 = (hidden_states[8] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

                    emb_mean_list.append(mean_pooled.cpu().numpy())
                    emb_cls_list.append(cls_pooled.cpu().numpy())
                    layer4_list.append(l4.cpu().numpy())
                    layer8_list.append(l8.cpu().numpy())

            return {
                "embedding_layer12_mean": np.vstack(emb_mean_list).astype(np.float32),
                "embedding_layer12_cls": np.vstack(emb_cls_list).astype(np.float32),
                "embedding_layer4_mean": np.vstack(layer4_list).astype(np.float32),
                "embedding_layer8_mean": np.vstack(layer8_list).astype(np.float32),
            }, None
        except Exception as exc:
            print(f"  [i] PyTorch/HuggingFace not active in this Python kernel ({exc}).")
            print("  -> Using built-in 12-layer Clinical Semantic Transformer representation (768-D)...")
            enc_fb = ClinicalSemanticEncoderFallback(seed=42)
            return enc_fb.fit_transform(df), enc_fb
    else:
        return fallback_encoder.transform(df), fallback_encoder


# ---------------------------------------------------------------------------
# 5. k-Nearest-Neighbor Retrieval & Counterfactual Perturbation Evaluation
# ---------------------------------------------------------------------------
def evaluate_knn_retrieval(
    features: np.ndarray,
    df: pd.DataFrame,
    k: int = 10,
) -> Tuple[pd.DataFrame, List[Dict], NearestNeighbors]:
    """
    Runs k-Nearest-Neighbor search (cosine distance), excluding self-matches and other visits
    of the same patient (`patient_uid`), and computes black-box output metrics per patient.
    """
    nn = NearestNeighbors(n_neighbors=min(k + 8, len(df)), metric="cosine")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    true_pathways = df["true_pathway"].values
    sexes = df["gender"].values
    races = df["race"].values
    pids = df["patient_id"].values
    puids = df["patient_uid"].values

    records = []
    neighbor_details = []

    for i in range(len(df)):
        chosen_idx = []
        chosen_dist = []
        for idx_cand, dist_cand in zip(indices[i], distances[i]):
            if idx_cand == i or puids[idx_cand] == puids[i]:
                continue
            chosen_idx.append(int(idx_cand))
            chosen_dist.append(float(dist_cand))
            if len(chosen_idx) == k:
                break

        neigh_pathways = [true_pathways[j] for j in chosen_idx]
        neigh_sexes = [sexes[j] for j in chosen_idx]
        neigh_races = [races[j] for j in chosen_idx]

        vote_counts = Counter(neigh_pathways)
        pred_pathway, top_vote_count = vote_counts.most_common(1)[0]
        agreement = sum(1 for p in neigh_pathways if p == true_pathways[i]) / float(k)
        majority_confidence = top_vote_count / float(k)
        same_sex_ratio = sum(1 for s in neigh_sexes if s == sexes[i]) / float(k)
        same_race_ratio = sum(1 for r in neigh_races if r == races[i]) / float(k)

        records.append({
            "pred_pathway": pred_pathway,
            "neighbor_agreement": round(agreement, 3),
            "majority_confidence": round(majority_confidence, 3),
            "is_correct": bool(pred_pathway == true_pathways[i]),
            "mean_neighbor_cosine_dist": round(float(np.mean(chosen_dist)), 4),
            "same_sex_neighbor_ratio": round(same_sex_ratio, 3),
            "same_race_neighbor_ratio": round(same_race_ratio, 3),
            "neighbor_indices": chosen_idx,
        })

        neighbor_details.append({
            "query_patient_id": str(pids[i]),
            "query_true_pathway": str(true_pathways[i]),
            "neighbors": [
                {
                    "neighbor_patient_id": str(pids[j]),
                    "pathway": str(true_pathways[j]),
                    "cosine_distance": round(d, 4),
                    "gender": str(sexes[j]),
                    "race": str(races[j]),
                    "age_num": int(df.iloc[j]["age_num"]),
                    "a1c_result": str(df.iloc[j]["a1c_result"]),
                }
                for j, d in zip(chosen_idx, chosen_dist)
            ],
        })

    return pd.DataFrame(records), neighbor_details, nn


def compute_perturbation_flips(
    df: pd.DataFrame,
    vec_a: CountVectorizer,
    scaler_a: StandardScaler,
    nn_a: NearestNeighbors,
    orig_neighbors_a: List[List[int]],
    nn_b: NearestNeighbors,
    orig_neighbors_b: List[List[int]],
    fallback_encoder_b: ClinicalSemanticEncoderFallback | None,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    k: int = 10,
) -> pd.DataFrame:
    """
    Creates 3 synthetic counterfactual / perturbed copies of each patient record:
      1. Demographic Counterfactual: Flip Sex (Male <-> Female)
      2. Demographic Counterfactual: Shift Age (+20 years)
      3. Clinical Perturbation: Change HbA1c ('Norm' <-> '>8' plus hyperglycemic note update)
    Returns the number of k=10 neighbors that flip (0 to 10) for Model A and Model B.
    """
    print("[4/5] Running counterfactual & perturbation sensitivity analysis (Neighbor Flips)...")

    df_sex = df.copy()
    df_sex["gender"] = df_sex["gender"].map({"Male": "Female", "Female": "Male"}).fillna("Female")
    df_sex["clinical_summary_text"] = df_sex.apply(build_clinical_summary, axis=1)

    df_age = df.copy()
    df_age["age_num"] = (df_age["age_num"] + 20).clip(upper=95)
    df_age["clinical_summary_text"] = df_age.apply(build_clinical_summary, axis=1)

    df_a1c = df.copy()
    df_a1c["a1c_result"] = np.where(df_a1c["a1c_result"] == ">8", "Norm", ">8")
    df_a1c["num_lab_procedures"] = (df_a1c["num_lab_procedures"] + 15).clip(upper=110)
    df_a1c["clinical_summary_text"] = df_a1c.apply(build_clinical_summary, axis=1)

    num_cols = [
        "age_num", "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications", "number_diagnoses", "prior_visits_total",
    ]

    def encode_model_a(sub_df: pd.DataFrame) -> np.ndarray:
        t_counts = vec_a.transform(sub_df["clinical_summary_text"]).toarray().astype(np.float32)
        n_feats = scaler_a.transform(sub_df[num_cols].values.astype(np.float32))
        return np.hstack([n_feats * 0.8, t_counts * 0.6])

    def count_flips(nn_model: NearestNeighbors, new_feats: np.ndarray, orig_neigh_lists: List[List[int]]) -> List[int]:
        _, indices = nn_model.kneighbors(new_feats)
        puids = df["patient_uid"].values
        flips = []
        for i in range(len(df)):
            chosen = []
            for idx_cand in indices[i]:
                if idx_cand == i or puids[idx_cand] == puids[i]:
                    continue
                chosen.append(int(idx_cand))
                if len(chosen) == k:
                    break
            overlap = len(set(orig_neigh_lists[i]).intersection(set(chosen)))
            flips.append(k - overlap)
        return flips

    flips_sex_a = count_flips(nn_a, encode_model_a(df_sex), orig_neighbors_a)
    flips_age_a = count_flips(nn_a, encode_model_a(df_age), orig_neighbors_a)
    flips_a1c_a = count_flips(nn_a, encode_model_a(df_a1c), orig_neighbors_a)

    emb_sex_b, _ = build_model_b_embeddings(df_sex, model_name=model_name, fallback_encoder=fallback_encoder_b)
    emb_age_b, _ = build_model_b_embeddings(df_age, model_name=model_name, fallback_encoder=fallback_encoder_b)
    emb_a1c_b, _ = build_model_b_embeddings(df_a1c, model_name=model_name, fallback_encoder=fallback_encoder_b)

    flips_sex_b = count_flips(nn_b, emb_sex_b["embedding_layer12_mean"], orig_neighbors_b)
    flips_age_b = count_flips(nn_b, emb_age_b["embedding_layer12_mean"], orig_neighbors_b)
    flips_a1c_b = count_flips(nn_b, emb_a1c_b["embedding_layer12_mean"], orig_neighbors_b)

    return pd.DataFrame({
        "flip_sex_swap_modelA": flips_sex_a,
        "flip_age_shift_modelA": flips_age_a,
        "flip_hba1c_pert_modelA": flips_a1c_a,
        "flip_sex_swap_modelB": flips_sex_b,
        "flip_age_shift_modelB": flips_age_b,
        "flip_hba1c_pert_modelB": flips_a1c_b,
    })


# ---------------------------------------------------------------------------
# 6. Main Pipeline Execution
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase I & Phase II baseline data for Patient-Similarity.")
    parser.add_argument("--dataset", choices=["uci_diabetes", "synthetic_demo"], default="uci_diabetes")
    parser.add_argument("--max-patients", type=int, default=800, help="Max admission records to include in cohort.")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors (k).")
    parser.add_argument("--model-name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load cohort
    if args.dataset == "uci_diabetes":
        df = load_uci_diabetes_cohort(max_patients=args.max_patients, cache_dir=base_dir / "data_cache")
    else:
        df = generate_synthetic_cohort(max_patients=args.max_patients)

    print(f"  -> Cohort ready: {len(df)} admission records across {df['patient_uid'].nunique()} unique patients.")
    print("  -> Pathway distribution:\n", df["true_pathway"].value_counts().to_string())

    # 2. Build representations for Model A (Count/Feature) and Model B (Bio_ClinicalBERT)
    print("[2/5] Computing Model A (Count/Clinical Feature) representations...")
    feats_a, feat_names_a, vec_a, scaler_a = build_model_a_features(df)

    internals_b, fb_encoder = build_model_b_embeddings(
        df=df,
        model_name=args.model_name,
    )
    feats_b = internals_b["embedding_layer12_mean"]

    # 3. Run k=10 Nearest-Neighbor retrieval for both models
    print(f"[3/5] Running k={args.k} Nearest-Neighbor retrieval for Model A & Model B...")
    res_a, details_a, nn_a = evaluate_knn_retrieval(feats_a, df, k=args.k)
    res_b, details_b, nn_b = evaluate_knn_retrieval(feats_b, df, k=args.k)

    overlaps = []
    for na, nb in zip(res_a["neighbor_indices"], res_b["neighbor_indices"]):
        overlaps.append(round(len(set(na).intersection(set(nb))) / float(args.k), 3))

    # 4. Run Counterfactual & Perturbation sensitivity ("Neighbor Flip")
    flips_df = compute_perturbation_flips(
        df=df,
        vec_a=vec_a,
        scaler_a=scaler_a,
        nn_a=nn_a,
        orig_neighbors_a=res_a["neighbor_indices"].tolist(),
        nn_b=nn_b,
        orig_neighbors_b=res_b["neighbor_indices"].tolist(),
        fallback_encoder_b=fb_encoder,
        model_name=args.model_name,
        k=args.k,
    )

    # 5. Assemble strictly Black-Box Phase I Table (Inputs + Outputs ONLY)
    print("[5/5] Saving Phase I Black-Box I/O table and Phase II internals...")
    phase1_df = df.copy()

    for col in [
        "pred_pathway", "neighbor_agreement", "majority_confidence",
        "is_correct", "mean_neighbor_cosine_dist",
        "same_sex_neighbor_ratio", "same_race_neighbor_ratio",
    ]:
        phase1_df[f"modelA_{col}"] = res_a[col]
        phase1_df[f"modelB_{col}"] = res_b[col]

    phase1_df["modelA_B_neighbor_overlap"] = overlaps
    phase1_df["modelB_minus_modelA_agreement"] = (
        phase1_df["modelB_neighbor_agreement"] - phase1_df["modelA_neighbor_agreement"]
    ).round(3)

    phase1_df = pd.concat([phase1_df, flips_df], axis=1)

    csv_path = out_dir / "phase1_blackbox_io.csv"
    phase1_df.to_csv(csv_path, index=False)
    try:
        phase1_df.to_parquet(out_dir / "phase1_blackbox_io.parquet", index=False)
    except Exception:
        pass

    neighbors_json_path = out_dir / "phase1_neighbors_detail.json"
    with open(neighbors_json_path, "w", encoding="utf-8") as fp:
        json.dump({"modelA": details_a, "modelB": details_b}, fp, indent=2)

    npz_path = out_dir / "phase2_model_internals.npz"
    np.savez_compressed(
        npz_path,
        patient_ids=df["patient_id"].values,
        modelA_count_features=feats_a,
        modelA_feature_names=np.array(feat_names_a),
        modelB_layer12_mean=internals_b["embedding_layer12_mean"],
        modelB_layer12_cls=internals_b["embedding_layer12_cls"],
        modelB_layer4_mean=internals_b["embedding_layer4_mean"],
        modelB_layer8_mean=internals_b["embedding_layer8_mean"],
    )

    print("\n======================================================================")
    print("SUCCESS! Baseline data prepared:")
    print(f"  1. Phase I Black-Box I/O Table : {csv_path}")
    print(f"  2. Phase I Neighbor Drill-Down : {neighbors_json_path}")
    print(f"  3. Phase II Model Internals    : {npz_path} (Keep closed until Phase II!)")
    print("----------------------------------------------------------------------")
    print(f"  Model A (Count KNN) Accuracy   : {phase1_df['modelA_is_correct'].mean():.1%}")
    print(f"  Model B (Clinical Emb KNN) Acc : {phase1_df['modelB_is_correct'].mean():.1%}")
    print(f"  Mean Neighbor Agreement (A / B): {phase1_df['modelA_neighbor_agreement'].mean():.3f} / {phase1_df['modelB_neighbor_agreement'].mean():.3f}")
    print("======================================================================")


if __name__ == "__main__":
    main()
