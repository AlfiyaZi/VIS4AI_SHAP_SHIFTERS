
from pathlib import Path

import numpy as np
import pandas as pd

#DATA_DIR = Path(__file__).resolve().parents[2] / "shared" / "data"
DATA_DIR = Path(__file__).resolve().parents[3] / "shared" / "data"
MODELS = ["model_a", "model_b", "random"]


def load():
    cohort = pd.read_parquet(DATA_DIR / "cohort_encounters.parquet")        # inputs + true label, one row per stay
    retrieval = pd.read_parquet(DATA_DIR / "retrieval_results.parquet")     # neighbours, votes, predictions
    df = cohort.merge(retrieval.drop(columns=["patient_nbr", "true_next_pathway"]),
                      on="encounter_id", validate="one_to_one")


    
    pos = pd.Index(df["encounter_id"])
    NB = {}
    for m in MODELS:
        ids = np.concatenate(df[f"{m}_neighbors"].to_numpy())
        rows = pos.get_indexer(ids)
        assert (rows >= 0).all(), f"{m}: neighbour id not found in cohort"
        NB[m] = rows.reshape(len(df), -1)
    return df, NB
