"""
Pattern: B8
Shows: Text output: share of stays that are never anyone's neighbour, per model and per pathway.
"""
import numpy as np
import pandas as pd

from common import load, MODELS

df, NB = load()

for m in MODELS:
    occ = np.bincount(NB[m].ravel(), minlength=len(df))
    never = pd.Series(occ == 0, index=df.index)
    print(f"\n{m}: never retrieved {never.mean():.1%} ({never.sum()} stays)")
    # Share of never-retrieved stays inside each pathway
    print(" per pathway:", never.groupby(df["true_next_pathway"]).mean().round(3).to_dict())
