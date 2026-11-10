""" idea 1: hubs.

k-occurrence = how often a stay appears in the neighbour lists of all other stays.
With 30,248 lists x 10 neighbours, the average is exactly 10. A hub has a much higher count.

Pattern: B8
"""
import numpy as np

from common import load, MODELS

df, NB = load()
show = ["gender", "age", "race", "medical_specialty", "admission_source", "A1Cresult",
        "insulin", "time_in_hospital", "num_medications", "true_next_pathway"]

occ = {}
for m in MODELS:

    occ[m] = np.bincount(NB[m].ravel(), minlength=len(df))
    top1 = np.sort(occ[m])[::-1][: len(df) // 100]          # the 1% most retrieved stays
    print(f"{m:8s} mean {occ[m].mean():.1f}  median {np.median(occ[m]):.0f}  max {occ[m].max()}  "
          f"top 1% hold {top1.sum() / occ[m].sum():.1%} of all neighbour slots")

for m in ["model_a", "model_b"]:
    i = occ[m].argmax()
    other = "model_b" if m == "model_a" else "model_a"
    print(f"\nTop hub of {m}: encounter {df.encounter_id[i]}, retrieved {occ[m][i]}x "
          f"({other}: {occ[other][i]}x)")
    print(df.loc[i, show].to_dict())
    print("Text Model B reads:", df.model_text[i])

    hub = occ[m] >= np.quantile(occ[m], 0.99)
    print(f"Top-1% hubs ({hub.sum()} stays), pathway mix:",
          df.loc[hub, "true_next_pathway"].value_counts(normalize=True).round(3).to_dict())
    print("Whole cohort, pathway mix:      ",
          df["true_next_pathway"].value_counts(normalize=True).round(3).to_dict())
