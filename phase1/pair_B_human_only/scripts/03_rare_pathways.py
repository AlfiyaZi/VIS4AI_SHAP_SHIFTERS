""" idea 3: rare pathways.
Pattern: B3 (RQ1, RQ5)

"""
import numpy as np

from common import load, MODELS

df, NB = load()
y = df["true_next_pathway"].to_numpy()

for p in ["Kidney / Renal", "Eye / Neurological"]:
    q = np.where(y == p)[0]                       # row positions of all stays with this pathway
    for m in MODELS:
        same = y[NB[m][q]] == p                   # matrix len(q) x 10: True where the neighbour shares the pathway
        print(f"{p:20s} {m:8s} n={len(q)}  "
              f"no same-pathway neighbour: {(~same.any(axis=1)).mean():.1%}  "
              f"mean same-pathway neighbours: {same.sum(axis=1).mean():.2f}/10")

# Example: the kidney stay for which Model B finds the most kidney neighbours.
q = np.where(y == "Kidney / Renal")[0]
counts = (y[NB["model_b"][q]] == "Kidney / Renal").sum(axis=1)
i = q[counts.argmax()]
print(f"\nExample kidney stay {df.encounter_id[i]}:")
for m in ["model_a", "model_b"]:
    print(f" {m}: neighbours' pathways {list(y[NB[m][i]])}  -> prediction {df[f'{m}_pred_pathway'][i]}")
