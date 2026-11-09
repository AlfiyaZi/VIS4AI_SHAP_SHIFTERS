""" idea 4: same patient, two stays.

Pattern: B7 (RQ4)

"""
import numpy as np

from common import load, MODELS

df, NB = load()

# First two stays of every patient that has at least two (rows are ordered by visit_index).
first_two = (df.sort_values("visit_index")
               .groupby("patient_nbr").head(2)
               .groupby("patient_nbr").filter(lambda g: len(g) == 2))
pairs = first_two.groupby("patient_nbr").apply(lambda g: tuple(g.index[:2]), include_groups=False)

for m in MODELS:
    shared = np.array([len(set(NB[m][i]) & set(NB[m][j])) for i, j in pairs])
    print(f"{m:8s} patients={len(shared)}  mean shared neighbours: {shared.mean():.2f}/10  "
          f"none shared: {(shared == 0).mean():.1%}")
