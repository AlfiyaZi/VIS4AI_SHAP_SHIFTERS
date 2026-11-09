
import numpy as np

from common import load

ENCOUNTER = 24437208

df, NB = load()
i = int(np.where(df["encounter_id"] == ENCOUNTER)[0][0])
cols = ["gender", "age", "medical_specialty", "admission_source", "A1Cresult", "insulin",
        "time_in_hospital", "num_medications", "true_next_pathway"]

print("Query stay:")
print(df.loc[i, cols].to_dict())
for m in ["model_a", "model_b"]:
    print(f"\n{m}: prediction {df[f'{m}_pred_pathway'][i]} (vote share {df[f'{m}_vote_share'][i]})")
    print(df.loc[NB[m][i], cols].to_string(index=False))
print(f"\nNeighbours shared by A and B: {df['ab_neighbor_overlap'][i]:.0%}")
