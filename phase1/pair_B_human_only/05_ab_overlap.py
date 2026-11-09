"""idea 5: where do Model A and Model B disagree?

ab_neighbor_overlap (precomputed) = share of the 10 neighbours that both models return.

Pattern: B6 (RQ1)

"""
from common import load

df, NB = load()

o = (df["ab_neighbor_overlap"] * 10).round().astype(int)          # shared neighbours as a count 0..10
print("How many neighbours A and B share (share of stays):")
print(o.value_counts(normalize=True).sort_index().round(3).to_dict())

# Mean overlap per specialty (only specialties with at least 300 stays, to avoid tiny groups).
g = df.groupby("medical_specialty")["ab_neighbor_overlap"].agg(["mean", "size"])
g = g[g["size"] >= 300].sort_values("mean")
print("\nLowest overlap:\n", g.head(4).round(3))
print("\nHighest overlap:\n", g.tail(3).round(3))
