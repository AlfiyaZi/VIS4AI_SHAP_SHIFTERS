"""Step 7: how Model B classifies ONE stay, before and after the HbA1c edit (explaining by example, L05).

record -> text -> tokens (coloured by Integrated Gradients towards the 1st neighbour) -> 768-D vector
-> 10 nearest stays -> majority vote. Then the same stay with A1C "None" -> ">8".
Example = an emergency stay with unknown specialty (the most crowded group, Phase I) whose whole
neighbourhood is replaced by the edit. Map = 2-D MDS of cosine distances of the centered vectors
(query, both neighbour sets, and the 150 stays closest to the query as background).

Writes figures/fig_walkthrough_B.png.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from sklearn.manifold import MDS

from common import CACHE, FIG, K, BertEncoder, l2n, load_cohort, record_values, render

PATHWAY_COL = {"Uncomplicated / Other": "#9AA0A6", "Cardiovascular": "#B33A2E",
               "Kidney / Renal": "#1F5FAE", "Eye / Neurological": "#2E9E6A"}
df = load_cohort()
X = np.asarray(np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")[:, -1, :], dtype=np.float32)
Xn = l2n(X)
patient = df["patient_nbr"].to_numpy()
y = df["true_next_pathway"].to_numpy()

cand = df.index[df["admission_source"].eq("Emergency Room") & df["medical_specialty"].eq("Unknown")
                & df["A1Cresult"].eq("None") & df["model_b_flips_hba1c_elevation"].eq(10)
                & df["model_b_pred_changed_hba1c_elevation"]]
q = int(cand[0])
enc = BertEncoder()
v = record_values(df.iloc[q])
texts = {"original": render(v), "HbA1c > 8": render({**v, "A1C": ">8"})}


def neighbours(vec):
    d = 1.0 - Xn @ vec
    d[patient == patient[q]] = np.inf
    nb = np.argsort(d, kind="stable")[:K]
    return nb, d[nb]


def ig_tokens(text, target, steps=32):
    x = enc.tok(text, return_tensors="pt")
    ids, attn = x["input_ids"], x["attention_mask"]
    emb = enc.model.get_input_embeddings()
    with torch.no_grad():
        inp = emb(ids)
        b_ids = ids.clone(); b_ids[0, 1:-1] = enc.tok.pad_token_id
        base = emb(b_ids)
    tgt = torch.tensor(target)
    grads = torch.zeros_like(inp)
    for a in torch.linspace(1 / steps, 1, steps):
        z = (base + a * (inp - base)).requires_grad_(True)
        h = enc.model(inputs_embeds=z, attention_mask=attn).last_hidden_state
        e = (h * attn.unsqueeze(-1)).sum(1)[0] / attn.sum()
        g, = torch.autograd.grad(torch.dot(e / e.norm(), tgt), z)
        grads += g / steps
    attr = ((inp - base) * grads).sum(-1)[0, 1:-1].detach().numpy()
    return enc.tok.convert_ids_to_tokens(ids[0])[1:-1], attr


views = {}
for name, text in texts.items():
    vec = l2n(enc.encode([text]))[0]
    nb, dist = neighbours(vec)
    toks, attr = ig_tokens(text, Xn[nb[0]])
    labels, counts = np.unique(y[nb], return_counts=True)
    tied = set(labels[counts == counts.max()])
    pred = next(lab for lab in y[nb] if lab in tied)        # Phase I rule: tie -> best-ranked neighbour
    views[name] = dict(vec=vec, nb=nb, dist=dist, toks=toks, attr=attr,
                       vote=dict(zip(labels, counts)), pred=pred)
print("stay", df["encounter_id"].iloc[q], "true:", y[q])
for name, vw in views.items():
    print(f"{name:10s} pred={vw['pred']}  vote={vw['vote']}  d10={vw['dist'][-1]:.5f}")
shift = 1 - float(views["original"]["vec"] @ views["HbA1c > 8"]["vec"])
print(f"shift {shift:.5f} = {shift / views['original']['dist'][-1]:.1f} x d10;  overlap",
      len(set(views["original"]["nb"]) & set(views["HbA1c > 8"]["nb"])))

# ---- figure ---------------------------------------------------------------------------
bg = np.argsort(1.0 - Xn @ views["original"]["vec"])[:150]
pts = np.vstack([views["original"]["vec"], views["HbA1c > 8"]["vec"],
                 Xn[views["original"]["nb"]], Xn[views["HbA1c > 8"]["nb"]], Xn[bg]])
c = l2n(pts - X.mean(0) / np.linalg.norm(X.mean(0)) * (pts @ l2n(X.mean(0)))[:, None])
D = np.clip(1 - c @ c.T, 0, None)
xy = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto").fit_transform(D)
q0, q1, nb0, nb1, bgxy = xy[0], xy[1], xy[2:2 + K], xy[2 + K:2 + 2 * K], xy[2 + 2 * K:]

fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.12)
norm = TwoSlopeNorm(0, -max(abs(views[n]["attr"]).max() for n in views), max(abs(views[n]["attr"]).max() for n in views))
for r, name in enumerate(views):
    ax = fig.add_subplot(gs[r, 0]); ax.axis("off")
    ax.set_title(f"{'①' if r == 0 else '③'} Tokens of the {name} text, coloured by Integrated Gradients "
                 "(red = pulls towards the 1st neighbour)", loc="left", fontsize=10)
    xpos, ypos = 0.0, 0.8
    for t, a in zip(views[name]["toks"], views[name]["attr"]):
        w = 0.012 * len(t.replace("##", "")) + 0.012
        if xpos + w > 1.0:
            xpos, ypos = 0.0, ypos - 0.3
        ax.text(xpos, ypos, t.replace("##", ""), fontsize=9.5, family="monospace", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.15", fc=plt.cm.RdBu_r(norm(a)), ec="none"))
        xpos += w
ax = fig.add_subplot(gs[0:2, 1])
ax.scatter(*bgxy.T, s=10, c="#E3E3E3", label="150 closest stays")
for nb, pts2, mk, name in [(views["original"]["nb"], nb0, "o", "original"), (views["HbA1c > 8"]["nb"], nb1, "s", "HbA1c > 8")]:
    for (px, py), i in zip(pts2, nb):
        ax.plot([q0[0] if name == "original" else q1[0], px], [q0[1] if name == "original" else q1[1], py],
                color="#CCCCCC", lw=0.6, zorder=1)
    ax.scatter(*pts2.T, s=55, marker=mk, c=[PATHWAY_COL[y[i]] for i in nb], edgecolor="k", lw=0.5, zorder=2,
               label=f"neighbours ({name})")
ax.scatter(*q0, s=220, marker="*", c="k", zorder=3, label="stay, original")
ax.scatter(*q1, s=220, marker="*", c="#E07B39", zorder=3, label="stay, HbA1c > 8")
ax.annotate("", xy=q1, xytext=q0, arrowprops=dict(arrowstyle="->", color="#E07B39", lw=1.5))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"② Where the stay lands in Model B's space (MDS, centered cosine)\n"
             f"edit shift = {shift / views['original']['dist'][-1]:.1f} × distance to 10th neighbour", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax = fig.add_subplot(gs[2, 1])
paths = list(PATHWAY_COL)
for r, name in enumerate(views):
    left = 0
    for p_ in paths:
        n = views[name]["vote"].get(p_, 0)
        ax.barh(r, n, left=left, color=PATHWAY_COL[p_], label=p_ if r == 0 else None)
        if n:
            ax.text(left + n / 2, r, str(n), ha="center", va="center", color="white", fontsize=9)
        left += n
ax.set_yticks([0, 1], [f"original → {views['original']['pred'].split(' /')[0]}",
                       f"HbA1c > 8 → {views['HbA1c > 8']['pred'].split(' /')[0]}"])
ax.invert_yaxis(); ax.set_xlim(0, 10); ax.set_xlabel("votes of the 10 neighbours")
ax.set_title(f"④ Majority vote (true next pathway: {y[q]})", fontsize=10)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.45))
fig.suptitle(f"How Model B classifies one emergency stay with unknown specialty (stay {df['encounter_id'].iloc[q]})",
             fontsize=13, x=0.01, ha="left")
fig.savefig(FIG / "fig_walkthrough_B.png", dpi=150, bbox_inches="tight")
print("saved", FIG / "fig_walkthrough_B.png")
