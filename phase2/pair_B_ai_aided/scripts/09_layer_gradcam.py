"""Step 9 (technique: Grad-CAM, Lectures 04/05, adapted to a text transformer): which tokens matter
at which layer?

Grad-CAM for images: weight each channel of an inner layer's activation map by its pooled gradient,
sum, ReLU. Here the "map" is the sequence of token vectors after layer l (tokens x 768), the target is
the similarity of the stay to its 1st Phase I neighbour (the neighbour vector is fixed):
    alpha_l = mean over tokens of d sim / d h_l          (pooled gradient, 768-D)
    cam_l(t) = ReLU( h_l(t) . alpha_l )                  (one score per token and layer)
For 150 stays we sum cam per layer over template tokens vs. value tokens (and per field), and draw the
full layer x token map for the walk-through stay of script 07.

Writes (gitignored) outputs/layer_gradcam_B.parquet and figures/fig_layer_gradcam_B.png.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import (CACHE, FIG, NAVY, OUT, TEMPLATE_C, VALUE_C, BertEncoder, apply_style, l2n, load_cohort,
                    token_parts)

apply_style()

df = load_cohort()
pos = {e: i for i, e in enumerate(df["encounter_id"])}
enc = BertEncoder()
tok, model = enc.tok, enc.model


def gradcam(text: str, target: np.ndarray):
    x = tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=128)
    offsets = x.pop("offset_mapping")[0].tolist()
    out = model(**x)
    hs = out.hidden_states                                    # 13 x [1, T, 768]
    for h in hs:
        h.retain_grad()
    m = x["attention_mask"].unsqueeze(-1).float()
    e = (hs[-1] * m).sum(1)[0] / m.sum()
    sim = torch.dot(e / e.norm(), torch.tensor(target))
    sim.backward()
    cams = []
    for h in hs:
        alpha = h.grad[0].mean(0)                             # pooled gradient per channel
        cams.append(torch.relu(h[0] @ alpha).detach().numpy())
    special = tok.get_special_tokens_mask(x["input_ids"][0].tolist(), already_has_special_tokens=True)
    toks = tok.convert_ids_to_tokens(x["input_ids"][0])
    return np.stack(cams), toks, token_parts(text, offsets, special)


X = l2n(np.asarray(np.load(CACHE / "phase2_model_b_layers.npy", mmap_mode="r")[:, -1, :],
                   dtype=np.float32))
rng = np.random.default_rng(2)                                 # same sample as 04_token_saliency.py
sample = rng.choice(len(df), 150, replace=False)
rows = []
cache = OUT / "layer_gradcam_B.parquet"
for n, q in enumerate([] if cache.exists() else sample):
    nb = pos[df["model_b_neighbors"].iloc[q][0]]
    cams, toks, parts = gradcam(df["model_text"].iloc[q], X[nb])
    for layer in range(cams.shape[0]):
        tot = cams[layer].sum() + 1e-12
        for (field, part), c in zip(parts, cams[layer]):
            rows.append({"q": int(df["encounter_id"].iloc[q]), "layer": layer, "field": field, "part": part,
                         "cam_share": float(c / tot)})
    if (n + 1) % 50 == 0:
        print(f"{n + 1}/150")
if cache.exists():
    res = pd.read_parquet(cache)
else:
    res = pd.DataFrame(rows)
    res.to_parquet(cache, index=False)
by_part = res.groupby(["layer", "part"]).cam_share.sum().unstack() / res.q.nunique()
print("share of Grad-CAM mass per layer (mean over stays):\n", by_part.round(3).to_string())

# ---- figure: layer x token map for the walk-through stay + template/value share per layer -------
cand = df.index[df["admission_source"].eq("Emergency Room") & df["medical_specialty"].eq("Unknown")
                & df["A1Cresult"].eq("None") & df["model_b_flips_hba1c_elevation"].eq(10)
                & df["model_b_pred_changed_hba1c_elevation"]]
q = int(cand[0])
cams, toks, parts = gradcam(df["model_text"].iloc[q], X[pos[df["model_b_neighbors"].iloc[q][0]]])
cams = cams / (cams.sum(1, keepdims=True) + 1e-12)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(17, 7.5), gridspec_kw={"height_ratios": [2.2, 1]})
im = ax.imshow(cams[:, 1:-1], aspect="auto", cmap="Reds"); ax.grid(False)
ax.set_xticks(range(len(toks) - 2), [t.replace("##", "") for t in toks[1:-1]], rotation=90, fontsize=7)
for lbl, (field, part) in zip(ax.get_xticklabels(), parts[1:-1]):
    lbl.set_color(VALUE_C if part == "value" else "#868E96")
    lbl.set_fontweight("bold" if part == "value" else "normal")
ax.set_ylabel("layer"); ax.set_yticks(range(13))
ax.set_title(f"1 | Grad-CAM per layer and token, stay {df['encounter_id'].iloc[q]} (similarity to its 1st neighbour; "
             "red labels = patient values, grey = template)", loc="left", fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.015, label="share of the layer's Grad-CAM")
ax2.plot(by_part.index, by_part["template"], "o-", color=TEMPLATE_C, label="template tokens")
ax2.plot(by_part.index, by_part["value"], "o-", color=VALUE_C, label="value tokens (the patient)")
if "special" in by_part:
    ax2.plot(by_part.index, by_part["special"], "o-", color=NAVY, label="[CLS] / [SEP]")
ax2.set_xlabel("layer"); ax2.set_ylabel("share of Grad-CAM"); ax2.set_ylim(0, 1)
ax2.set_title("2 | Mean over 150 stays: where the similarity evidence sits, layer by layer", loc="left", fontsize=10)
ax2.legend(frameon=False, ncol=3)
plt.tight_layout()
fig.savefig(FIG / "fig_layer_gradcam_B.png", dpi=150, bbox_inches="tight")
print("saved", FIG / "fig_layer_gradcam_B.png")
