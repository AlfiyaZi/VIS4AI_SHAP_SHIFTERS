"""Step 12 (training experiment): what changes inside Bio_ClinicalBERT when it is TRAINED on the outcome?

Model B uses Bio_ClinicalBERT as is (never trained for our task). Phase II found that its space is
organised by department words, not by the next pathway. Here we fine-tune a copy ("Model B'") on the
next pathway and watch the space change across checkpoints - the assignment's "examine when a
behaviour emerges during training / how representations change across checkpoints".

Setup
  * patients split 80 / 20 (no patient in both); training uses only train patients
  * same input text and same mean pooling as Model B, plus a linear head (4 pathways),
    class-weighted cross-entropy (weights ~ 1/sqrt(frequency), so rare pathways are not ignored)
  * 1 epoch, AdamW lr 2e-5, batch 32, CPU
  * checkpoint 0 = Model B itself (no training)
At every checkpoint, on a fixed evaluation set (4,800 index stays of train patients + 1,200 query stays
of test patients), the SAME measurements as for Model B:
  KNN (k = 10, vote) accuracy and per-pathway recall of the test queries; the head's own accuracy;
  random-pair cosine (the "cone"); variance share of outcome / specialty / admission source;
  HbA1c edit shift / neighbourhood radius for 300 test stays.

Writes (gitignored) outputs/finetune_metrics.csv, shared/data_cache/ft_emb_step{N}.npy and
figures/fig_finetune_B.png.
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from common import (CACHE, FIG, K, MAX_LENGTH, MODEL_NAME, NAVY, OUT, PATHWAY_COLORS, PATHWAY_ORDER, apply_style,
                    l2n, load_cohort, record_values, render)

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--eval-every", type=int, default=75)
p.add_argument("--plot-only", action="store_true", help="re-draw the figure from outputs/finetune_metrics.csv")
args = p.parse_args()
apply_style()
torch.manual_seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))

from transformers import AutoModel, AutoTokenizer

df = load_cohort()
y_all = df["true_next_pathway"].to_numpy()
cls_idx = {c: i for i, c in enumerate(PATHWAY_ORDER)}
yi = np.array([cls_idx[c] for c in y_all])
rng = np.random.default_rng(11)
pats = df["patient_nbr"].unique()
test_p = set(rng.choice(pats, int(0.2 * len(pats)), replace=False))
is_test = df["patient_nbr"].isin(test_p).to_numpy()
train_idx, test_idx = np.where(~is_test)[0], np.where(is_test)[0]
eval_index = rng.choice(train_idx, 4800, replace=False)
eval_query = rng.choice(test_idx, 1200, replace=False)
edit_q = rng.choice(test_idx[df["A1Cresult"].iloc[test_idx].isin(["None", "Norm"]).to_numpy()], 300, replace=False)
texts = df["model_text"].tolist()
edit_texts = [render({**record_values(df.iloc[q]), "A1C": ">8"}) for q in edit_q]
print(f"train stays {len(train_idx):,}, test stays {len(test_idx):,} (patients split 80/20)")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME)
head = nn.Linear(bert.config.hidden_size, len(PATHWAY_ORDER))
freq = np.bincount(yi[train_idx], minlength=len(PATHWAY_ORDER)) / len(train_idx)
w = torch.tensor(1 / np.sqrt(freq), dtype=torch.float32); w = w / w.mean()
loss_fn = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.AdamW(list(bert.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=0.01)


def pool(batch_texts):
    enc = tok(batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    h = bert(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1)


@torch.no_grad()
def embed(idx_or_texts, bs=128):
    bert.eval()
    tx = [texts[i] for i in idx_or_texts] if not isinstance(idx_or_texts[0], str) else list(idx_or_texts)
    out = np.concatenate([pool(tx[b:b + bs]).numpy() for b in range(0, len(tx), bs)])
    bert.train()
    return out


def vote(labels):
    """Majority of the 10 neighbour labels; tie -> class of the best-ranked neighbour (Phase I rule)."""
    vals, cnt = np.unique(labels, return_counts=True)
    tied = set(vals[cnt == cnt.max()])
    return next(v for v in labels if v in tied)


def variance_share(x, labels):
    x = l2n(x); x = x - x.mean(0)
    X = pd.get_dummies(pd.Series(labels).astype(str)).to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    return float(((X @ coef) ** 2).sum() / (x ** 2).sum())


def evaluate(step):
    t0 = time.time()
    emb = embed(np.concatenate([eval_index, eval_query]))
    np.save(CACHE / f"ft_emb_step{step}.npy", emb.astype(np.float16))
    ix, qx = l2n(emb[:len(eval_index)]), l2n(emb[len(eval_index):])
    d = 1 - qx @ ix.T
    nb = np.argsort(d, axis=1)[:, :K]
    lab = y_all[eval_index][nb]
    pred = np.array([vote(row) for row in lab])
    yq = y_all[eval_query]
    with torch.no_grad():
        logits = head(torch.tensor(emb[len(eval_index):]))
    head_pred = np.array(PATHWAY_ORDER)[logits.argmax(1).numpy()]
    allx = np.concatenate([ix, qx])
    r = np.random.default_rng(0)
    i, j = r.integers(0, len(allx), 3000), r.integers(0, len(allx), 3000)
    sel = np.concatenate([eval_index, eval_query])
    # HbA1c edit: shift of the edited test stay vs. distance to its 10th neighbour in the index
    e_orig, e_edit = l2n(embed(list(edit_q))), l2n(embed(edit_texts))
    d10 = np.sort(1 - e_orig @ ix.T, axis=1)[:, K - 1]
    shift = 1 - (e_orig * e_edit).sum(1)
    row = {"step": step, "knn_accuracy": float((pred == yq).mean()),
           "knn_macro_recall": float(np.mean([(pred[yq == c] == c).mean() for c in PATHWAY_ORDER])),
           **{f"knn_recall {c}": float((pred[yq == c] == c).mean()) for c in PATHWAY_ORDER},
           "head_accuracy": float((head_pred == yq).mean()),
           "head_macro_recall": float(np.mean([(head_pred[yq == c] == c).mean() for c in PATHWAY_ORDER])),
           "same_pathway_neighbours": float((lab == yq[:, None]).mean()),
           "random_pair_cosine": float((allx[i] * allx[j]).sum(1).mean()),
           "var_outcome": variance_share(allx, y_all[sel]),
           "var_specialty": variance_share(allx, df["medical_specialty"].to_numpy()[sel]),
           "var_source": variance_share(allx, df["admission_source"].to_numpy()[sel]),
           "hba1c_shift_over_radius": float(np.median(shift / np.maximum(d10, 1e-9))),
           "median_d10": float(np.median(d10))}
    print(f"[step {step:4d}] knn acc {row['knn_accuracy']:.3f} macro-rec {row['knn_macro_recall']:.3f} | "
          f"head acc {row['head_accuracy']:.3f} | cos {row['random_pair_cosine']:.4f} | var outcome "
          f"{row['var_outcome']:.3f} specialty {row['var_specialty']:.3f} | HbA1c shift/r "
          f"{row['hba1c_shift_over_radius']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    return row


if args.plot_only:
    n_steps = 0
else:
    rows = [evaluate(0)]
order = np.random.default_rng(12).permutation(train_idx)
if not args.plot_only:
    n_steps = int(args.epochs * len(order) / args.batch)
bert.train()
t0 = time.time()
for step in range(1, n_steps + 1):
    b = order[((step - 1) * args.batch) % len(order):][:args.batch]
    logits = head(pool([texts[i] for i in b]))
    loss = loss_fn(logits, torch.tensor(yi[b]))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 25 == 0:
        print(f"step {step}/{n_steps} loss {loss.item():.3f} ({(time.time() - t0) / step:.2f}s/step)", flush=True)
    if step % args.eval_every == 0 or step == n_steps:
        rows.append(evaluate(step))
        pd.DataFrame(rows).to_csv(OUT / "finetune_metrics.csv", index=False)
if args.plot_only:
    res = pd.read_csv(OUT / "finetune_metrics.csv")
else:
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "finetune_metrics.csv", index=False)

# ---- figure ---------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
ax = axes[0]
ax.plot(res.step, res.knn_accuracy, "o-", color="#E63946", label="KNN accuracy (Model B' vote)")
ax.plot(res.step, res.knn_macro_recall, "o--", color="#E63946", label="KNN macro recall")
ax.plot(res.step, res.head_macro_recall, "s--", color=NAVY, label="trained head macro recall")
ax.axhline(0.25, color=NAVY, ls=":", lw=0.8, label="chance macro recall (0.25)")
ax.axhline((y_all[eval_query] == "Uncomplicated / Other").mean(), color="#ADB5BD", ls=":", label="majority share")
ax.set_title("classification"); ax.legend(fontsize=8)
ax = axes[1]
for c in PATHWAY_ORDER:
    ax.plot(res.step, res[f"knn_recall {c}"], "o-", color=PATHWAY_COLORS[c], label=c)
ax.set_title("KNN recall per pathway"); ax.legend(fontsize=8)
ax = axes[2]
ax.plot(res.step, res.var_outcome, "o-", color="#E63946", label="next pathway (outcome)")
ax.plot(res.step, res.var_specialty, "o-", color="#457B9D", label="specialty")
ax.plot(res.step, res.var_source, "o-", color="#2A9D8F", label="admission source")
ax.set_title("share of the space explained by ..."); ax.legend(fontsize=8)
ax = axes[3]
ax.plot(res.step, res.random_pair_cosine, "o-", color=NAVY, label="random-pair cosine (left)")
ax2 = ax.twinx(); ax2.plot(res.step, res.hba1c_shift_over_radius, "s--", color="#E63946", label="HbA1c shift / radius (right)")
ax2.grid(False); ax2.spines["right"].set_visible(True); ax.set_title("geometry")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
for a in axes:
    a.set_xlabel("training step (0 = Model B)")
fig.suptitle("Fine-tuning Bio_ClinicalBERT on the next pathway: what changes inside, checkpoint by checkpoint", x=0.01, ha="left")
plt.tight_layout()
fig.savefig(FIG / "fig_finetune_B.png", dpi=150, bbox_inches="tight")
print("saved", FIG / "fig_finetune_B.png")
