"""Step 4 (technique: gradient saliency / Integrated Gradients, model-specific, Lecture 05).

Question: inside the text, which tokens make Model B call a neighbour similar?
Target scalar: cosine similarity between the query stay and its 1st Phase I neighbour
(the neighbour embedding is fixed; gradients flow through the query only).
Attribution: Integrated Gradients on the input token embeddings, baseline = the same sequence with
every non-special token replaced by [PAD]'s embedding (32 steps). Token attributions are summed per
field and split into *template* tokens (field names, ':', ',', identical in every record) and *value*
tokens (the patient's actual values).

Writes (gitignored) outputs/token_ig_B.parquet (one row per token) and prints the per-field summary.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from common import FIELDS, OUT, BertEncoder, field_spans, load_cohort

p = argparse.ArgumentParser()
p.add_argument("--stays", type=int, default=150)
p.add_argument("--steps", type=int, default=32)
args = p.parse_args()

df = load_cohort()
pos = {e: i for i, e in enumerate(df["encounter_id"])}
enc = BertEncoder()
tok, model = enc.tok, enc.model
emb_layer = model.get_input_embeddings()
rng = np.random.default_rng(2)
sample = rng.choice(len(df), args.stays, replace=False)


def pooled(inputs_embeds, attn):
    h = model(inputs_embeds=inputs_embeds, attention_mask=attn).last_hidden_state
    m = attn.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1)


rows = []
for n, q in enumerate(sample):
    text = df["model_text"].iloc[q]
    nb_text = df["model_text"].iloc[pos[df["model_b_neighbors"].iloc[q][0]]]
    with torch.no_grad():
        e_nb = enc.encode([nb_text])[0]
    e_nb = torch.tensor(e_nb / np.linalg.norm(e_nb))

    x = tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=128)
    offsets = x.pop("offset_mapping")[0].tolist()
    ids, attn = x["input_ids"], x["attention_mask"]
    with torch.no_grad():
        inp = emb_layer(ids)
        base_ids = ids.clone()
        special = torch.tensor(tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True)).bool()
        base_ids[0, ~special] = tok.pad_token_id
        base = emb_layer(base_ids)

    grads = torch.zeros_like(inp)
    for a in torch.linspace(1.0 / args.steps, 1.0, args.steps):
        z = (base + a * (inp - base)).requires_grad_(True)
        e = pooled(z, attn)[0]
        sim = torch.dot(e / e.norm(), e_nb)
        g, = torch.autograd.grad(sim, z)
        grads += g / args.steps
    attr = ((inp - base) * grads).sum(-1)[0].detach().numpy()        # one score per token
    with torch.no_grad():
        e_q = pooled(inp, attn)[0]
        e_b = pooled(base, attn)[0]
        completeness = float(torch.dot(e_q / e_q.norm(), e_nb) - torch.dot(e_b / e_b.norm(), e_nb))

    spans = field_spans(text)
    tokens = tok.convert_ids_to_tokens(ids[0])
    for t_i, (tkn, (s, e_), a) in enumerate(zip(tokens, offsets, attr)):
        if special[t_i]:
            field, part = "[special]", "special"
        else:
            field, part = "[template]", "template"
            for label, (fs, fe) in spans.items():
                if s >= fs and e_ <= fe:
                    field, part = label, "value"
                    break
            if part == "template":                                    # name of which field?
                for label, (fs, fe) in spans.items():
                    if e_ <= fs and s >= fs - len(label) - 2:
                        field = label
        rows.append({"q": int(df["encounter_id"].iloc[q]), "token": tkn, "pos": t_i, "field": field,
                     "part": part, "attr": float(a), "completeness": completeness})
    if (n + 1) % 25 == 0:
        print(f"{n + 1}/{len(sample)} stays")

res = pd.DataFrame(rows)
res.to_parquet(OUT / "token_ig_B.parquet", index=False)
per_stay = res.groupby("q").agg(sum_attr=("attr", "sum"), completeness=("completeness", "first"))
print("IG completeness, median |sum attr - delta sim|:", float((per_stay.sum_attr - per_stay.completeness).abs().median()))
print("share of |attribution| by part:\n", (res.groupby("part").attr.apply(lambda s: s.abs().sum()) / res.attr.abs().sum()).round(3))
print("share of tokens by part:\n", res.part.value_counts(normalize=True).round(3))
print((res.groupby(["field", "part"]).attr.apply(lambda s: s.abs().sum()) / res.attr.abs().sum())
      .unstack().round(3).sort_values("value", ascending=False).to_string())
