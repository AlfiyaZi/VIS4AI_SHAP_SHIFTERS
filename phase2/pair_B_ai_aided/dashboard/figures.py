"""Plotly figures for the Phase II dashboard. Same fonts, colours and look as Pair A's Phase I dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_layer import FIELDS, GROUPS, PATHWAY_ORDER, DashboardData

FONT = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
            size=12, color="#212529")
BASE = dict(font=FONT, paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=52, r=16, t=42, b=48))
GRID = dict(showgrid=True, gridcolor="#F1F3F5", zeroline=False)
PATHWAY_COLORS = {"Kidney / Renal": "#8E6CBF", "Cardiovascular": "#D55E00",
                  "Eye / Neurological": "#0072B2", "Uncomplicated / Other": "#A0A0A0"}
CA, CB, NAVY, LIGHT, MUTED = "#457B9D", "#E63946", "#1D3557", "#ADB5BD", "#6C757D"
QUAL = ["#2A9D8F", "#E76F51", "#457B9D", "#F4A261", "#E63946", "#8E6CBF", "#52B788", "#DDA15E", "#BC6C25"]
FIELD_COLORS = {f: c for f, c in zip(FIELDS, QUAL + ["#6D597A", "#B5838D", "#264653", "#A8DADC", "#E9C46A", "#90A955"])}
COLOR_BY = {
    "pathway": "Next pathway (true)",
    "specialty": "Specialty (top 6)",
    "age": "Age",
    "correct": "Who is right (A / B)",
    "flips_b": "HbA1c fragility, Model B",
    "flips_a": "HbA1c fragility, Model A",
}


def empty(msg: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=13, color=MUTED), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(**BASE, height=height, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ---- view 1: the map -----------------------------------------------------------------------
def map_figure(d: DashboardData, space: str, color_by: str, group: str, row: int | None,
               whatif: dict | None) -> go.Figure:
    xy = d.maps[space]
    df = d.df
    mask = d.groups[group]
    fig = go.Figure()
    if group != "all":                                      # context: everything else, faint
        fig.add_trace(go.Scattergl(x=xy[~mask, 0], y=xy[~mask, 1], mode="markers", hoverinfo="skip",
                                   marker=dict(size=3, color="#E9ECEF"), showlegend=False))
    idx = np.flatnonzero(mask)
    idx = idx[np.random.default_rng(0).permutation(len(idx))]  # random draw order: no class hides another
    hover = ("stay %{customdata[0]}<br>%{customdata[1]}<br>age %{customdata[2]}, %{customdata[3]}"
             "<br>next: %{customdata[4]}<extra></extra>")
    cd = np.stack([df.encounter_id.to_numpy()[idx], df.medical_specialty.to_numpy()[idx], df.age.to_numpy()[idx],
                   df.admission_source.to_numpy()[idx], df.true_next_pathway.to_numpy()[idx]], axis=1)
    if color_by == "age":
        fig.add_trace(go.Scattergl(x=xy[idx, 0], y=xy[idx, 1], mode="markers", customdata=cd, hovertemplate=hover,
                                   marker=dict(size=3.5, color=df.age_mid.to_numpy()[idx], colorscale="Viridis",
                                               colorbar=dict(title="age", thickness=10, len=0.6)), showlegend=False))
    elif color_by in ("flips_a", "flips_b"):
        col = "model_a_flips_hba1c_elevation" if color_by == "flips_a" else "model_b_flips_hba1c_elevation"
        v = df[col].astype(float).to_numpy()[idx]
        fig.add_trace(go.Scattergl(x=xy[idx, 0], y=xy[idx, 1], mode="markers", customdata=cd, hovertemplate=hover,
                                   marker=dict(size=3.5, color=np.nan_to_num(v, nan=-1), cmin=0, cmax=10,
                                               colorscale=[[0, "#F1F3F5"], [0.5, "#F4A261"], [1, "#9D0208"]],
                                               colorbar=dict(title="neighbours<br>replaced", thickness=10, len=0.6)),
                                   showlegend=False))
    else:
        if color_by == "pathway":
            lab, pal = df.true_next_pathway, PATHWAY_COLORS
        elif color_by == "specialty":
            top = df.medical_specialty.value_counts().index[:6]
            lab = df.medical_specialty.where(df.medical_specialty.isin(top), "other")
            pal = {k: QUAL[i] for i, k in enumerate(top)} | {"other": LIGHT}
        else:
            ca, cb = df.model_a_correct.astype(bool), df.model_b_correct.astype(bool)
            lab = pd.Series(np.select([ca & cb, ca & ~cb, ~ca & cb], ["both correct", "only A correct", "only B correct"],
                                      "both wrong"))
            pal = {"both correct": NAVY, "only A correct": CA, "only B correct": CB, "both wrong": "#DEE2E6"}
        labs = lab.to_numpy()[idx]
        fig.add_trace(go.Scattergl(x=xy[idx, 0], y=xy[idx, 1], mode="markers", customdata=cd, hovertemplate=hover,
                                   marker=dict(size=3.5, color=[pal[v] for v in labs]), showlegend=False))
        for k, c in pal.items():                             # legend entries only
            fig.add_trace(go.Scattergl(x=[None], y=[None], mode="markers", marker=dict(size=9, color=c), name=k))
    if row is not None:
        for m, color, sym in [("A", CA, "circle-open"), ("B", CB, "diamond-open")]:
            nb, _ = d.phase1_neighbours(m, row)
            fig.add_trace(go.Scattergl(x=xy[nb, 0], y=xy[nb, 1], mode="markers", name=f"Model {m} neighbours",
                                       marker=dict(size=11, color=color, symbol=sym, line=dict(width=2)),
                                       hoverinfo="skip"))
        fig.add_trace(go.Scattergl(x=[xy[row, 0]], y=[xy[row, 1]], mode="markers", name="selected stay",
                                   marker=dict(size=17, color="#FFD60A", symbol="star", line=dict(width=1.5, color=NAVY)),
                                   hoverinfo="skip"))
        if whatif and whatif.get("row") == row:             # edited stay: placed at its new neighbours' centre
            m = "A" if space.startswith("Model A") else "B"
            new = np.array(whatif[m]["new"])
            p = xy[new].mean(0)
            fig.add_annotation(x=p[0], y=p[1], ax=xy[row, 0], ay=xy[row, 1], xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor=NAVY, text="")
            fig.add_trace(go.Scattergl(x=xy[new, 0], y=xy[new, 1], mode="markers", name="neighbours after edit",
                                       marker=dict(size=11, color=NAVY, symbol="x-thin-open", line=dict(width=2.5)),
                                       hoverinfo="skip"))
    n = int(mask.sum())
    fig.update_layout(**{**BASE, "margin": dict(l=8, r=8, t=36, b=8)}, height=560, dragmode="pan",
                      title=dict(text=f"{space} · {COLOR_BY[color_by]} · {GROUPS[group]} ({n:,} stays)",
                                 font=dict(size=13)),
                      legend=dict(orientation="h", y=-0.02, font=dict(size=10.5)),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      uirevision=space)
    return fig


# ---- view 2: why these neighbours? ---------------------------------------------------------
def match_figure(d: DashboardData, row: int) -> go.Figure:
    """Two neighbour x field grids: filled = same value as the selected stay."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.06,
                        subplot_titles=("Model A: its 10 neighbours", "Model B: its 10 neighbours"))
    labels = list(FIELDS)
    y = d.df.true_next_pathway.to_numpy()
    for c, (m, color) in enumerate([("A", CA), ("B", CB)], start=1):
        nb, dist = d.phase1_neighbours(m, row)
        mm = d.match_matrix(row, nb)
        names = [f"<span style='color:{PATHWAY_COLORS[y[j]]}'>■</span> #{i + 1} {y[j].split(' /')[0][:6]} · d {dist[i]:.4f}"
                 for i, j in enumerate(nb)]
        fig.add_trace(go.Heatmap(z=mm, x=labels, y=names, colorscale=[[0, "#F8F9FA"], [1, color]], zmin=0, zmax=1,
                                 showscale=False, xgap=2, ygap=2,
                                 hovertemplate="%{y}<br>%{x}: %{customdata}<extra></extra>",
                                 customdata=[[("same" if v else f"differs ({d.values[j][f]})") for v, f in zip(r_, labels)]
                                             for r_, j in zip(mm, nb)]), row=1, col=c)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10), row=1, col=c)
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10), row=1, col=c)
        fig.layout.annotations[c - 1].text += f" (fields shared: {mm.sum(1).mean():.1f} / 15)"
    fig.update_layout(**{**BASE, "margin": dict(l=10, r=10, t=40, b=10)}, height=420)
    return fig


def vote_figure(d: DashboardData, row: int, whatif: dict | None = None) -> go.Figure:
    y = d.df.true_next_pathway.to_numpy()
    truth = y[row]
    fig = go.Figure()
    for m, color in [("A", CA), ("B", CB)]:
        nb, _ = d.phase1_neighbours(m, row)
        shares = [(y[nb] == c).mean() for c in PATHWAY_ORDER]
        fig.add_trace(go.Bar(y=[s.split(" /")[0] for s in PATHWAY_ORDER], x=shares, orientation="h", name=f"Model {m}",
                             marker_color=color, text=[f"{s:.0%}" for s in shares], textposition="outside"))
        if whatif and whatif.get("row") == row:
            after = [whatif[m]["shares_after"][c] for c in PATHWAY_ORDER]
            fig.add_trace(go.Bar(y=[s.split(" /")[0] for s in PATHWAY_ORDER], x=after, orientation="h",
                                 name=f"Model {m} after edit", marker=dict(color="white", line=dict(color=color, width=2)),
                                 text=[f"{s:.0%}" for s in after], textposition="outside"))
    fig.update_layout(**{**BASE, "margin": dict(l=10, r=30, t=40, b=30)}, height=280, barmode="group",
                      title=dict(text=f"Vote of the 10 neighbours (tie → class of the nearest) · true: <b>{truth}</b>",
                                 font=dict(size=12.5)),
                      xaxis=dict(range=[0, 1.15], tickformat=".0%", **GRID), yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
    return fig


def decomposition_figure(d: DashboardData, row: int) -> go.Figure:
    """Model A, exact: each neighbour's similarity split into one bar segment per field."""
    nb, _ = d.phase1_neighbours("A", row)
    dec = d.decomposition_a(row, nb)
    names = [f"#{i + 1}" for i in range(len(nb))]
    fig = go.Figure()
    for f in FIELDS:
        fig.add_trace(go.Bar(y=names, x=dec[f], orientation="h", name=f, marker_color=FIELD_COLORS[f],
                             hovertemplate=f"{f}: %{{x:.3f}}<extra></extra>"))
    fig.update_layout(**{**BASE, "margin": dict(l=10, r=10, t=40, b=30)}, height=420, barmode="relative",
                      title=dict(text="Model A, exact: similarity to each neighbour = sum of one term per field",
                                 font=dict(size=12.5)),
                      xaxis=dict(title="cosine similarity", **GRID), yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=-0.22, font=dict(size=9.5), traceorder="normal"))
    return fig


def swap_figure(sw: dict) -> go.Figure:
    """Similarity gained when the neighbour takes the stay's value for one field (Model A and B)."""
    fig = go.Figure()
    fields = list(dict.fromkeys(list(sw["A"].field) + list(sw["B"].field)))
    for m, color in [("A", CA), ("B", CB)]:
        s = sw[m].set_index("field").cost.reindex(fields)
        tot = sw[m].cost.sum()
        fig.add_trace(go.Bar(y=fields, x=s / tot if tot > 0 else s, orientation="h", name=f"Model {m}",
                             marker_color=color, hovertemplate="%{y}: %{x:.0%} of the gap<extra></extra>"))
    fig.update_layout(**{**BASE, "margin": dict(l=10, r=10, t=44, b=30)}, height=max(240, 36 * len(fields) + 80),
                      barmode="group", title=dict(text=f"Selected stay vs neighbour {sw['other']}: which differing "
                                                      f"field costs the most similarity?", font=dict(size=12.5)),
                      xaxis=dict(title="share of the similarity gap closed by fixing this field alone",
                                 tickformat=".0%", **GRID),
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.2))
    return fig


# ---- view 4: global weights and groups ------------------------------------------------------
def weights_figure(d: DashboardData, measure: str, highlight: list | None = None) -> go.Figure:
    w = d.weights
    if measure == "shapley":
        cols = [("shapley_A", "Model A: cost when the field differs", CA),
                ("shapley_B", "Model B: cost when the field differs", CB),
                ("outcome", "Outcome: permutation importance", NAVY)]
        order = w["outcome"].sort_values().index
    else:
        cols = [("variance_A", "Model A: share of the space", CA), ("variance_B", "Model B: share of the space", CB)]
        order = w["variance_B"].sort_values().index
    fig = go.Figure()
    for col, name, color in cols:
        v = w[col].reindex(order)
        fig.add_trace(go.Bar(y=list(order), x=v / v.max(), orientation="h", name=name, marker_color=color,
                             hovertemplate="%{y}: %{x:.2f} of the largest<extra></extra>"))
    if highlight:
        for f in highlight:
            if f in order:
                fig.add_hrect(y0=list(order).index(f) - 0.5, y1=list(order).index(f) + 0.5, fillcolor="#FFD60A",
                              opacity=0.25, line_width=0)
    fig.update_layout(**{**BASE, "margin": dict(l=10, r=10, t=40, b=30)}, height=470, barmode="group",
                      title=dict(text="What each model weighs vs what predicts the outcome (relative to the largest field)"
                                 if measure == "shapley" else "Share of each model's space explained by one field",
                                 font=dict(size=12.5)),
                      xaxis=dict(range=[0, 1.05], **GRID), legend=dict(orientation="h", y=-0.08, font=dict(size=10)))
    return fig


def groups_figure(d: DashboardData, group: str) -> go.Figure:
    df = d.df
    rows = []
    for g in ("all", group) if group != "all" else ("all",):
        m = d.groups[g]
        rows.append({"group": GROUPS[g], "A accuracy": df.model_a_correct[m].mean(), "B accuracy": df.model_b_correct[m].mean(),
                     "A: HbA1c edit replaces (of 10)": df.model_a_flips_hba1c_elevation[m].astype(float).mean() / 10,
                     "B: HbA1c edit replaces (of 10)": df.model_b_flips_hba1c_elevation[m].astype(float).mean() / 10,
                     "neighbours shared by A and B": df.ab_neighbor_overlap[m].astype(float).mean()})
    t = pd.DataFrame(rows).set_index("group")
    fig = go.Figure()
    for i, g in enumerate(t.index):
        fig.add_trace(go.Bar(x=t.columns, y=t.loc[g], name=g, marker_color=[LIGHT, NAVY][i] if len(t) > 1 else NAVY,
                             text=[f"{v:.0%}" if "accuracy" in c or "shared" in c else f"{v * 10:.1f}" for c, v in t.loc[g].items()],
                             textposition="outside"))
    fig.update_layout(**{**BASE, "margin": dict(l=40, r=10, t=40, b=60)}, height=330, barmode="group",
                      title=dict(text=f"{GROUPS[group]} ({int(d.groups[group].sum()):,} stays) vs all stays",
                                 font=dict(size=12.5)),
                      yaxis=dict(range=[0, 1.12], tickformat=".0%", **GRID), legend=dict(orientation="h", y=-0.25))
    return fig
