"""
Figure builders for the four coordinated views of the Pair A dashboard.
Engineered for high performance, clear visual hierarchy, and reliable interactive selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_layer import (
    A1C_COLORS,
    A1C_ORDER,
    EDITS,
    MODELS,
    PATHWAY_COLORS,
    PATHWAY_SHORT,
    PATHWAYS,
    SEX_SYMBOLS,
    DashboardData,
)

FONT = dict(
    family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    size=12,
    color="#212529",
)
BASE_LAYOUT = dict(
    font=FONT,
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=52, r=16, t=42, b=48),
)
GRID = dict(showgrid=True, gridcolor="#F1F3F5", zeroline=False)
DIM_OPACITY = 0.08


def _selected_positions(ids_in_trace: list[int], selected: set[int] | None) -> list[int] | None:
    """Return matching 0-based indices if a selection filter is active, or None if unfiltered."""
    if selected is None:
        return None
    return [i for i, eid in enumerate(ids_in_trace) if eid in selected]


# ---------------------------------------------------------------------------
# View 1: Jittered Multi-Channel Scatter Explorer
# ---------------------------------------------------------------------------
def scatter_explorer(
    data: DashboardData,
    selected: set[int] | None,
    x_model: str,
    jitter: bool,
    contours: bool,
    uirevision_key: str = "0",
) -> go.Figure:
    df = data.df
    xcol = f"{x_model}_agreement"
    xlab = "Model B (Bio_ClinicalBERT)" if x_model == "model_b" else "Model A (Tabular KNN)"

    fig = make_subplots(
        rows=1,
        cols=len(PATHWAYS),
        shared_yaxes=True,
        horizontal_spacing=0.035,
        subplot_titles=[
            f"<b>{PATHWAY_SHORT[p]}</b> (n={int((df['true_next_pathway'] == p).sum()):,})"
            for p in PATHWAYS
        ],
    )

    for col_idx, pathway in enumerate(PATHWAYS, start=1):
        sub = df[df["true_next_pathway"] == pathway]
        if len(sub) == 0:
            continue

        for a1c in A1C_ORDER:
            sub_a1c = sub[sub["A1Cresult"] == a1c]
            if len(sub_a1c) == 0:
                continue

            x_vals = (sub_a1c[xcol] + (sub_a1c["jitter_x"] if jitter else 0.0)).round(4).tolist()
            y_vals = (sub_a1c["number_diagnoses"] + (sub_a1c["jitter_y"] if jitter else 0.0)).round(3).tolist()
            ids = [int(e) for e in sub_a1c.index]
            symbols = [SEX_SYMBOLS.get(g, "circle") for g in sub_a1c["gender"]]
            hover_text = [
                f"Sex: {g} | Age: {a} | HbA1c: {a1c}"
                for g, a in zip(sub_a1c["gender"], sub_a1c["age"])
            ]

            fig.add_trace(
                go.Scattergl(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name=f"HbA1c: {a1c}",
                    legendgroup=f"a1c-{a1c}",
                    showlegend=(col_idx == 1),
                    customdata=ids,
                    text=hover_text,
                    marker=dict(
                        color=A1C_COLORS.get(a1c, "#BDBDBD"),
                        symbol=symbols,
                        size=5,
                        opacity=0.68,
                        line=dict(width=0),
                    ),
                    selectedpoints=_selected_positions(ids, selected),
                    selected=dict(marker=dict(opacity=0.95, size=6.5)),
                    unselected=dict(marker=dict(opacity=DIM_OPACITY)),
                    hovertemplate=(
                        "<b>Encounter #%{customdata}</b><br>"
                        f"True Outcome: {pathway}<br>"
                        f"{xlab} Agreement: %{{x:.2f}}<br>"
                        "Diagnoses Count: %{y:.0f}<br>"
                        "%{text}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        if contours and len(sub) > 40:
            cx = (sub[xcol] + sub["jitter_x"]).round(4).tolist()
            cy = (sub["number_diagnoses"] + sub["jitter_y"]).round(3).tolist()
            fig.add_trace(
                go.Histogram2dContour(
                    x=cx,
                    y=cy,
                    ncontours=7,
                    contours=dict(coloring="none"),
                    showscale=False,
                    line=dict(color="rgba(30,30,30,0.55)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    xbins=dict(size=0.1),
                    ybins=dict(size=1),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_xaxes(
            range=[-0.08, 1.08],
            tickvals=[0.0, 0.5, 1.0],
            ticktext=["0%", "50%", "100%"],
            tickfont=dict(size=10),
            title_text="Neighbor Agreement (Vote Share)" if col_idx == 2 else None,
            title_font=dict(size=11),
            row=1,
            col=col_idx,
            **GRID,
        )
        fig.update_yaxes(
            range=[0, 17],
            dtick=4,
            row=1,
            col=col_idx,
            **GRID,
        )

    fig.update_yaxes(title_text="Diagnoses Count (Comorbidity)", row=1, col=1)
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=52, r=14, t=34, b=58)},
        height=390,
        dragmode="lasso",
        uirevision=f"scatter-{x_model}-{uirevision_key}",
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0.0,
            font=dict(size=11),
            itemsizing="constant",
            title=dict(text="Color (● Female, ◆ Male): ", font=dict(size=11)),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# View 2: Reordered Error & Confusion Breakdown Matrix
# ---------------------------------------------------------------------------
def confusion_heatmap(
    data: DashboardData,
    model: str,
    normalize: bool,
    clustered: bool,
    highlight: tuple[str, str] | None,
) -> go.Figure:
    order = data.order if clustered else PATHWAYS
    counts = data.confusion[model].loc[order, order]
    rates = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    z_mat = (rates if normalize else counts).to_numpy(dtype=float).tolist()

    text = [
        [f"<b>{rates.iat[i, j]:.0%}</b><br>{int(counts.iat[i, j]):,}" for j in range(len(order))]
        for i in range(len(order))
    ]
    short = [PATHWAY_SHORT[p] for p in order]

    df = data.df
    pred_col = "majority_pred_pathway" if model == "majority" else f"{model}_pred_pathway"
    acc = float((df[pred_col] == df["true_next_pathway"]).mean())
    recalls = [float(rates.loc[p, p]) for p in order if counts.loc[p].sum() > 0]
    macro_rec = float(np.mean(recalls)) if recalls else 0.0

    # Plain nested Python lists preserve str, int, float types in JSON
    cd = [
        [[t, p, int(counts.loc[t, p]), float(rates.loc[t, p])] for p in order]
        for t in order
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z_mat,
            x=short,
            y=short,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10, family="Inter, system-ui, sans-serif"),
            colorscale="Blues",
            showscale=False,
            zmin=0,
            zmax=1.0 if normalize else None,
            customdata=cd,
            hovertemplate=(
                "True Next Pathway: <b>%{customdata[0]}</b><br>"
                "Predicted Pathway: <b>%{customdata[1]}</b><br>"
                "Encounters: <b>%{customdata[2]:,}</b> (%{customdata[3]:.1%} of row)"
                "<extra>Click cell to filter cohort</extra>"
            ),
            xgap=2,
            ygap=2,
        )
    )

    # Dotted outline on the diagonal cells (correct predictions)
    for i in range(len(order)):
        fig.add_shape(
            type="rect",
            x0=i - 0.5,
            x1=i + 0.5,
            y0=i - 0.5,
            y1=i + 0.5,
            line=dict(color="#1D3557", width=1.5, dash="dot"),
        )

    # Bold orange outline on the currently clicked cell
    if highlight:
        t, p = highlight
        if t in order and p in order:
            i, j = order.index(t), order.index(p)
            fig.add_shape(
                type="rect",
                x0=j - 0.5,
                x1=j + 0.5,
                y0=i - 0.5,
                y1=i + 0.5,
                line=dict(color="#E8590C", width=3.5),
            )

    short_titles = {
        "model_a": "Model A (Tabular KNN)",
        "model_b": "Model B (ClinicalBERT)",
        "majority": "Majority Baseline",
    }
    show_y_title = model == "model_a"
    left_margin = 56 if show_y_title else 42

    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=left_margin, r=6, t=46, b=44)},
        height=390,
        title=dict(
            text=(
                f"<b>{short_titles.get(model, MODELS[model])}</b><br>"
                f"<sup>Acc: <b>{acc:.1%}</b> | Bal. Recall: <b>{macro_rec:.1%}</b></sup>"
            ),
            x=0.02,
            font=dict(size=12),
        ),
        xaxis=dict(title=dict(text="Predicted", font=dict(size=10.5)), side="bottom", tickfont=dict(size=9.5)),
        yaxis=dict(
            title=dict(text="True Pathway" if show_y_title else None, font=dict(size=10.5)),
            autorange="reversed",
            tickfont=dict(size=9.5),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# View 3: Input-Space UMAP & Longitudinal Trajectories
# ---------------------------------------------------------------------------
UMAP_COLOR_MODES = {
    "driver": "Primary Driver ('change')",
    "true_next_pathway": "True Next Pathway",
    "disparity": "Agreement Disparity (B - A)",
    "model_a_agreement": "Model A Agreement",
    "model_b_agreement": "Model B Agreement",
    "umap_cluster": "HDBSCAN Cluster",
}


def umap_view(
    data: DashboardData,
    selected: set[int] | None,
    color_mode: str,
    show_traj: bool,
    n_patients: int,
    annotate: bool,
    uirevision_key: str = "0",
) -> go.Figure:
    df = data.df
    fig = go.Figure()
    driver = str(data.driver_ranking.iloc[0]["feature"])

    # Compute fixed axis bounds with 6% padding so the UMAP viewport is always stable and complete
    xmin, xmax = float(df["umap_x"].min()), float(df["umap_x"].max())
    ymin, ymax = float(df["umap_y"].min()), float(df["umap_y"].max())
    xpad = (xmax - xmin) * 0.06
    ypad = (ymax - ymin) * 0.06
    x_range = [xmin - xpad, xmax + xpad]
    y_range = [ymin - ypad, ymax + ypad]

    # Continuous color scales
    if color_mode in ("disparity", "model_a_agreement", "model_b_agreement"):
        col = "agreement_disparity_b_minus_a" if color_mode == "disparity" else color_mode
        is_div = color_mode == "disparity"
        ids = [int(e) for e in df.index]
        c_vals = df[col].round(4).tolist()

        fig.add_trace(
            go.Scattergl(
                x=df["umap_x"].round(4).tolist(),
                y=df["umap_y"].round(4).tolist(),
                mode="markers",
                name=UMAP_COLOR_MODES.get(color_mode, color_mode),
                customdata=ids,
                marker=dict(
                    color=c_vals,
                    colorscale="RdBu" if is_div else "Viridis",
                    cmin=-1.0 if is_div else 0.0,
                    cmax=1.0,
                    cmid=0.0 if is_div else None,
                    size=4.5,
                    opacity=0.72,
                    colorbar=dict(
                        title=dict(text="B − A" if is_div else "Agree", side="right"),
                        thickness=12,
                        len=0.78,
                    ),
                ),
                selectedpoints=_selected_positions(ids, selected),
                selected=dict(marker=dict(opacity=0.95, size=6)),
                unselected=dict(marker=dict(opacity=DIM_OPACITY)),
                hovertemplate=(
                    "<b>Encounter #%{customdata}</b><br>"
                    f"{UMAP_COLOR_MODES.get(color_mode, color_mode)}: %{{marker.color:.2f}}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
    else:
        # Categorical color mapping using real per-category Scattergl traces
        if color_mode == "true_next_pathway":
            categories = PATHWAYS
            palette = PATHWAY_COLORS
            key_series = df["true_next_pathway"].astype(str)
            label_prefix = "True Next Pathway"
        elif color_mode == "umap_cluster":
            key_series = df["umap_cluster"].map(lambda c: "noise" if c < 0 else f"C{int(c)}")
            categories = sorted(key_series.unique(), key=lambda s: (s == "noise", len(s), s))
            palette = _qualitative(categories)
            label_prefix = "HDBSCAN Cluster"
        else:
            raw_key = df[driver].astype(str)
            top = raw_key.value_counts().index[:9].tolist()
            key_series = raw_key.where(raw_key.isin(top), "other values")
            categories = top + (["other values"] if (key_series == "other values").any() else [])
            palette = _qualitative(categories)
            label_prefix = f"Primary Driver ({driver})"

        for cat in categories:
            sub = df[key_series == cat]
            if len(sub) == 0:
                continue
            sub_ids = [int(e) for e in sub.index]
            fig.add_trace(
                go.Scattergl(
                    x=sub["umap_x"].round(4).tolist(),
                    y=sub["umap_y"].round(4).tolist(),
                    mode="markers",
                    name=f"{cat} (n={len(sub):,})",
                    customdata=sub_ids,
                    marker=dict(
                        color=palette.get(cat, "#ADB5BD"),
                        size=4.5,
                        opacity=0.72,
                    ),
                    selectedpoints=_selected_positions(sub_ids, selected),
                    selected=dict(marker=dict(opacity=0.95, size=6)),
                    unselected=dict(marker=dict(opacity=DIM_OPACITY)),
                    hovertemplate=(
                        "<b>Encounter #%{customdata}</b><br>"
                        f"{label_prefix}: <b>{cat}</b>"
                        "<extra></extra>"
                    ),
                )
            )

    if show_traj:
        _add_trajectories(fig, df, selected, n_patients)

    if annotate and len(data.cluster_summary):
        for _, r in data.cluster_summary.iterrows():
            fig.add_annotation(
                x=float(r["x"]),
                y=float(r["y"]),
                text=f"<b>C{int(r['cluster'])}</b>: {r['label']}",
                showarrow=False,
                font=dict(size=10, color="#1D3557"),
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#ADB5BD",
                borderwidth=1,
                borderpad=3,
            )

    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=16, r=16, t=16, b=52)},
        height=430,
        dragmode="lasso",
        uirevision=f"umap-{color_mode}-{uirevision_key}",
        xaxis=dict(range=x_range, showticklabels=False, title=None, **GRID),
        yaxis=dict(range=y_range, showticklabels=False, title=None, **GRID),
        legend=dict(
            font=dict(size=11),
            orientation="h",
            y=-0.06,
            x=0.0,
            itemsizing="constant",
        ),
    )
    return fig


def _qualitative(groups: list[str]) -> dict[str, str]:
    base = [
        "#2A9D8F",
        "#E76F51",
        "#457B9D",
        "#F4A261",
        "#E63946",
        "#8E6CBF",
        "#52B788",
        "#DDA15E",
        "#BC6C25",
    ]
    out = {g: base[i % len(base)] for i, g in enumerate(groups)}
    for g in ("noise", "other values"):
        if g in out:
            out[g] = "#ADB5BD"
    return out


def _add_trajectories(fig: go.Figure, df: pd.DataFrame, selected: set[int] | None, n: int):
    """Directed V1 -> V2 -> V3 paths of repeat admissions."""
    counts = df.groupby("patient_nbr")["encounter_id"].transform("size")
    multi = df[counts >= 2]
    if selected:
        pids = multi.loc[multi.index.isin(selected), "patient_nbr"].unique()
    else:
        pids = multi["patient_nbr"].unique()
    if len(pids) > n:
        pids = np.random.default_rng(3).choice(pids, size=n, replace=False)

    paths = multi[multi["patient_nbr"].isin(set(pids))].sort_values(["patient_nbr", "visit_index"])
    xs, ys, txt = [], [], []
    for pid, g in paths.groupby("patient_nbr", sort=False):
        xs += g["umap_x"].round(4).tolist() + [None]
        ys += g["umap_y"].round(4).tolist() + [None]
        txt += [
            f"Patient #{int(pid)}, Visit #{int(v)} (Encounter #{int(eid)})"
            for v, eid in zip(g["visit_index"], g.index)
        ] + [None]

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=f"Trajectories ({len(pids)} pts)",
            line=dict(color="rgba(29, 53, 87, 0.55)", width=1.4),
            marker=dict(symbol="arrow", angleref="previous", size=8, color="rgba(29, 53, 87, 0.85)"),
            text=txt,
            hovertemplate="%{text}<extra></extra>",
        )
    )


# ---------------------------------------------------------------------------
# View 4: Counterfactual Neighbor Flips & Patient Inspector
# ---------------------------------------------------------------------------
def flips_view(data: DashboardData, selected: set[int] | None) -> go.Figure:
    df = data.df
    sub = df.loc[df.index.isin(selected)] if selected else df
    fig = go.Figure()
    colors = {"model_a": "#457B9D", "model_b": "#E63946"}

    for model in ("model_a", "model_b"):
        means, errs, ns, labels = [], [], [], []
        for edit in EDITS:
            v = sub[f"{model}_flips_{edit}"].dropna().astype(float)
            m_val = float(v.mean()) if len(v) else 0.0
            e_val = float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            means.append(round(m_val, 3))
            errs.append(round(e_val, 3))
            ns.append(int(len(v)))
            labels.append(f"{m_val:.1f}" if len(v) else "N/A")

        fig.add_trace(
            go.Bar(
                x=list(EDITS.values()),
                y=means,
                name=MODELS[model],
                marker_color=colors[model],
                text=labels,
                textposition="outside",
                textfont=dict(size=11),
                error_y=dict(type="data", array=errs, thickness=1.3, width=4),
                customdata=ns,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Mean Flips: <b>%{y:.2f}</b> of 10 neighbors<br>"
                    "Evaluated Encounters: n=%{customdata:,}"
                    "<extra>" + MODELS[model] + "</extra>"
                ),
            )
        )

    scope = f"Filtered Cohort (n={len(sub):,})" if selected else f"Full Cohort (n={len(sub):,})"
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=48, r=16, t=36, b=46)},
        height=235,
        barmode="group",
        title=dict(
            text=f"<b>Counterfactual Neighbor Flips (of k=10)</b> — <sup>{scope} (±95% CI)</sup>",
            x=0.01,
            font=dict(size=12.5),
        ),
        yaxis=dict(title="Flipped Neighbors", range=[0, 11.2], dtick=2, **GRID),
        legend=dict(orientation="h", y=-0.26, x=0.0, font=dict(size=11)),
    )
    return fig


def neighbor_breakdown(data: DashboardData, eid: int) -> go.Figure:
    row = data.df.loc[eid]
    true_cat = str(row["true_next_pathway"])
    fig = go.Figure()
    labels = {"model_a": "Model A", "model_b": "Model B", "random": "Random"}

    for p in PATHWAYS:
        vals = []
        for m in labels:
            nb = [int(n) for n in row[f"{m}_neighbors"]]
            path = data.df.loc[nb, "true_next_pathway"]
            vals.append(int((path == p).sum()))

        fig.add_trace(
            go.Bar(
                y=list(labels.values()),
                x=vals,
                orientation="h",
                name=PATHWAY_SHORT[p],
                marker=dict(
                    color=PATHWAY_COLORS[p],
                    line=dict(
                        color="#111111" if p == true_cat else "#FFFFFF",
                        width=2.2 if p == true_cat else 0.5,
                    ),
                ),
                hovertemplate="%{y}: <b>%{x} / 10</b> neighbors in " + p + "<extra></extra>",
            )
        )

    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=68, r=12, t=28, b=34)},
        barmode="stack",
        height=155,
        title=dict(
            text=f"10 Retrieved Neighbors by Outcome (True: <b>{true_cat}</b> outlined in black)",
            x=0.01,
            font=dict(size=11.5),
        ),
        xaxis=dict(range=[0, 10], dtick=2, title=None, **GRID),
        legend=dict(orientation="h", y=-0.34, x=0.0, font=dict(size=10.5)),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 2 & 3: Advanced Diagnostic & Analytical Views
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (robust for small n and p near 0/1)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


VOTE_SHARE_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def calibration_curve(data: DashboardData, selected: set[int] | None) -> go.Figure:
    """Reliability diagram: label-free confidence (vote share) vs. empirical accuracy.

    IMPORTANT: this deliberately uses ``*_vote_share`` (how many of the 10 neighbors voted for
    the predicted class) and NOT ``*_agreement``. ``agreement`` is computed against the TRUE
    label (share of neighbors whose pathway equals the query's true pathway), so binning
    accuracy by agreement is circular: agreement >= 0.6 guarantees a correct majority vote.
    Vote share is available at prediction time, so it is a genuine confidence signal.
    """
    df = data.df
    sub = df.loc[df.index.isin(selected)] if selected else df
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    labels = [f"{int(v * 10)}/10" for v in VOTE_SHARE_LEVELS]
    styles = {
        "model_a": ("Model A (Tabular KNN)", "#457B9D", "solid", "circle"),
        "model_b": ("Model B (ClinicalBERT KNN)", "#E63946", "dot", "diamond"),
    }

    for model, (name, color, dash, symbol) in styles.items():
        share = sub[f"{model}_vote_share"].round(1)
        counts, acc, lo, hi = [], [], [], []
        for level in VOTE_SHARE_LEVELS:
            mask = np.isclose(share, level)
            n = int(mask.sum())
            k = int(sub.loc[mask, f"{model}_correct"].sum())
            counts.append(n)
            if n >= 10:
                low, high = wilson_ci(k, n)
                acc.append(round(k / n * 100, 1))
                lo.append(round((k / n - low) * 100, 1))
                hi.append(round((high - k / n) * 100, 1))
            else:  # too few encounters for a meaningful rate
                acc.append(None)
                lo.append(None)
                hi.append(None)

        if model == "model_a":
            fig.add_trace(
                go.Bar(
                    x=labels, y=counts, name="Encounters per vote share (Model A)",
                    marker_color="#E9ECEF", marker_line=dict(color="#CED4DA", width=1),
                    hovertemplate="Vote share %{x}: <b>%{y:,}</b> encounters<extra>Model A</extra>",
                ),
                secondary_y=False,
            )
        fig.add_trace(
            go.Scatter(
                x=labels, y=acc, name=f"{name} accuracy", mode="lines+markers",
                line=dict(color=color, width=3, dash=dash), marker=dict(size=7, symbol=symbol),
                error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo, thickness=1.2, width=3),
                customdata=counts,
                hovertemplate=(f"<b>{name}</b><br>Vote share %{{x}}<br>Accuracy: <b>%{{y:.1f}}%</b>"
                               "<br>n = %{customdata:,} (95% Wilson CI)<extra></extra>"),
                connectgaps=False,
            ),
            secondary_y=True,
        )

    # Reference lines: what you get WITHOUT any retrieval
    maj = float((sub["majority_pred_pathway"] == sub["true_next_pathway"]).mean() * 100)
    rnd = float((sub["random_pred_pathway"] == sub["true_next_pathway"]).mean() * 100)
    for y, text, color in ((maj, f"Majority baseline {maj:.1f}%", "#495057"),
                           (rnd, f"Random-neighbor baseline {rnd:.1f}%", "#ADB5BD")):
        fig.add_hline(y=y, line=dict(color=color, dash="dash", width=1.3), secondary_y=True,
                      annotation_text=text, annotation_position="bottom right",
                      annotation_font=dict(size=10, color=color))

    scope = f"Selection (n={len(sub):,})" if selected else f"Full cohort (n={len(sub):,})"
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=55, r=55, t=42, b=48)},
        height=370,
        title=dict(text=f"<b>Reliability: vote share (confidence) vs. accuracy</b> — <sup>{scope}</sup>",
                   x=0.01, font=dict(size=13)),
        xaxis=dict(title="Vote share = neighbors (of 10) voting for the predicted pathway", **GRID),
        legend=dict(orientation="h", y=-0.24, x=0.0, font=dict(size=10.5)),
    )
    fig.update_yaxes(title_text="Encounters", secondary_y=False, **GRID)
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100], dtick=20, secondary_y=True)
    return fig


def disparity_battle(data: DashboardData, selected: set[int] | None, slice_by: str) -> go.Figure:
    """Head-to-head battleground on encounters where Model A and Model B make conflicting predictions."""
    df = data.df
    sub = df.loc[df.index.isin(selected)] if selected else df
    disagree = sub[sub["model_a_pred_pathway"] != sub["model_b_pred_pathway"]].copy()

    if len(disagree) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No model disagreements in current selection", showarrow=False, font=dict(size=13))
        fig.update_layout(**BASE_LAYOUT, height=370)
        return fig

    disagree["outcome"] = "Both Wrong"
    disagree.loc[disagree["model_a_pred_pathway"] == disagree["true_next_pathway"], "outcome"] = "Model A Wins"
    disagree.loc[disagree["model_b_pred_pathway"] == disagree["true_next_pathway"], "outcome"] = "Model B Wins"

    slice_col = slice_by if slice_by in disagree.columns else "age"
    # Filter out tiny categories
    top_slices = disagree[slice_col].value_counts()
    valid_slices = top_slices[top_slices >= 10].index.tolist()
    if pd.api.types.is_numeric_dtype(disagree[slice_col]):
        valid_slices = sorted(valid_slices)
    elif slice_col == "age":
        age_order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        valid_slices = [a for a in age_order if a in valid_slices]

    dis_filtered = disagree[disagree[slice_col].isin(valid_slices)]
    ct = pd.crosstab(dis_filtered[slice_col], dis_filtered["outcome"]).reindex(valid_slices).fillna(0)
    totals = ct.sum(axis=1).astype(int).tolist()
    # Category labels carry n; small groups (n < 50) are flagged because their shares are noisy
    x_labels = [f"{s}<br><sup>n={t:,}{' ⚠' if t < 50 else ''}</sup>" for s, t in zip(valid_slices, totals)]

    fig = go.Figure()
    colors = {"Model A Wins": "#457B9D", "Model B Wins": "#E63946", "Both Wrong": "#495057"}

    for outcome in ["Model A Wins", "Model B Wins", "Both Wrong"]:
        if outcome in ct.columns:
            vals = ct[outcome].astype(int).tolist()
            pcts = [(v / t * 100) if t > 0 else 0 for v, t in zip(vals, totals)]
            hover = [f"{v:,} of {t:,} disagreements ({p:.1f}%)" for v, t, p in zip(vals, totals, pcts)]
            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=[round(p, 2) for p in pcts],
                    name=outcome,
                    marker_color=colors.get(outcome, "#999"),
                    text=[f"{p:.0f}%" if p >= 8 else "" for p in pcts],
                    textposition="inside",
                    textfont=dict(color="white", size=10),
                    customdata=hover,
                    hovertemplate="%{x}<br><b>" + outcome + "</b>: %{customdata}<extra></extra>",
                )
            )

    fig.add_hline(y=50, line=dict(color="white", width=1.5, dash="dot"))
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=50, r=16, t=42, b=56)},
        barmode="stack",
        height=370,
        title=dict(
            text=(f"<b>Who is right when A and B disagree?</b> ({len(disagree):,} conflicting stays, "
                  f"by {slice_col.replace('_', ' ')})<br><sup>Bars are 100%-normalised per group; "
                  "⚠ = fewer than 50 conflicts (treat as anecdotal)</sup>"),
            x=0.01,
            font=dict(size=13),
        ),
        xaxis=dict(title=None, tickfont=dict(size=10), **GRID),
        yaxis=dict(title="Share of conflicts (%)", range=[0, 100], dtick=25, **GRID),
        legend=dict(orientation="h", y=-0.24, x=0.0, font=dict(size=11)),
    )
    return fig


def subgroup_fairness(data: DashboardData, selected: set[int] | None, group_by: str) -> go.Figure:
    """Subgroup audit that controls for each group's own class mix.

    Raw accuracy is misleading across groups because the share of the majority class
    ("Uncomplicated / Other") differs between groups: a group with fewer 'Other' outcomes looks
    worse even if the model behaves identically. We therefore plot the *lift* of each model over
    that group's own majority-class baseline, with 95% Wilson confidence intervals.
    Lift < 0 means the model is worse than always predicting 'Other' for that group.
    """
    df = data.df
    sub = df.loc[df.index.isin(selected)] if selected else df
    grp_col = group_by if group_by in sub.columns else "race"

    vc = sub[grp_col].value_counts()
    valid_groups = vc[vc >= 30].index.tolist()
    if grp_col == "age":
        age_order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        valid_groups = [a for a in age_order if a in valid_groups]

    fig = go.Figure()
    x_labels = [f"{g}<br><sup>n={int(vc[g]):,}</sup>" for g in valid_groups]
    styles = {"model_a": ("Model A (Tabular KNN)", "#457B9D"), "model_b": ("Model B (ClinicalBERT KNN)", "#E63946")}

    for model, (name, color) in styles.items():
        lift, lo, hi, hover = [], [], [], []
        for g in valid_groups:
            gs = sub[sub[grp_col] == g]
            n = len(gs)
            k = int(gs[f"{model}_correct"].sum())
            base = float((gs["majority_pred_pathway"] == gs["true_next_pathway"]).mean())
            low, high = wilson_ci(k, n)
            acc = k / n
            lift.append(round((acc - base) * 100, 2))
            lo.append(round((acc - low) * 100, 2))
            hi.append(round((high - acc) * 100, 2))
            hover.append(f"accuracy {acc:.1%} vs. majority baseline {base:.1%} (n={n:,})")
        fig.add_trace(
            go.Bar(
                x=x_labels, y=lift, name=name, marker_color=color,
                error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo, thickness=1.2, width=3),
                customdata=hover,
                hovertemplate="%{x}<br><b>" + name + "</b><br>Lift: %{y:+.1f} pp<br>%{customdata}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line=dict(color="#212529", width=1.2))
    scope = f"Selection (n={len(sub):,})" if selected else f"Full cohort (n={len(sub):,})"
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=55, r=16, t=50, b=52)},
        barmode="group",
        height=360,
        title=dict(
            text=(f"<b>Accuracy lift over each group's own majority baseline, by {grp_col.replace('_', ' ')}</b>"
                  f" — <sup>{scope}; groups with n &lt; 30 hidden; whiskers = 95% Wilson CI</sup>"),
            x=0.01,
            font=dict(size=13),
        ),
        yaxis=dict(title="Lift (percentage points)", zeroline=False, **{k: v for k, v in GRID.items() if k != "zeroline"}),
        legend=dict(orientation="h", y=-0.22, x=0.0, font=dict(size=11)),
    )
    return fig


def cohort_profiler(data: DashboardData, selected: set[int] | None) -> go.Figure:
    """Normalized clinical complexity profile (Baseline = 100%) so disparate scales are comparable."""
    df = data.df
    metrics = [
        ("time_in_hospital", "Stay Length (days)"),
        ("num_medications", "Medications Count"),
        ("num_lab_procedures", "Lab Procedures"),
        ("number_diagnoses", "Diagnoses Count"),
        ("number_emergency", "Past Emergency Visits"),
        ("number_inpatient", "Past Inpatient Visits"),
    ]

    base_means = [float(df[col].mean()) for col, _ in metrics]
    if selected:
        sub = df.loc[df.index.isin(selected)]
        sel_means = [float(sub[col].mean()) for col, _ in metrics]
        title_text = (
            f"<b>Normalized Clinical Profile: Active Selection (n={len(sub):,}) vs. Baseline (100%)</b>"
            "<br><sup>Bars scaled to % of Hospital Baseline so small-scale counts (e.g. Emergency=0.33) and Labs (42.5) are comparable</sup>"
        )
    else:
        sel_means = base_means
        title_text = (
            f"<b>Normalized Clinical Profile: Hospital Baseline (n={len(df):,}, Index = 100%)</b>"
            "<br><sup>Tip: Lasso a cohort in Tab 1 (Studio) to compare its relative clinical elevation against the 100% baseline</sup>"
        )

    labels = [label for _, label in metrics]
    base_idx = [100.0 for _ in base_means]
    sel_idx = [(s / b * 100.0) if b > 0 else 100.0 for s, b in zip(sel_means, base_means)]
    diff_pcts = [idx - 100.0 for idx in sel_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=base_idx,
            orientation="h",
            name="Hospital Baseline (100%)",
            marker_color="#CED4DA",
            text=[f"Base: {b:.2f}" for b in base_means],
            textposition="inside",
            insidetextanchor="middle",
            customdata=base_means,
            hovertemplate="<b>%{y}</b><br>Hospital Baseline Mean: <b>%{customdata:.2f}</b> (100%)<extra></extra>",
        )
    )

    if selected:
        fig.add_trace(
            go.Bar(
                y=labels,
                x=sel_idx,
                orientation="h",
                name="Active Selection (% of Base)",
                marker_color=["#E76F51" if d >= 10.0 else ("#2A9D8F" if d <= -10.0 else "#1D3557") for d in diff_pcts],
                text=[f"{s:.2f} ({d:+.1f}%)" for s, d in zip(sel_means, diff_pcts)],
                textposition="outside",
                textfont=dict(size=11, color="#1D3557"),
                customdata=list(zip(sel_means, base_means, diff_pcts)),
                hovertemplate=(
                    "<b>%{y}</b><br>Selection Mean: <b>%{customdata[0]:.2f}</b> vs Base <b>%{customdata[1]:.2f}</b>"
                    "<br>Relative Shift: <b>%{customdata[2]:+.1f}%</b><extra></extra>"
                ),
            )
        )

    fig.add_vline(x=100.0, line_dash="dash", line_color="#1D3557", line_width=1.2)
    max_x = max(145.0, max(sel_idx) * 1.28)
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=155, r=65, t=48, b=45)},
        barmode="group",
        height=360,
        title=dict(text=title_text, x=0.01, font=dict(size=13)),
        xaxis=dict(title="Relative Level (% of Hospital Mean, Baseline = 100%)", range=[0, max_x], **GRID),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.22, x=0.0, font=dict(size=11)),
    )
    return fig


def cohort_pathway_breakdown(data: DashboardData, selected: set[int] | None) -> go.Figure:
    """Complication pathway distribution comparing selection to hospital baseline."""
    df = data.df
    base_dist = df["true_next_pathway"].value_counts(normalize=True)

    if selected:
        sub = df.loc[df.index.isin(selected)]
        sel_dist = sub["true_next_pathway"].value_counts(normalize=True).reindex(PATHWAYS).fillna(0)
        title_text = f"<b>Complication Outcomes: Selection vs. Baseline</b> (n={len(sub):,})"
    else:
        sel_dist = base_dist.reindex(PATHWAYS).fillna(0)
        title_text = f"<b>Complication Outcomes: Hospital Baseline</b> (n={len(df):,})<br><sup>Tip: Lasso a cohort in Tab 1 to see outcome distribution shifts</sup>"

    short_labels = [PATHWAY_SHORT[p] for p in PATHWAYS]
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=short_labels,
            y=(base_dist.reindex(PATHWAYS).fillna(0) * 100).round(1).tolist(),
            name="Baseline Distribution (%)",
            marker_color="#CED4DA",
            text=[f"{v:.1f}%" for v in (base_dist.reindex(PATHWAYS).fillna(0) * 100)],
            textposition="auto",
            hovertemplate="Baseline: <b>%{y:.1f}%</b> in %{x}<extra></extra>",
        )
    )

    if selected:
        fig.add_trace(
            go.Bar(
                x=short_labels,
                y=(sel_dist * 100).round(1).tolist(),
                name="Selection Distribution (%)",
                marker_color=[PATHWAY_COLORS[p] for p in PATHWAYS],
                text=[f"{v:.1f}%" for v in (sel_dist * 100)],
                textposition="outside",
                textfont=dict(size=11, color="#1D3557"),
                hovertemplate="Selection: <b>%{y:.1f}%</b> in %{x}<extra></extra>",
            )
        )

    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=50, r=20, t=42, b=45)},
        barmode="group",
        height=360,
        title=dict(text=title_text, x=0.01, font=dict(size=13)),
        yaxis=dict(title="Encounters (%)", range=[0, 105], dtick=20, **GRID),
        legend=dict(orientation="h", y=-0.22, x=0.0, font=dict(size=11)),
    )
    return fig
