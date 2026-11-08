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


# Confusion matrices
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
        [f"{rates.iat[i, j]:.0%}" for j in range(len(order))]
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
            textfont=dict(size=13, family="Inter, system-ui, sans-serif"),
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
        **{**BASE_LAYOUT, "margin": dict(l=left_margin + 14, r=12, t=55, b=65)},
        height=440,
        title=dict(
            text=(
                f"<b>{short_titles.get(model, MODELS[model])}</b><br>"
                f"<sup>Acc: <b>{acc:.1%}</b> | Bal. Recall: <b>{macro_rec:.1%}</b></sup>"
            ),
            x=0.02,
            font=dict(size=12),
        ),
        xaxis=dict(title=dict(text="Predicted", font=dict(size=10.5)), side="bottom", tickfont=dict(size=11)),
        yaxis=dict(
            title=dict(text="True Pathway" if show_y_title else None, font=dict(size=10.5)),
            autorange="reversed",
            tickfont=dict(size=11),
        ),
    )
    return fig



# Counterfactual sensitivity
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



# Reliability and subgroup analyses
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
    fig = go.Figure()
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
        )

    # Reference lines: what you get WITHOUT any retrieval
    maj = float((sub["majority_pred_pathway"] == sub["true_next_pathway"]).mean() * 100)
    rnd = float((sub["random_pred_pathway"] == sub["true_next_pathway"]).mean() * 100)
    for y, text, color in ((maj, f"Majority baseline {maj:.1f}%", "#495057"),):
        fig.add_hline(y=y, line=dict(color=color, dash="dash", width=1.3),
                      annotation_text=text, annotation_position="bottom right",
                      annotation_font=dict(size=10, color=color))

    scope = f"Selection (n={len(sub):,})" if selected else f"Full cohort (n={len(sub):,})"
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=55, r=55, t=42, b=48)},
        height=370,
        title=dict(text=f"<b>Does Neighbor Consensus Predict Accuracy?</b> — <sup>{scope}</sup>",
                   x=0.01, font=dict(size=13)),
        xaxis=dict(title="Vote share = neighbors (of 10) voting for the predicted pathway", **GRID),
        legend=dict(orientation="h", y=-0.24, x=0.0, font=dict(size=10.5)),
    )
    fig.update_yaxes(title_text="Observed accuracy (%)", range=[0, 100], dtick=20)
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


