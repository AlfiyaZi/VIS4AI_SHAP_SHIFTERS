"""Figure builders for the four coordinated views of the Pair A dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_layer import (
    A1C_COLORS, A1C_ORDER, EDITS, MODELS, PATHWAY_COLORS, PATHWAY_SHORT, PATHWAYS,
    SEX_SYMBOLS, DashboardData,
)

FONT = dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color="#222")
BASE_LAYOUT = dict(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                   margin=dict(l=50, r=20, t=60, b=45))
GRID = dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False)
DIM_OPACITY = 0.06


def _selected_positions(ids_in_trace: np.ndarray, selected: set[int] | None):
    if selected is None:
        return None
    return [i for i, e in enumerate(ids_in_trace) if e in selected]


# ---------------------------------------------------------------------------
# View 1: jittered multi-channel scatter
# ---------------------------------------------------------------------------
def scatter_explorer(data: DashboardData, selected: set[int] | None, x_model: str,
                     jitter: bool, contours: bool) -> go.Figure:
    df = data.df
    xcol = f"{x_model}_agreement"
    xlab = "Model B" if x_model == "model_b" else "Model A"
    fig = make_subplots(
        rows=1, cols=len(PATHWAYS), shared_yaxes=True, horizontal_spacing=0.025,
        subplot_titles=[f"{p}  (n={int((df['true_next_pathway'] == p).sum()):,})" for p in PATHWAYS],
    )
    for col, pathway in enumerate(PATHWAYS, start=1):
        sub = df[df["true_next_pathway"] == pathway]
        x = sub[xcol] + (sub["jitter_x"] if jitter else 0)
        y = sub["number_diagnoses"] + (sub["jitter_y"] if jitter else 0)
        for a1c in A1C_ORDER:
            for sex, symbol in SEX_SYMBOLS.items():
                m = (sub["A1Cresult"] == a1c) & (sub["gender"] == sex)
                if not m.any():
                    continue
                ids = sub.index[m].to_numpy()
                fig.add_trace(go.Scattergl(
                    x=x[m], y=y[m], mode="markers", customdata=ids,
                    name=f"HbA1c {a1c} | {sex}", legendgroup=f"{a1c}|{sex}",
                    showlegend=col == 1,
                    marker=dict(color=A1C_COLORS[a1c], symbol=symbol, size=5, opacity=0.55,
                                line=dict(width=0)),
                    selectedpoints=_selected_positions(ids, selected),
                    selected=dict(marker=dict(opacity=0.9)),
                    unselected=dict(marker=dict(opacity=DIM_OPACITY)),
                    hovertemplate=(f"encounter %{{customdata}}<br>{xlab} agreement %{{x:.2f}}"
                                   "<br>diagnoses %{y:.0f}<extra>" + f"HbA1c {a1c}, {sex}</extra>"),
                ), row=1, col=col)
        if contours and len(sub) > 20:
            fig.add_trace(go.Histogram2dContour(
                x=sub[xcol] + sub["jitter_x"], y=sub["number_diagnoses"] + sub["jitter_y"],
                ncontours=8, contours=dict(coloring="none"), showscale=False,
                line=dict(color="rgba(20,20,20,0.75)", width=1), hoverinfo="skip",
                showlegend=False, xbins=dict(size=0.1), ybins=dict(size=1),
            ), row=1, col=col)
        fig.update_xaxes(range=[-0.06, 1.06], dtick=0.2, row=1, col=col, **GRID)
    fig.update_yaxes(title_text="Number of diagnoses (comorbidity count)", row=1, col=1)
    fig.update_yaxes(**GRID)
    fig.update_layout(
        **BASE_LAYOUT, height=430, dragmode="lasso", uirevision=f"scatter-{x_model}",
        title=dict(text=f"<b>View 1</b> - {xlab} neighbor agreement (k=10) vs. comorbidity, "
                        "faceted by true next pathway", x=0.01),
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=11)),
    )
    fig.add_annotation(text=f"{xlab} neighbor agreement (share of 10 neighbors with the same next pathway)",
                       x=0.5, y=-0.13, xref="paper", yref="paper", showarrow=False)
    return fig


# ---------------------------------------------------------------------------
# View 2: reordered confusion matrices
# ---------------------------------------------------------------------------
def confusion_heatmap(data: DashboardData, model: str, normalize: bool, clustered: bool,
                      highlight: tuple[str, str] | None) -> go.Figure:
    order = data.order if clustered else PATHWAYS
    counts = data.confusion[model].loc[order, order]
    rates = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    z = rates.to_numpy() if normalize else counts.to_numpy()
    text = [[f"{rates.iat[i, j]:.0%}<br>({counts.iat[i, j]:,})" for j in range(len(order))]
            for i in range(len(order))]
    short = [PATHWAY_SHORT[p] for p in order]

    df = data.df
    pred_col = "majority_pred_pathway" if model == "majority" else f"{model}_pred_pathway"
    acc = (df[pred_col] == df["true_next_pathway"]).mean()
    recalls = [rates.loc[p, p] for p in order if counts.loc[p].sum() > 0]
    fig = go.Figure(go.Heatmap(
        z=z, x=short, y=short, text=text, texttemplate="%{text}", textfont=dict(size=11),
        colorscale="Blues", showscale=False, zmin=0, zmax=1 if normalize else None,
        customdata=np.array([[[t, p] for p in order] for t in order]),
        hovertemplate="true %{customdata[0]}<br>predicted %{customdata[1]}<br>%{text}<extra></extra>",
        xgap=2, ygap=2,
    ))
    for i in range(len(order)):  # outline the diagonal (correct) cells
        fig.add_shape(type="rect", x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                      line=dict(color="#333", width=1.2, dash="dot"))
    if highlight:
        t, p = highlight
        if t in order and p in order:
            i, j = order.index(t), order.index(p)
            fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                          line=dict(color="#E8590C", width=3))
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=75, r=10, t=70, b=55)}, height=360,
        title=dict(text=f"<b>{MODELS[model]}</b><br><sup>accuracy {acc:.1%} | "
                        f"macro recall {np.mean(recalls):.1%}</sup>", x=0.02),
        xaxis=dict(title="predicted (neighbor vote)", side="bottom"),
        yaxis=dict(title="true next pathway", autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# View 3: input-space UMAP + trajectories
# ---------------------------------------------------------------------------
UMAP_COLOR_MODES = {
    "disparity": "Agreement disparity (B - A)",
    "model_a_agreement": "Model A agreement",
    "model_b_agreement": "Model B agreement",
    "true_next_pathway": "True next pathway",
    "driver": "Primary cluster driver",
    "umap_cluster": "HDBSCAN cluster",
}


def umap_view(data: DashboardData, selected: set[int] | None, color_mode: str,
              show_traj: bool, n_patients: int, annotate: bool) -> go.Figure:
    df = data.df
    fig = go.Figure()
    driver = data.driver_ranking.iloc[0]["feature"]

    def add_points(sub: pd.DataFrame, name: str, marker: dict, showlegend: bool, hover: str):
        ids = sub.index.to_numpy()
        fig.add_trace(go.Scattergl(
            x=sub["umap_x"], y=sub["umap_y"], mode="markers", customdata=ids, name=name,
            showlegend=showlegend, marker=marker,
            selectedpoints=_selected_positions(ids, selected),
            selected=dict(marker=dict(opacity=0.95)),
            unselected=dict(marker=dict(opacity=DIM_OPACITY)),
            hovertemplate="encounter %{customdata}<br>" + hover + "<extra></extra>",
        ))

    if color_mode in ("disparity", "model_a_agreement", "model_b_agreement"):
        col = "agreement_disparity_b_minus_a" if color_mode == "disparity" else color_mode
        is_div = color_mode == "disparity"
        add_points(df, UMAP_COLOR_MODES[color_mode], dict(
            color=df[col], colorscale="RdBu" if is_div else "Viridis",
            cmin=-1 if is_div else 0, cmax=1, cmid=0 if is_div else None, size=4, opacity=0.6,
            colorbar=dict(title=dict(text="B - A" if is_div else "agreement", side="right"),
                          thickness=12, len=0.75),
        ), False, f"{UMAP_COLOR_MODES[color_mode]} %{{marker.color:.2f}}")
    else:
        if color_mode == "true_next_pathway":
            groups, palette = PATHWAYS, PATHWAY_COLORS
            key = df["true_next_pathway"]
        elif color_mode == "umap_cluster":
            key = df["umap_cluster"].map(lambda c: "noise" if c < 0 else f"C{c}")
            groups = sorted(key.unique(), key=lambda s: (s == "noise", len(s), s))
            palette = _qualitative(groups)
        else:
            key = df[driver].astype(str)
            top = key.value_counts().index[:9].tolist()
            key = key.where(key.isin(top), "other values")
            groups = top + (["other values"] if (key == "other values").any() else [])
            palette = _qualitative(groups)
        for g in groups:
            sub = df[key == g]
            if len(sub):
                add_points(sub, str(g), dict(color=palette[g], size=4, opacity=0.6), True, str(g))

    if show_traj:
        _add_trajectories(fig, df, selected, n_patients)

    if annotate and len(data.cluster_summary):
        for _, r in data.cluster_summary.iterrows():
            fig.add_annotation(
                x=r["x"], y=r["y"], text=f"<b>C{r['cluster']}</b> {r['label']}",
                showarrow=False, font=dict(size=10, color="#111"),
                bgcolor="rgba(255,255,255,0.78)", bordercolor="#666", borderwidth=0.5,
            )
    top = data.driver_ranking.iloc[0]
    fig.update_layout(
        **BASE_LAYOUT, height=560, dragmode="lasso", uirevision="umap",
        title=dict(text=("<b>View 3</b> - UMAP of input features (black box closed)<br>"
                         f"<sup>primary driver of cluster separation: <b>{top['feature']}</b> "
                         f"({top['kind'].split(' ')[0]}, association {top['association']:.2f})</sup>"),
                   x=0.01),
        xaxis=dict(showticklabels=False, title="UMAP 1", **GRID),
        yaxis=dict(showticklabels=False, title="UMAP 2", **GRID),
        legend=dict(font=dict(size=10), itemsizing="constant"),
    )
    return fig


def _qualitative(groups: list[str]) -> dict[str, str]:
    base = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1",
            "#FF9DA7", "#9C755F", "#BAB0AC", "#1F77B4", "#2CA02C", "#D62728", "#9467BD"]
    out = {g: base[i % len(base)] for i, g in enumerate(groups)}
    for g in ("noise", "other values"):
        if g in out:
            out[g] = "#D0D0D0"
    return out


def _add_trajectories(fig: go.Figure, df: pd.DataFrame, selected: set[int] | None, n: int):
    """Directed V1 -> V2 -> V3 paths of repeat admissions (patients with >= 2 labelled visits)."""
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
        xs += g["umap_x"].tolist() + [None]
        ys += g["umap_y"].tolist() + [None]
        txt += [f"patient {pid}, visit {v}" for v in g["visit_index"]] + [None]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name=f"trajectories ({len(pids)} patients)",
        line=dict(color="rgba(30,30,30,0.45)", width=1),
        marker=dict(symbol="arrow", angleref="previous", size=9, color="rgba(30,30,30,0.75)"),
        text=txt, hovertemplate="%{text}<extra></extra>",
    ))


# ---------------------------------------------------------------------------
# View 4: counterfactual flips + inspector chart
# ---------------------------------------------------------------------------
def flips_view(data: DashboardData, selected: set[int] | None) -> go.Figure:
    df = data.df
    sub = df.loc[df.index.isin(selected)] if selected else df
    fig = go.Figure()
    colors = {"model_a": "#4E79A7", "model_b": "#E15759"}
    for model in ("model_a", "model_b"):
        means, errs, ref, ns = [], [], [], []
        for edit in EDITS:
            v = sub[f"{model}_flips_{edit}"].dropna().astype(float)
            means.append(v.mean() if len(v) else np.nan)
            errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
            ns.append(len(v))
            ref.append(df[f"{model}_flips_{edit}"].dropna().astype(float).mean())
        fig.add_trace(go.Bar(
            x=list(EDITS.values()), y=means, name=MODELS[model], marker_color=colors[model],
            error_y=dict(type="data", array=errs, thickness=1.2, width=4),
            customdata=ns, hovertemplate="%{x}<br>mean flips %{y:.2f} / 10<br>n=%{customdata:,}<extra>"
                                         + MODELS[model] + "</extra>",
        ))
        if selected:
            fig.add_trace(go.Scatter(
                x=list(EDITS.values()), y=ref, mode="markers", showlegend=model == "model_a",
                name="whole cohort", marker=dict(symbol="line-ew-open", size=26,
                                                 line=dict(width=2.5, color="#111")),
                offsetgroup=model, hovertemplate="whole cohort %{y:.2f}<extra></extra>",
            ))
    scope = f"selection (n={len(sub):,})" if selected else f"whole cohort (n={len(sub):,})"
    fig.update_layout(
        **BASE_LAYOUT, height=360, barmode="group", scattermode="group",
        title=dict(text=f"<b>View 4</b> - Mean neighbor flips under edits<br><sup>{scope}; "
                        "95% CI; age/HbA1c only where the edit applies</sup>", x=0.01),
        yaxis=dict(title="neighbors replaced (of 10)", range=[0, 10.4], **GRID),
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    return fig


def neighbor_breakdown(data: DashboardData, eid: int) -> go.Figure:
    row = data.df.loc[eid]
    true = row["true_next_pathway"]
    fig = go.Figure()
    labels = {"model_a": "Model A", "model_b": "Model B", "random": "Random"}
    for p in PATHWAYS:
        vals = []
        for m in labels:
            nb = [int(n) for n in row[f"{m}_neighbors"]]
            path = data.df.loc[nb, "true_next_pathway"]
            vals.append(int((path == p).sum()))
        fig.add_trace(go.Bar(
            y=list(labels.values()), x=vals, orientation="h", name=PATHWAY_SHORT[p],
            marker=dict(color=PATHWAY_COLORS[p],
                        line=dict(color="#111" if p == true else "white", width=2 if p == true else 0.5)),
            hovertemplate="%{y}: %{x} neighbors " + p + "<extra></extra>",
        ))
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(l=70, r=10, t=40, b=30)}, barmode="stack", height=210,
        title=dict(text=f"Neighbor next pathways (true: <b>{PATHWAY_SHORT[true]}</b>, outlined)",
                   x=0.01, font=dict(size=12)),
        xaxis=dict(range=[0, 10], dtick=1, title=None, **GRID),
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
    )
    return fig
