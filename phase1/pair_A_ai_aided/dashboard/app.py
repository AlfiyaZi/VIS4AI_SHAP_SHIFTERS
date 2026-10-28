"""
Main Dash application for Pair A (AI-aided) Phase I analysis.

Coordinates four visual views connected by interactive selection:
- View 1: Jittered Multi-Channel Scatter Explorer (agreement vs. diagnoses count)
- View 2: Reordered Error & Confusion Breakdown Matrix (side-by-side heatmaps)
- View 3: Input-Space UMAP & Longitudinal Trajectories (drift across repeat visits)
- View 4: Counterfactual Neighbor Flips & Patient Inspector (drill-down into any encounter)
"""

from __future__ import annotations

import json
from pathlib import Path

import dash
from dash import Input, Output, State, ctx, dcc, html
import dash.exceptions
import numpy as np
import pandas as pd

import data_layer
import figures

app = dash.Dash(
    __name__,
    title="SHAP_SHIFTERS - Phase I Explorer",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Load data contract from shared/data
data = data_layer.load()
df = data.df
metrics = data.metrics

# Cohort summary stats
total_encounters = len(df)
unique_patients = df["patient_nbr"].nunique()
acc_a = (df["model_a_pred_pathway"] == df["true_next_pathway"]).mean()
acc_b = (df["model_b_pred_pathway"] == df["true_next_pathway"]).mean()
acc_maj = (df["majority_pred_pathway"] == df["true_next_pathway"]).mean()

# App Layout
app.layout = html.Div(
    style={
        "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        "backgroundColor": "#F8F9FA",
        "minHeight": "100vh",
        "padding": "16px 24px",
        "color": "#212529",
    },
    children=[
        # Header & Summary Banner
        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "20px 24px",
                "borderRadius": "8px",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                "marginBottom": "16px",
            },
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.Div([
                            html.H2(
                                "Phase I: Black-Box Patient Retrieval Explorer",
                                style={"margin": "0 0 6px 0", "fontSize": "22px", "fontWeight": "700", "color": "#1A1A1A"},
                            ),
                            html.P(
                                "Team SHAP_SHIFTERS | Pair A (AI-Aided) | JKU AI & Visualization Lab WS26",
                                style={"margin": 0, "fontSize": "13px", "color": "#6C757D"},
                            ),
                        ]),
                        html.Div(
                            style={"display": "flex", "gap": "16px", "textAlign": "right"},
                            children=[
                                html.Div([
                                    html.Span("Cohort Size", style={"fontSize": "11px", "color": "#6C757D", "display": "block"}),
                                    html.Strong(f"{total_encounters:,} visits ({unique_patients:,} pts)", style={"fontSize": "14px"}),
                                ]),
                                html.Div([
                                    html.Span("Model A Acc", style={"fontSize": "11px", "color": "#6C757D", "display": "block"}),
                                    html.Strong(f"{acc_a:.1%}", style={"fontSize": "14px", "color": "#4E79A7"}),
                                ]),
                                html.Div([
                                    html.Span("Model B Acc", style={"fontSize": "11px", "color": "#6C757D", "display": "block"}),
                                    html.Strong(f"{acc_b:.1%}", style={"fontSize": "14px", "color": "#E15759"}),
                                ]),
                                html.Div([
                                    html.Span("Majority Baseline", style={"fontSize": "11px", "color": "#6C757D", "display": "block"}),
                                    html.Strong(f"{acc_maj:.1%}", style={"fontSize": "14px", "color": "#555555"}),
                                ]),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Global Control Bar & Active Filter Status
        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "12px 20px",
                "borderRadius": "8px",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                "marginBottom": "16px",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "12px",
            },
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "16px"},
                    children=[
                        html.Button(
                            "Reset All Selections",
                            id="btn-reset",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#0D6EFD",
                                "color": "#FFFFFF",
                                "border": "none",
                                "padding": "6px 14px",
                                "borderRadius": "4px",
                                "cursor": "pointer",
                                "fontWeight": "600",
                                "fontSize": "13px",
                            },
                        ),
                        html.Div(id="selection-status-badge", style={"fontSize": "13px", "fontWeight": "500"}),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "16px", "fontSize": "13px"},
                    children=[
                        html.Span("Quick Encounter Inspect:"),
                        dcc.Dropdown(
                            id="encounter-inspect-dropdown",
                            options=[{"label": f"Encounter #{eid} ({df.loc[eid, 'true_next_pathway']})", "value": eid}
                                     for eid in df.index[:150]],
                            value=int(df.index[0]),
                            clearable=False,
                            style={"width": "260px"},
                        ),
                    ],
                ),
            ],
        ),

        # Store component for coordinated state (active selection of encounter IDs)
        dcc.Store(id="store-selected-ids", data=None),
        dcc.Store(id="store-active-cell", data=None),

        # Main Grid: Top Row (View 1 & View 2)
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "16px"},
            children=[
                # View 1 Container: Multi-Channel Jittered Scatter Explorer
                html.Div(
                    style={
                        "backgroundColor": "#FFFFFF",
                        "padding": "16px",
                        "borderRadius": "8px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                            children=[
                                html.H3("View 1: Jittered Multi-Channel Scatter Explorer", style={"margin": 0, "fontSize": "15px", "fontWeight": "600"}),
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "fontSize": "12px", "alignItems": "center"},
                                    children=[
                                        dcc.RadioItems(
                                            id="v1-model-x",
                                            options=[{"label": "Model B X-Axis", "value": "model_b"},
                                                     {"label": "Model A X-Axis", "value": "model_a"}],
                                            value="model_b",
                                            inline=True,
                                        ),
                                        dcc.Checklist(
                                            id="v1-options",
                                            options=[{"label": " Jitter", "value": "jitter"},
                                                     {"label": " Density Contours", "value": "contours"}],
                                            value=["jitter"],
                                            inline=True,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="graph-view1-scatter", config={"displayModeBar": True}),
                    ],
                ),

                # View 2 Container: Reordered Error & Confusion Breakdown Matrix
                html.Div(
                    style={
                        "backgroundColor": "#FFFFFF",
                        "padding": "16px",
                        "borderRadius": "8px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                            children=[
                                html.H3("View 2: Reordered Error & Confusion Breakdown Matrix", style={"margin": 0, "fontSize": "15px", "fontWeight": "600"}),
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "fontSize": "12px", "alignItems": "center"},
                                    children=[
                                        dcc.Checklist(
                                            id="v2-options",
                                            options=[{"label": " Clustered Order", "value": "clustered"},
                                                     {"label": " Row Normalize (%)", "value": "normalize"}],
                                            value=["clustered", "normalize"],
                                            inline=True,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "8px"},
                            children=[
                                dcc.Graph(id="graph-view2-model-a", config={"displayModeBar": False}),
                                dcc.Graph(id="graph-view2-model-b", config={"displayModeBar": False}),
                                dcc.Graph(id="graph-view2-majority", config={"displayModeBar": False}),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Main Grid: Bottom Row (View 3 & View 4)
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                # View 3 Container: Input-Space UMAP & Longitudinal Trajectories
                html.Div(
                    style={
                        "backgroundColor": "#FFFFFF",
                        "padding": "16px",
                        "borderRadius": "8px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                            children=[
                                html.H3("View 3: Input-Space UMAP & Longitudinal Trajectories", style={"margin": 0, "fontSize": "15px", "fontWeight": "600"}),
                                html.Div(
                                    style={"display": "flex", "gap": "8px", "alignItems": "center", "fontSize": "12px"},
                                    children=[
                                        html.Span("Color:"),
                                        dcc.Dropdown(
                                            id="v3-color-mode",
                                            options=[{"label": v, "value": k} for k, v in figures.UMAP_COLOR_MODES.items()],
                                            value="disparity",
                                            clearable=False,
                                            style={"width": "180px"},
                                        ),
                                        dcc.Checklist(
                                            id="v3-traj-toggle",
                                            options=[{"label": " Trajectories", "value": "traj"},
                                                     {"label": " Driver Annotations", "value": "annot"}],
                                            value=["annot"],
                                            inline=True,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="graph-view3-umap", config={"displayModeBar": True}),
                    ],
                ),

                # View 4 Container: Counterfactual Neighbor Flips & Patient Inspector
                html.Div(
                    style={
                        "backgroundColor": "#FFFFFF",
                        "padding": "16px",
                        "borderRadius": "8px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.H3("View 4: Counterfactual Neighbor Flips & Patient Inspector", style={"margin": "0 0 8px 0", "fontSize": "15px", "fontWeight": "600"}),
                        dcc.Graph(id="graph-view4-flips", config={"displayModeBar": False}),
                        html.Hr(style={"margin": "12px 0", "borderColor": "#E9ECEF"}),
                        # Inspection Detail Card
                        html.Div(id="view4-inspector-card"),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks: State Management & Cross-Filtering
# ---------------------------------------------------------------------------

@app.callback(
    Output("store-selected-ids", "data"),
    Output("store-active-cell", "data"),
    Input("btn-reset", "n_clicks"),
    Input("graph-view1-scatter", "selectedData"),
    Input("graph-view3-umap", "selectedData"),
    Input("graph-view2-model-a", "clickData"),
    Input("graph-view2-model-b", "clickData"),
    Input("graph-view2-majority", "clickData"),
    State("v2-options", "value"),
)
def handle_cross_selection(reset_clicks, v1_sel, v3_sel, v2_click_a, v2_click_b, v2_click_maj, v2_opts):
    triggered = ctx.triggered_id
    if not triggered or triggered == "btn-reset":
        return None, None

    # View 1 Scatter Lasso/Box selection
    if triggered == "graph-view1-scatter" and v1_sel and "points" in v1_sel:
        eids = [p["customdata"] for p in v1_sel["points"] if "customdata" in p]
        return eids if eids else None, None

    # View 3 UMAP Lasso/Box selection
    if triggered == "graph-view3-umap" and v3_sel and "points" in v3_sel:
        eids = [p["customdata"] for p in v3_sel["points"] if "customdata" in p]
        return eids if eids else None, None

    # View 2 Confusion Matrix Cell Click
    for click, model in [(v2_click_a, "model_a"), (v2_click_b, "model_b"), (v2_click_maj, "majority")]:
        if triggered == f"graph-view2-{model}" and click and "points" in click:
            pt = click["points"][0]
            if "customdata" in pt:
                true_cat, pred_cat = pt["customdata"][0], pt["customdata"][1]
                pred_col = "majority_pred_pathway" if model == "majority" else f"{model}_pred_pathway"
                m = (df["true_next_pathway"] == true_cat) & (df[pred_col] == pred_cat)
                eids = df.index[m].tolist()
                return eids, {"true": true_cat, "pred": pred_cat, "model": model}

    return None, None


@app.callback(
    Output("selection-status-badge", "children"),
    Input("store-selected-ids", "data"),
    Input("store-active-cell", "data"),
)
def update_status_badge(selected_ids, active_cell):
    if not selected_ids:
        return html.Span("Showing All Cohort Patients (No active filter)", style={"color": "#6C757D"})
    n = len(selected_ids)
    pct = (n / len(df)) * 100
    cell_info = ""
    if active_cell:
        cell_info = f" [Confusion: True '{active_cell['true']}' -> Pred '{active_cell['pred']}']"
    return html.Span(
        f"Filtered: {n:,} of {len(df):,} visits ({pct:.1f}%){cell_info}",
        style={"color": "#0D6EFD", "fontWeight": "600"},
    )


@app.callback(
    Output("graph-view1-scatter", "figure"),
    Input("store-selected-ids", "data"),
    Input("v1-model-x", "value"),
    Input("v1-options", "value"),
)
def render_view1(selected_ids, model_x, options):
    selected_set = set(selected_ids) if selected_ids else None
    jitter = "jitter" in (options or [])
    contours = "contours" in (options or [])
    return figures.scatter_explorer(data, selected_set, model_x, jitter, contours)


@app.callback(
    Output("graph-view2-model-a", "figure"),
    Output("graph-view2-model-b", "figure"),
    Output("graph-view2-majority", "figure"),
    Input("v2-options", "value"),
    Input("store-active-cell", "data"),
)
def render_view2(options, active_cell):
    clustered = "clustered" in (options or [])
    normalize = "normalize" in (options or [])
    hl_a = (active_cell["true"], active_cell["pred"]) if (active_cell and active_cell.get("model") == "model_a") else None
    hl_b = (active_cell["true"], active_cell["pred"]) if (active_cell and active_cell.get("model") == "model_b") else None
    hl_maj = (active_cell["true"], active_cell["pred"]) if (active_cell and active_cell.get("model") == "majority") else None

    fig_a = figures.confusion_heatmap(data, "model_a", normalize, clustered, hl_a)
    fig_b = figures.confusion_heatmap(data, "model_b", normalize, clustered, hl_b)
    fig_maj = figures.confusion_heatmap(data, "majority", normalize, clustered, hl_maj)
    return fig_a, fig_b, fig_maj


@app.callback(
    Output("graph-view3-umap", "figure"),
    Input("store-selected-ids", "data"),
    Input("v3-color-mode", "value"),
    Input("v3-traj-toggle", "value"),
)
def render_view3(selected_ids, color_mode, traj_options):
    selected_set = set(selected_ids) if selected_ids else None
    show_traj = "traj" in (traj_options or [])
    annotate = "annot" in (traj_options or [])
    return figures.umap_view(data, selected_set, color_mode, show_traj, n_patients=35, annotate=annotate)


@app.callback(
    Output("graph-view4-flips", "figure"),
    Input("store-selected-ids", "data"),
)
def render_view4_flips(selected_ids):
    selected_set = set(selected_ids) if selected_ids else None
    return figures.flips_view(data, selected_set)


@app.callback(
    Output("view4-inspector-card", "children"),
    Input("encounter-inspect-dropdown", "value"),
    Input("graph-view1-scatter", "clickData"),
    Input("graph-view3-umap", "clickData"),
)
def render_patient_inspector(dropdown_eid, v1_click, v3_click):
    eid = dropdown_eid
    # Allow clicking points in scatter or UMAP to select encounter
    if ctx.triggered_id == "graph-view1-scatter" and v1_click and "points" in v1_click:
        eid = v1_click["points"][0].get("customdata", eid)
    elif ctx.triggered_id == "graph-view3-umap" and v3_click and "points" in v3_click:
        eid = v3_click["points"][0].get("customdata", eid)

    if eid not in df.index:
        eid = int(df.index[0])

    row = df.loc[eid]
    neigh_fig = figures.neighbor_breakdown(data, eid)

    # Demographic & clinical badges
    return html.Div([
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "10px"},
            children=[
                html.Div([
                    html.H4(f"Encounter #{eid} (Patient #{row['patient_nbr']}, Visit #{row['visit_index']})",
                            style={"margin": "0 0 4px 0", "fontSize": "14px", "fontWeight": "700"}),
                    html.Span(f"Serialized Note: \"{row['model_text']}\"",
                              style={"fontSize": "11px", "color": "#495057", "fontStyle": "italic"}),
                ]),
                html.Div(
                    style={"textAlign": "right"},
                    children=[
                        html.Span("True Next Pathway:", style={"fontSize": "11px", "color": "#6C757D", "display": "block"}),
                        html.Span(
                            row["true_next_pathway"],
                            style={
                                "backgroundColor": data_layer.PATHWAY_COLORS.get(row["true_next_pathway"], "#666"),
                                "color": "#FFF",
                                "padding": "2px 8px",
                                "borderRadius": "4px",
                                "fontWeight": "600",
                                "fontSize": "12px",
                            },
                        ),
                    ],
                ),
            ],
        ),

        # Retrieved Neighbors Stacked Breakdown
        dcc.Graph(figure=neigh_fig, config={"displayModeBar": False}),

        # Raw Encounter Fields Grid
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "8px",
                "marginTop": "10px",
                "fontSize": "11px",
                "backgroundColor": "#F1F3F5",
                "padding": "10px",
                "borderRadius": "4px",
            },
            children=[
                html.Div([html.Strong("Model A Pred: "), html.Span(f"{row['model_a_pred_pathway']} (Agr: {row['model_a_agreement']:.2f})")]),
                html.Div([html.Strong("Model B Pred: "), html.Span(f"{row['model_b_pred_pathway']} (Agr: {row['model_b_agreement']:.2f})")]),
                html.Div([html.Strong("Gender / Age: "), html.Span(f"{row['gender']} / {row['age']}")]),
                html.Div([html.Strong("HbA1c / Glu: "), html.Span(f"{row['A1Cresult']} / {row['max_glu_serum']}")]),
                html.Div([html.Strong("Stay (Days): "), html.Span(f"{row['time_in_hospital']}")]),
                html.Div([html.Strong("Labs / Meds: "), html.Span(f"{row['num_lab_procedures']} / {row['num_medications']}")]),
                html.Div([html.Strong("Diagnoses: "), html.Span(f"{row['number_diagnoses']}")]),
                html.Div([html.Strong("Insulin: "), html.Span(f"{row['insulin']}")]),
            ],
        ),
    ])


if __name__ == "__main__":
    app.run(debug=True, port=8050)
