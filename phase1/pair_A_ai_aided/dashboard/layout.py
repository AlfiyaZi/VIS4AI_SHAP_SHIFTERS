"""
Layout components for the Pair A (AI-aided) Phase I dashboard.
"""

from __future__ import annotations

from dash import dcc, html

from data_layer import DashboardData
import figures


def create_header_banner(df, metrics: dict) -> html.Div:
    total_encounters = len(df)
    unique_patients = df["patient_nbr"].nunique()
    acc_a = (df["model_a_pred_pathway"] == df["true_next_pathway"]).mean()
    acc_b = (df["model_b_pred_pathway"] == df["true_next_pathway"]).mean()
    acc_maj = (df["majority_pred_pathway"] == df["true_next_pathway"]).mean()

    return html.Div(
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
    )


def create_global_toolbar(df) -> html.Div:
    sample_eids = df.index[:150]
    return html.Div(
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
                                 for eid in sample_eids],
                        value=int(sample_eids[0]),
                        clearable=False,
                        style={"width": "260px"},
                    ),
                ],
            ),
        ],
    )


def create_view1_container() -> html.Div:
    return html.Div(
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
    )


def create_view2_container() -> html.Div:
    return html.Div(
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
    )


def create_view3_container() -> html.Div:
    return html.Div(
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
    )


def create_view4_container() -> html.Div:
    return html.Div(
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
            html.Div(id="view4-inspector-card"),
        ],
    )


def build_layout(data: DashboardData) -> html.Div:
    df = data.df
    metrics = data.metrics

    return html.Div(
        style={
            "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
            "backgroundColor": "#F8F9FA",
            "minHeight": "100vh",
            "padding": "16px 24px",
            "color": "#212529",
        },
        children=[
            # Header & Summary Banner
            create_header_banner(df, metrics),

            # Global Control Bar & Active Filter Status
            create_global_toolbar(df),

            # Store component for coordinated state
            dcc.Store(id="store-selected-ids", data=None),
            dcc.Store(id="store-active-cell", data=None),

            # Main Grid: Top Row (View 1 & View 2)
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "16px"},
                children=[
                    create_view1_container(),
                    create_view2_container(),
                ],
            ),

            # Main Grid: Bottom Row (View 3 & View 4)
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                children=[
                    create_view3_container(),
                    create_view4_container(),
                ],
            ),
        ],
    )
