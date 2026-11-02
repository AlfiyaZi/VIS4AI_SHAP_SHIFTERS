"""
Callbacks for the Pair A (AI-aided) Phase I dashboard.
Coordinates state management and cross-filtering between the 4 views.
"""

from __future__ import annotations

from dash import Input, Output, State, ctx, dcc, html
import numpy as np

import data_layer
import figures


def register_callbacks(app, data: data_layer.DashboardData):
    df = data.df

    # -----------------------------------------------------------------------
    # Coordinated State: Cross-Filtering & Selections
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Status Badge
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # View 1: Jittered Multi-Channel Scatter Explorer
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # View 2: Reordered Error & Confusion Breakdown Matrix
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # View 3: Input-Space UMAP & Longitudinal Trajectories
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # View 4: Counterfactual Neighbor Flips
    # -----------------------------------------------------------------------
    @app.callback(
        Output("graph-view4-flips", "figure"),
        Input("store-selected-ids", "data"),
    )
    def render_view4_flips(selected_ids):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.flips_view(data, selected_set)

    # -----------------------------------------------------------------------
    # View 4: Patient Inspector Card
    # -----------------------------------------------------------------------
    @app.callback(
        Output("view4-inspector-card", "children"),
        Input("encounter-inspect-dropdown", "value"),
        Input("graph-view1-scatter", "clickData"),
        Input("graph-view3-umap", "clickData"),
    )
    def render_patient_inspector(dropdown_eid, v1_click, v3_click):
        eid = dropdown_eid
        if ctx.triggered_id == "graph-view1-scatter" and v1_click and "points" in v1_click:
            eid = v1_click["points"][0].get("customdata", eid)
        elif ctx.triggered_id == "graph-view3-umap" and v3_click and "points" in v3_click:
            eid = v3_click["points"][0].get("customdata", eid)

        if eid not in df.index:
            eid = int(df.index[0])

        row = df.loc[eid]
        neigh_fig = figures.neighbor_breakdown(data, eid)

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
