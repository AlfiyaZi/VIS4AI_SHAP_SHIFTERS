"""
Callbacks for the Pair A (AI-aided) Phase I dashboard.
Coordinates state management, selection revisions, and cross-filtering across Views 1, 2, 3, and 4.
"""

from __future__ import annotations

from dash import Input, Output, State, ctx, dcc, html, no_update
import numpy as np
import figures
import data_layer


def _extract_eid(pt: dict) -> int | None:
    """Safely extract integer encounter_id from a Plotly point dictionary."""
    if not pt or "customdata" not in pt or pt["customdata"] is None:
        return None
    cd = pt["customdata"]
    if isinstance(cd, (list, tuple, np.ndarray)):
        return int(cd[0]) if len(cd) > 0 else None
    try:
        return int(cd)
    except (ValueError, TypeError):
        return None


def register_callbacks(app, data: data_layer.DashboardData):
    df = data.df

    # -----------------------------------------------------------------------
    # Coordinated State: Cross-Filtering & Selection Revision Tracking
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
        State("store-active-cell", "data"),
    )
    def handle_cross_selection(
        reset_clicks,
        v1_sel,
        v3_sel,
        v2_click_a,
        v2_click_b,
        v2_click_maj,
        prev_state,
    ):
        prev = prev_state or {"v1_rev": 0, "v3_rev": 0}
        v1_rev = int(prev.get("v1_rev", 0))
        v3_rev = int(prev.get("v3_rev", 0))

        triggered = ctx.triggered_id

        # 1. Initial page load
        if not triggered:
            return None, {"source": "reset", "v1_rev": 0, "v3_rev": 0}

        # 2. Reset button explicitly clicked
        if triggered == "btn-reset":
            return None, {"source": "reset", "v1_rev": v1_rev + 1, "v3_rev": v3_rev + 1}

        # 3. View 1 Scatter Lasso/Box selection
        if triggered == "graph-view1-scatter":
            if v1_sel and "points" in v1_sel and len(v1_sel["points"]) > 0:
                eids = [_extract_eid(p) for p in v1_sel["points"]]
                valid_eids = list(dict.fromkeys(e for e in eids if e is not None and e in df.index))
                if valid_eids:
                    # Keep v1_rev steady so View 1 selection stays aligned; bump v3_rev to clear stale View 3 box
                    return valid_eids, {"source": "scatter", "v1_rev": v1_rev, "v3_rev": v3_rev + 1}
            # When figure updates or user clears without button, do not automatically wipe state
            return no_update, no_update

        # 4. View 3 UMAP Lasso/Box selection
        if triggered == "graph-view3-umap":
            if v3_sel and "points" in v3_sel and len(v3_sel["points"]) > 0:
                eids = [_extract_eid(p) for p in v3_sel["points"]]
                valid_eids = list(dict.fromkeys(e for e in eids if e is not None and e in df.index))
                if valid_eids:
                    # Keep v3_rev steady so View 3 lasso stays aligned; bump v1_rev to clear stale View 1 box
                    return valid_eids, {"source": "umap", "v1_rev": v1_rev + 1, "v3_rev": v3_rev}
            # When figure updates or user clears without button, do not automatically wipe state
            return no_update, no_update

        # 5. View 2 Confusion Matrix Cell Click
        for click, model, comp_id in [
            (v2_click_a, "model_a", "graph-view2-model-a"),
            (v2_click_b, "model_b", "graph-view2-model-b"),
            (v2_click_maj, "majority", "graph-view2-majority"),
        ]:
            if triggered == comp_id and click and "points" in click:
                pt = click["points"][0]
                if "customdata" in pt and pt["customdata"] is not None:
                    cd = pt["customdata"]
                    true_cat = str(cd[0])
                    pred_cat = str(cd[1])
                    # Toggle off if clicking the exact same active cell again
                    if (
                        prev.get("source") == "confusion"
                        and prev.get("model") == model
                        and prev.get("true") == true_cat
                        and prev.get("pred") == pred_cat
                    ):
                        return None, {"source": "reset", "v1_rev": v1_rev + 1, "v3_rev": v3_rev + 1}

                    pred_col = "majority_pred_pathway" if model == "majority" else f"{model}_pred_pathway"
                    mask = (df["true_next_pathway"] == true_cat) & (df[pred_col] == pred_cat)
                    eids = [int(e) for e in df.index[mask]]
                    return (
                        eids,
                        {
                            "source": "confusion",
                            "true": true_cat,
                            "pred": pred_cat,
                            "model": model,
                            "v1_rev": v1_rev + 1,
                            "v3_rev": v3_rev + 1,
                        },
                    )

        return no_update, no_update

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
            return html.Span(
                f"● Showing Full Cohort ({len(df):,} hospital stays across {df['patient_nbr'].nunique():,} patients)",
                style={"color": "#198754", "fontWeight": "600", "fontSize": "12.5px"},
            )
        n = len(selected_ids)
        pct = (n / len(df)) * 100.0

        source = (active_cell or {}).get("source", "")
        if source == "confusion":
            m_label = data_layer.MODELS.get(active_cell.get("model", ""), active_cell.get("model", ""))
            desc = f"View 2 Cell: True '{active_cell['true']}' → Predicted '{active_cell['pred']}' ({m_label})"
        elif source == "scatter":
            desc = "View 1 Scatter Lasso / Box Selection"
        elif source == "umap":
            desc = "View 3 UMAP Lasso / Box Selection"
        else:
            desc = "Active Cohort Filter"

        return html.Span(
            f"🔍 Active Filter: {n:,} of {len(df):,} visits ({pct:.1f}%) — {desc}",
            style={"color": "#0D6EFD", "fontWeight": "700", "fontSize": "12.5px"},
        )

    # -----------------------------------------------------------------------
    # View 1: Jittered Multi-Channel Scatter Explorer
    # -----------------------------------------------------------------------
    @app.callback(
        Output("graph-view1-scatter", "figure"),
        Input("store-selected-ids", "data"),
        Input("store-active-cell", "data"),
        Input("v1-model-x", "value"),
        Input("v1-options", "value"),
    )
    def render_view1(selected_ids, active_cell, model_x, options):
        selected_set = set(selected_ids) if selected_ids else None
        jitter = "jitter" in (options or [])
        contours = "contours" in (options or [])
        v1_rev = str((active_cell or {}).get("v1_rev", 0))
        return figures.scatter_explorer(
            data,
            selected_set,
            model_x or "model_b",
            jitter,
            contours,
            uirevision_key=v1_rev,
        )

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
        is_conf = active_cell and active_cell.get("source") == "confusion"
        hl_a = (active_cell["true"], active_cell["pred"]) if (is_conf and active_cell.get("model") == "model_a") else None
        hl_b = (active_cell["true"], active_cell["pred"]) if (is_conf and active_cell.get("model") == "model_b") else None
        hl_maj = (active_cell["true"], active_cell["pred"]) if (is_conf and active_cell.get("model") == "majority") else None

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
        Input("store-active-cell", "data"),
        Input("v3-color-mode", "value"),
        Input("v3-traj-toggle", "value"),
    )
    def render_view3(selected_ids, active_cell, color_mode, traj_options):
        selected_set = set(selected_ids) if selected_ids else None
        show_traj = "traj" in (traj_options or [])
        annotate = "annot" in (traj_options or [])
        v3_rev = str((active_cell or {}).get("v3_rev", 0))
        return figures.umap_view(
            data,
            selected_set,
            color_mode or "driver",
            show_traj,
            n_patients=30,
            annotate=annotate,
            uirevision_key=v3_rev,
        )

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
        triggered = ctx.triggered_id
        if triggered == "graph-view1-scatter" and v1_click and "points" in v1_click:
            extracted = _extract_eid(v1_click["points"][0])
            if extracted is not None:
                eid = extracted
        elif triggered == "graph-view3-umap" and v3_click and "points" in v3_click:
            extracted = _extract_eid(v3_click["points"][0])
            if extracted is not None:
                eid = extracted

        if eid not in df.index:
            eid = int(df.index[0])

        row = df.loc[eid]
        neigh_fig = figures.neighbor_breakdown(data, eid)
        true_pathway = str(row["true_next_pathway"])
        pred_a = str(row["model_a_pred_pathway"])
        pred_b = str(row["model_b_pred_pathway"])
        ok_a = "✅" if pred_a == true_pathway else "❌"
        ok_b = "✅" if pred_b == true_pathway else "❌"

        return html.Div(
            style={
                "backgroundColor": "#F8F9FA",
                "borderRadius": "8px",
                "border": "1px solid #DEE2E6",
                "padding": "12px 14px",
            },
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "flexWrap": "wrap",
                        "gap": "8px",
                        "marginBottom": "8px",
                    },
                    children=[
                        html.Div([
                            html.Strong(f"Active Patient Dossier: Encounter #{int(eid)}", style={"fontSize": "14px", "color": "#1D3557"}),
                            html.Span(
                                f" (Patient #{int(row['patient_nbr'])}, Stay #{int(row['visit_index'])} of {int(row['n_visits_patient'])})",
                                style={"fontSize": "12px", "color": "#6C757D"},
                            ),
                        ]),
                        html.Div([
                            html.Span("True Next Pathway: ", style={"fontSize": "11.5px", "color": "#495057", "fontWeight": "600"}),
                            html.Span(
                                true_pathway,
                                style={
                                    "backgroundColor": data_layer.PATHWAY_COLORS.get(true_pathway, "#666"),
                                    "color": "#FFFFFF",
                                    "padding": "3px 9px",
                                    "borderRadius": "4px",
                                    "fontWeight": "700",
                                    "fontSize": "11.5px",
                                },
                            ),
                        ]),
                    ],
                ),
                # Serialized Clinical Note
                html.Div(
                    style={
                        "backgroundColor": "#FFFFFF",
                        "padding": "7px 10px",
                        "borderRadius": "5px",
                        "border": "1px solid #E9ECEF",
                        "fontSize": "11px",
                        "color": "#343A40",
                        "marginBottom": "8px",
                        "fontFamily": "Consolas, Monaco, monospace",
                        "lineHeight": "1.35",
                    },
                    children=[
                        html.Strong("Bio_ClinicalBERT Prompt: ", style={"color": "#1D3557"}),
                        html.Span(f"\"{row['model_text']}\""),
                    ],
                ),
                # 10 Retrieved Neighbors Stacked Bar Chart
                dcc.Graph(figure=neigh_fig, config={"displayModeBar": False, "displaylogo": False}),
                # Clinical Vitals & Predictions Grid
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(4, 1fr)",
                        "gap": "6px",
                        "marginTop": "8px",
                        "fontSize": "11px",
                        "backgroundColor": "#FFFFFF",
                        "padding": "8px 10px",
                        "borderRadius": "5px",
                        "border": "1px solid #E9ECEF",
                    },
                    children=[
                        html.Div([html.Strong("Model A Vote: "), html.Span(f"{ok_a} {pred_a.split(' ')[0]} ({row['model_a_agreement']:.0%})")]),
                        html.Div([html.Strong("Model B Vote: "), html.Span(f"{ok_b} {pred_b.split(' ')[0]} ({row['model_b_agreement']:.0%})")]),
                        html.Div([html.Strong("Sex / Age: "), html.Span(f"{row['gender']} / {row['age']}")]),
                        html.Div([html.Strong("Stay / Diagnoses: "), html.Span(f"{row['time_in_hospital']}d / {row['number_diagnoses']} diag")]),
                        html.Div([html.Strong("Max Glucose: "), html.Span(f"{row['max_glu_serum']}")]),
                        html.Div([html.Strong("HbA1c Result: "), html.Span(f"{row['A1Cresult']}")]),
                        html.Div([html.Strong("Insulin Regimen: "), html.Span(f"{row['insulin']}")]),
                        html.Div([html.Strong("Med Change: "), html.Span(f"{row['change']} (DiabMed: {row['diabetesMed']})")]),
                    ],
                ),
            ],
        )

    # -----------------------------------------------------------------------
    # Tab 2: Calibration, Disparity Battleground & Subgroup Fairness
    # -----------------------------------------------------------------------
    @app.callback(
        Output("graph-diag-calibration", "figure"),
        Input("store-selected-ids", "data"),
    )
    def render_calibration(selected_ids):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.calibration_curve(data, selected_set)

    @app.callback(
        Output("graph-diag-disparity", "figure"),
        Input("store-selected-ids", "data"),
        Input("diag-disparity-slice", "value"),
    )
    def render_disparity_battle(selected_ids, slice_by):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.disparity_battle(data, selected_set, slice_by or "age")

    @app.callback(
        Output("graph-diag-subgroup", "figure"),
        Input("store-selected-ids", "data"),
        Input("diag-subgroup-group", "value"),
    )
    def render_subgroup_fairness(selected_ids, group_by):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.subgroup_fairness(data, selected_set, group_by or "race")

    # -----------------------------------------------------------------------
    # Tab 3: Dynamic Cohort Profiler (Selection vs. Hospital Baseline)
    # -----------------------------------------------------------------------
    @app.callback(
        Output("graph-profile-vitals", "figure"),
        Input("store-selected-ids", "data"),
    )
    def render_cohort_profiler(selected_ids):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.cohort_profiler(data, selected_set)

    @app.callback(
        Output("graph-profile-pathways", "figure"),
        Input("store-selected-ids", "data"),
    )
    def render_cohort_pathways(selected_ids):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.cohort_pathway_breakdown(data, selected_set)

    # -----------------------------------------------------------------------
    # Tab 1: Curated Storyline (5-Min Video Walkthrough)
    # -----------------------------------------------------------------------
    @app.callback(
        Output("story-encounter-dropdown", "options"),
        Output("story-encounter-dropdown", "value"),
        Input("story-case-type", "value"),
        Input("store-selected-ids", "data"),
        State("story-encounter-dropdown", "value"),
    )
    def update_story_inspector_options(case_type, selected_ids, current_val):
        sub = df.loc[df.index.isin(selected_ids)] if selected_ids else df
        a = sub["model_a_correct"].astype(bool)
        b = sub["model_b_correct"].astype(bool)
        if case_type == "a_wins":
            sub = sub[a & ~b]
        elif case_type == "b_wins":
            sub = sub[b & ~a]
        elif case_type == "both_wrong":
            sub = sub[~a & ~b]

        if sub.empty:
            return [], None
        chosen = [int(e) for e in sub.index[:300]]
        if current_val is not None and current_val in sub.index and int(current_val) not in chosen:
            chosen.insert(0, int(current_val))
        options = [
            {"label": f"Encounter #{eid} · {df.loc[eid, 'gender']}, {df.loc[eid, 'age']} · Actual: {df.loc[eid, 'true_next_pathway']}", "value": eid}
            for eid in chosen
        ]
        new_val = int(current_val) if current_val in chosen else (chosen[0] if chosen else None)
        return options, new_val

    @app.callback(
        Output("story-retrieval-inspector", "children"),
        Input("story-encounter-dropdown", "value"),
    )
    def render_story_inspector(eid):
        if eid is None or eid not in df.index:
            return html.P("No encounter matches current filter. Try selecting 'Any Encounter'.", style={"fontSize": "12px", "color": "#6C757D"})
        row = df.loc[eid]
        ids_a = [int(v) for v in row["model_a_neighbors"]]
        ids_b = [int(v) for v in row["model_b_neighbors"]]
        overlap = set(ids_a) & set(ids_b)

        def _make_table(model, neighbors):
            distances = row[f"{model}_distances"]
            rows = []
            for rank, (nid, dist) in enumerate(zip(neighbors, distances), start=1):
                nb = df.loc[nid]
                sim = 1.0 - float(dist)
                path = str(nb["true_next_pathway"])
                is_shared = nid in overlap
                rows.append(
                    html.Tr(
                        [
                            html.Td(str(rank), style={"padding": "4px 8px", "color": "#6C757D"}),
                            html.Td(f"#{nid}", style={"fontFamily": "Consolas, monospace", "fontWeight": "600", "padding": "4px 8px"}),
                            html.Td(f"{sim:.3f}", style={"padding": "4px 8px", "fontWeight": "600"}),
                            html.Td(path, style={"padding": "4px 8px", "color": data_layer.PATHWAY_COLORS.get(path, "#333"), "fontWeight": "600"}),
                            html.Td(
                                html.Span(
                                    "Shared" if is_shared else "—",
                                    style={
                                        "backgroundColor": "#2A9D8F" if is_shared else "#F8F9FA",
                                        "color": "#FFFFFF" if is_shared else "#ADB5BD",
                                        "padding": "2px 6px",
                                        "borderRadius": "3px",
                                        "fontSize": "10.5px",
                                        "fontWeight": "600",
                                    },
                                ),
                                style={"padding": "4px 8px", "textAlign": "center"},
                            ),
                        ],
                        style={"backgroundColor": "#F0F9F5" if is_shared else "#FFFFFF", "borderBottom": "1px solid #E9ECEF"},
                    )
                )
            return html.Div(
                style={"flex": "1", "minWidth": "0"},
                children=[
                    html.H4(
                        "Model A · Tabular KNN (127-D Count)" if model == "model_a" else "Model B · Bio_ClinicalBERT (768-D Text)",
                        style={"margin": "0 0 4px", "fontSize": "13px", "color": "#1D3557" if model == "model_a" else "#E63946"},
                    ),
                    html.P(
                        f"Predicted: {row[f'{model}_pred_pathway']} · Vote share: {row[f'{model}_vote_share']:.0%}",
                        style={"fontSize": "12px", "margin": "0 0 6px", "color": "#495057"},
                    ),
                    html.Div(
                        style={"overflowX": "auto"},
                        children=[
                            html.Table(
                                style={"borderCollapse": "collapse", "width": "100%", "fontSize": "11.5px"},
                                children=[
                                    html.Thead(
                                        html.Tr([
                                            html.Th(s, style={"textAlign": "left", "padding": "5px 8px", "backgroundColor": "#F8F9FA", "borderBottom": "1px solid #CED4DA"})
                                            for s in ["Rank", "Encounter", "Cosine Sim", "Neighbor Pathway", "Overlap"]
                                        ])
                                    ),
                                    html.Tbody(rows),
                                ],
                            )
                        ],
                    ),
                ],
            )

        return html.Div(
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "padding": "8px 12px", "backgroundColor": "#F8F9FA", "borderRadius": "6px", "marginBottom": "10px", "fontSize": "12px", "flexWrap": "wrap", "gap": "8px"},
                    children=[
                        html.Div([
                            html.Strong(f"Encounter #{eid}"),
                            html.Span(f" · Patient #{int(row['patient_nbr'])} · {row['gender']}, {row['age']} · Stay: {row['time_in_hospital']}d · Diagnoses: {row['number_diagnoses']}"),
                        ]),
                        html.Div([
                            html.Span("Actual Next Pathway: "),
                            html.Strong(str(row["true_next_pathway"]), style={"color": data_layer.PATHWAY_COLORS.get(str(row["true_next_pathway"]), "#333")}),
                            html.Span(f" | Shared Neighbors: {len(overlap)} / 10", style={"marginLeft": "10px", "fontWeight": "700", "color": "#2A9D8F" if len(overlap) > 0 else "#E63946"}),
                        ]),
                    ],
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "16px"},
                    children=[_make_table("model_a", ids_a), _make_table("model_b", ids_b)],
                ),
                html.Details(
                    style={"marginTop": "10px", "fontSize": "11.5px"},
                    children=[
                        html.Summary("Show Bio_ClinicalBERT Serialized Input Text", style={"cursor": "pointer", "color": "#457B9D"}),
                        html.Pre(
                            str(row["model_text"]),
                            style={"fontFamily": "Consolas, monospace", "backgroundColor": "#F8F9FA", "padding": "8px", "borderRadius": "4px", "border": "1px solid #E9ECEF", "whiteSpace": "pre-wrap", "marginTop": "4px"},
                        ),
                    ],
                ),
            ]
        )

    @app.callback(
        Output("graph-story-calibration", "figure"),
        Input("store-selected-ids", "data"),
    )
    def render_story_calibration(selected_ids):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.calibration_curve(data, selected_set)

    @app.callback(
        Output("graph-story-flips", "figure"),
        Input("store-selected-ids", "data"),
    )
    def render_story_flips(selected_ids):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.flips_view(data, selected_set)

    @app.callback(
        Output("graph-story-umap", "figure"),
        Input("store-selected-ids", "data"),
    )
    def render_story_umap(selected_ids):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.umap_view(data, selected_set, color_mode="cluster", show_traj=True, n_patients=35, annotate=True)

    @app.callback(
        Output("graph-story-model-a", "figure"),
        Output("graph-story-model-b", "figure"),
        Input("store-active-cell", "data"),
    )
    def render_story_confusion(active_cell):
        hl_a = (active_cell["true"], active_cell["pred"]) if active_cell and active_cell.get("model") == "model_a" and "true" in active_cell else None
        hl_b = (active_cell["true"], active_cell["pred"]) if active_cell and active_cell.get("model") == "model_b" and "true" in active_cell else None
        return (
            figures.confusion_heatmap(data, "model_a", True, False, hl_a),
            figures.confusion_heatmap(data, "model_b", True, False, hl_b),
        )

    @app.callback(
        Output("graph-story-disparity", "figure"),
        Input("store-selected-ids", "data"),
        Input("story-disparity-slice", "value"),
    )
    def render_story_disparity(selected_ids, slice_by):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.disparity_battle(data, selected_set, slice_by or "age")

    @app.callback(
        Output("graph-story-subgroup", "figure"),
        Input("store-selected-ids", "data"),
        Input("story-subgroup-group", "value"),
    )
    def render_story_subgroup(selected_ids, group_by):
        selected_set = set(selected_ids) if selected_ids else None
        return figures.subgroup_fairness(data, selected_set, group_by or "race")
