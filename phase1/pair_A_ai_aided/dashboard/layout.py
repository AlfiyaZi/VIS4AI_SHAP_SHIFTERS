"""
Layout components for the Pair A (AI-aided) Phase I dashboard.
Organized in a strict 1 -> 2 -> 3 -> 4 visual hierarchy with dedicated control bars and zero duplicate IDs.
"""

from __future__ import annotations

from dash import dcc, html
from data_layer import DashboardData
import figures

CARD_STYLE = {
    "backgroundColor": "#FFFFFF",
    "borderRadius": "10px",
    "border": "1px solid #DEE2E6",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.04)",
    "padding": "16px 18px",
    "display": "flex",
    "flexDirection": "column",
}

CONTROL_BAR_STYLE = {
    "backgroundColor": "#F8F9FA",
    "border": "1px solid #E9ECEF",
    "borderRadius": "6px",
    "padding": "6px 12px",
    "marginTop": "8px",
    "marginBottom": "8px",
    "fontSize": "12px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
    "flexWrap": "wrap",
    "gap": "10px",
}

GRAPH_CONFIG_INTERACTIVE = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
}

GRAPH_CONFIG_CLEAN = {
    "displayModeBar": False,
    "displaylogo": False,
}


def create_header_banner(df, metrics: dict) -> html.Div:
    total_encounters = len(df)
    unique_patients = df["patient_nbr"].nunique()
    acc_a = (df["model_a_pred_pathway"] == df["true_next_pathway"]).mean()
    acc_b = (df["model_b_pred_pathway"] == df["true_next_pathway"]).mean()
    acc_maj = (df["majority_pred_pathway"] == df["true_next_pathway"]).mean()

    return html.Div(
        style={
            **CARD_STYLE,
            "borderTop": "4px solid #1D3557",
            "padding": "14px 20px",
            "marginBottom": "12px",
        },
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "gap": "14px",
                },
                children=[
                    html.Div([
                        html.H2(
                            "🏥 Phase I: Clinical Patient Retrieval & Behavioral Audit Studio",
                            style={"margin": "0 0 3px 0", "fontSize": "19px", "fontWeight": "700", "color": "#1D3557"},
                        ),
                        html.P(
                            "Team SHAP_SHIFTERS | Pair A (AI-Aided Visual Analytics) — Coordinated Views 1 → 2 → 3 → 4",
                            style={"margin": 0, "fontSize": "12.5px", "color": "#6C757D"},
                        ),
                    ]),
                    html.Div(
                        style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                        children=[
                            _metric_badge("Cohort Size", f"{total_encounters:,} visits", f"{unique_patients:,} patients", "#212529"),
                            _metric_badge("Model A (Tabular)", f"{acc_a:.1%}", "127-D KNN", "#457B9D"),
                            _metric_badge("Model B (BERT)", f"{acc_b:.1%}", "768-D KNN", "#E63946"),
                            _metric_badge("Majority Baseline", f"{acc_maj:.1%}", "Always 'Other'", "#495057"),
                            _metric_badge("Kidney / Eye Recall", "0.0%", "Class Imbalance Blindspot", "#8E6CBF"),
                        ],
                    ),
                ],
            ),
        ],
    )


def _metric_badge(title: str, val: str, sub: str, color: str) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "#F8F9FA",
            "border": "1px solid #E9ECEF",
            "borderRadius": "8px",
            "padding": "5px 12px",
            "minWidth": "108px",
            "textAlign": "center",
        },
        children=[
            html.Span(title, style={"fontSize": "10.5px", "color": "#6C757D", "display": "block", "fontWeight": "600"}),
            html.Strong(val, style={"fontSize": "14.5px", "color": color, "display": "block", "lineHeight": "1.2"}),
            html.Span(sub, style={"fontSize": "10px", "color": "#868E96"}),
        ],
    )


def create_global_toolbar(df) -> html.Div:
    sample_eids = df.index[:300]
    return html.Div(
        style={
            **CARD_STYLE,
            "padding": "10px 18px",
            "marginBottom": "14px",
            "flexDirection": "row",
            "justifyContent": "space-between",
            "alignItems": "center",
            "flexWrap": "wrap",
            "gap": "12px",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "14px", "flexWrap": "wrap"},
                children=[
                    html.Button(
                        "✕ Reset / Clear Selection",
                        id="btn-reset",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#1D3557",
                            "color": "#FFFFFF",
                            "border": "none",
                            "padding": "7px 15px",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "fontWeight": "600",
                            "fontSize": "12.5px",
                        },
                    ),
                    html.Div(id="selection-status-badge", style={"fontSize": "13px", "fontWeight": "500"}),
                ],
            ),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "10px", "fontSize": "12.5px"},
                children=[
                    html.Span("🔍 Inspect Patient Stay:", style={"fontWeight": "600", "color": "#1D3557"}),
                    dcc.Dropdown(
                        id="encounter-inspect-dropdown",
                        options=[
                            {
                                "label": f"Encounter #{int(eid)} — {df.loc[eid, 'true_next_pathway'].split(' ')[0]} ({df.loc[eid, 'gender']}, {df.loc[eid, 'age']})",
                                "value": int(eid),
                            }
                            for eid in sample_eids
                        ],
                        value=int(sample_eids[0]),
                        clearable=False,
                        style={"width": "300px"},
                    ),
                ],
            ),
        ],
    )


def create_view1_container() -> html.Div:
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div([
                html.H3(
                    "View 1: Neighbor Agreement vs. Comorbidity Explorer",
                    style={"margin": "0 0 2px 0", "fontSize": "15px", "fontWeight": "700", "color": "#1D3557"},
                ),
                html.Span(
                    "Faceted by True Next Complication Pathway. Drag a Lasso or Box to cross-filter all views; click any point to inspect.",
                    style={"fontSize": "12px", "color": "#6C757D"},
                ),
            ]),
            html.Div(
                style=CONTROL_BAR_STYLE,
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                        children=[
                            html.Strong("X-Axis:", style={"color": "#1D3557"}),
                            dcc.RadioItems(
                                id="v1-model-x",
                                options=[
                                    {"label": " Model B Agreement (BERT)", "value": "model_b"},
                                    {"label": " Model A Agreement (Tabular)", "value": "model_a"},
                                ],
                                value="model_b",
                                inline=True,
                                inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                        children=[
                            html.Strong("Display:", style={"color": "#1D3557"}),
                            dcc.Checklist(
                                id="v1-options",
                                options=[
                                    {"label": " Jitter Points", "value": "jitter"},
                                    {"label": " Density Contours", "value": "contours"},
                                ],
                                value=["jitter"],
                                inline=True,
                                inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Graph(id="graph-view1-scatter", config=GRAPH_CONFIG_INTERACTIVE),
        ],
    )


def create_view2_container() -> html.Div:
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div([
                html.H3(
                    "View 2: Reordered Error & Confusion Breakdown Matrix",
                    style={"margin": "0 0 2px 0", "fontSize": "15px", "fontWeight": "700", "color": "#1D3557"},
                ),
                html.Span(
                    "Compares Model A, Model B, and Majority Baseline. Click any cell (True × Predicted) to filter the entire cohort.",
                    style={"fontSize": "12px", "color": "#6C757D"},
                ),
            ]),
            html.Div(
                style=CONTROL_BAR_STYLE,
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                        children=[
                            html.Strong("Matrix Options:", style={"color": "#1D3557"}),
                            dcc.Checklist(
                                id="v2-options",
                                options=[
                                    {"label": " Clustered Class Order (Optimal Leaf)", "value": "clustered"},
                                    {"label": " Row-Normalized (%)", "value": "normalize"},
                                ],
                                value=["clustered", "normalize"],
                                inline=True,
                                inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                            ),
                        ],
                    ),
                    html.Span("Tip: Dotted cells = diagonal (correct predictions)", style={"fontSize": "11px", "color": "#6C757D"}),
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1.08fr 1fr 1fr", "gap": "8px"},
                children=[
                    dcc.Graph(id="graph-view2-model-a", config=GRAPH_CONFIG_CLEAN),
                    dcc.Graph(id="graph-view2-model-b", config=GRAPH_CONFIG_CLEAN),
                    dcc.Graph(id="graph-view2-majority", config=GRAPH_CONFIG_CLEAN),
                ],
            ),
        ],
    )


def create_view3_container(data: DashboardData) -> html.Div:
    top_drivers = data.driver_ranking.head(5)
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div([
                html.H3(
                    "View 3: Input-Space UMAP & Longitudinal Patient Trajectories",
                    style={"margin": "0 0 2px 0", "fontSize": "15px", "fontWeight": "700", "color": "#1D3557"},
                ),
                html.Span(
                    "2D UMAP of the 30,248 patient stays. Use Lasso or Box selection to isolate regions; click a point to inspect.",
                    style={"fontSize": "12px", "color": "#6C757D"},
                ),
            ]),
            html.Div(
                style=CONTROL_BAR_STYLE,
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                        children=[
                            html.Strong("Color By:", style={"color": "#1D3557"}),
                            dcc.Dropdown(
                                id="v3-color-mode",
                                options=[{"label": v, "value": k} for k, v in figures.UMAP_COLOR_MODES.items()],
                                value="driver",
                                clearable=False,
                                style={"width": "220px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                        children=[
                            html.Strong("Overlays:", style={"color": "#1D3557"}),
                            dcc.Checklist(
                                id="v3-traj-toggle",
                                options=[
                                    {"label": " Trajectories (V1→V2→V3)", "value": "traj"},
                                    {"label": " Cluster Labels", "value": "annot"},
                                ],
                                value=["annot"],
                                inline=True,
                                inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Graph(id="graph-view3-umap", config=GRAPH_CONFIG_INTERACTIVE),
            # Compact summary strip of top statistical feature drivers separating the UMAP islands
            html.Div(
                style={
                    "marginTop": "8px",
                    "padding": "8px 12px",
                    "backgroundColor": "#F8F9FA",
                    "borderRadius": "6px",
                    "border": "1px solid #E9ECEF",
                    "fontSize": "11.5px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "flexWrap": "wrap",
                    "gap": "8px",
                },
                children=[
                    html.Span(
                        "📊 Top Features Separating UMAP Islands (Cramér's V / η):",
                        style={"fontWeight": "700", "color": "#1D3557"},
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                        children=[
                            html.Span(
                                f"{i + 1}. {row['feature']} ({row['association']:.2f})",
                                style={
                                    "backgroundColor": "#EBF2FA" if i < 2 else "#FFFFFF",
                                    "border": "1px solid #DEE2E6",
                                    "borderRadius": "4px",
                                    "padding": "2px 8px",
                                    "fontWeight": "600" if i < 2 else "400",
                                },
                            )
                            for i, (_, row) in enumerate(top_drivers.iterrows())
                        ],
                    ),
                ],
            ),
        ],
    )


def create_view4_container() -> html.Div:
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div([
                html.H3(
                    "View 4: Counterfactual Neighbor Flips & Patient Inspector",
                    style={"margin": "0 0 2px 0", "fontSize": "15px", "fontWeight": "700", "color": "#1D3557"},
                ),
                html.Span(
                    "Top: Sensitivity of k=10 retrieved neighbors under synthetic clinical edits. Bottom: Active encounter dossier.",
                    style={"fontSize": "12px", "color": "#6C757D"},
                ),
            ]),
            html.Div(
                style={"marginTop": "6px"},
                children=[dcc.Graph(id="graph-view4-flips", config=GRAPH_CONFIG_CLEAN)],
            ),
            html.Hr(style={"margin": "10px 0", "border": "none", "borderTop": "1px solid #E9ECEF"}),
            html.Div(id="view4-inspector-card"),
        ],
    )


def create_tab_disparity() -> html.Div:
    return html.Div(
        style={"marginTop": "10px"},
        children=[
            # Informative Phase I Synthesis Banner
            html.Div(
                style={
                    "padding": "10px 16px",
                    "backgroundColor": "#FFFFFF",
                    "borderRadius": "8px",
                    "border": "1px solid #E9ECEF",
                    "marginBottom": "12px",
                    "fontSize": "12.5px",
                    "color": "#333",
                    "lineHeight": "1.4",
                },
                children=[
                    html.Strong("🔍 Phase I Verified Reliability Findings: ", style={"color": "#1D3557"}),
                    html.Span("1. Vote-Share Calibration: KNN only beats the 66.4% majority-class baseline when >=8/10 neighbors agree on the predicted class (~31% of stays, reaching 68–78% accuracy). "),
                    html.Span("2. Age Disagreement Split: Across the 5,918 A-vs-B conflicts, Model A wins 58.5% vs 34.9% on patients <50 yrs (n=482), whereas Model B edges out 51.2% vs 46.0% on patients 80+ (n=1,321). "),
                    html.Span("3. Base-Rate Confounding: Lower raw accuracy on African American stays (60.0% vs 63.7% Caucasian) is explained by a lower majority-class base rate (63.1% vs 66.9%); relative lift over each group's majority baseline is identical (-3.1 pp vs -3.3 pp)."),
                ],
            ),
            # Row 1: Calibration & Danger Zone (Left 52%) + Disagreement Battleground (Right 48%)
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1.1fr 1fr", "gap": "14px", "marginBottom": "14px"},
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div([
                                html.H3(
                                    "Test-Time Calibration: Predicted Vote Share vs. Empirical Accuracy",
                                    style={"margin": "0 0 2px 0", "fontSize": "15px", "fontWeight": "700", "color": "#1D3557"},
                                ),
                                html.Span(
                                    "Bins predictions by label-free test-time vote share (40% to 100%) with 95% Wilson CIs. Only >=80% consensus beats the 66.4% majority baseline.",
                                    style={"fontSize": "12px", "color": "#6C757D"},
                                ),
                            ]),
                            dcc.Graph(id="graph-diag-calibration", config=GRAPH_CONFIG_CLEAN),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "4px", "flexWrap": "wrap", "gap": "8px"},
                                children=[
                                    html.Div([
                                        html.H3(
                                            "Model A vs. Model B Disagreement Battleground",
                                            style={"margin": "0 0 2px 0", "fontSize": "15px", "fontWeight": "700", "color": "#1D3557"},
                                        ),
                                        html.Span("100%-normalized win shares across the 5,918 conflicts, with per-bracket sample size n (⚠ flags n < 50).", style={"fontSize": "12px", "color": "#6C757D"}),
                                    ]),
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "6px", "fontSize": "12px"},
                                        children=[
                                            html.Strong("Slice By:", style={"color": "#1D3557"}),
                                            dcc.Dropdown(
                                                id="diag-disparity-slice",
                                                options=[
                                                    {"label": "Age Group", "value": "age"},
                                                    {"label": "True Complication Pathway", "value": "true_next_pathway"},
                                                    {"label": "Diagnoses Count (Comorbidity)", "value": "number_diagnoses"},
                                                    {"label": "Length of Hospital Stay", "value": "time_in_hospital"},
                                                    {"label": "Medication Change", "value": "change"},
                                                ],
                                                value="age",
                                                clearable=False,
                                                style={"width": "220px"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            dcc.Graph(id="graph-diag-disparity", config=GRAPH_CONFIG_CLEAN),
                        ],
                    ),
                ],
            ),
            # Row 2: Demographic & Clinical Subgroup Fairness Audit
            html.Div(
                style=CARD_STYLE,
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px", "flexWrap": "wrap", "gap": "8px"},
                        children=[
                            html.Div([
                                html.H3(
                                    "Subgroup Accuracy Lift vs. Subgroup Majority Baseline (with 95% Wilson CIs)",
                                    style={"margin": "0 0 2px 0", "fontSize": "15px", "fontWeight": "700", "color": "#1D3557"},
                                ),
                                html.Span("Controls for subgroup base-rate differences (e.g. % Uncomplicated / Other per group) to separate true model bias from base-rate confounding.", style={"fontSize": "12px", "color": "#6C757D"}),
                            ]),
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "8px", "fontSize": "12px"},
                                children=[
                                    html.Strong("Subgroup:", style={"color": "#1D3557"}),
                                    dcc.RadioItems(
                                        id="diag-subgroup-group",
                                        options=[
                                            {"label": " Race / Ethnicity", "value": "race"},
                                            {"label": " Admission Pathway", "value": "admission_type"},
                                            {"label": " Age Brackets", "value": "age"},
                                        ],
                                        value="race",
                                        inline=True,
                                        inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Graph(id="graph-diag-subgroup", config=GRAPH_CONFIG_CLEAN),
                ],
            ),
        ],
    )


def create_tab_profiler() -> html.Div:
    return html.Div(
        style={"marginTop": "10px"},
        children=[
            # Instructions Banner
            html.Div(
                style={
                    "padding": "10px 16px",
                    "backgroundColor": "#FFFFFF",
                    "borderRadius": "8px",
                    "border": "1px solid #E9ECEF",
                    "marginBottom": "12px",
                    "fontSize": "12.5px",
                    "color": "#333",
                    "lineHeight": "1.4",
                },
                children=[
                    html.Strong("🧬 How to Use the Dynamic Cohort Profiler: ", style={"color": "#1D3557"}),
                    html.Span("Switch to Tab 1 (Studio) and drag a Lasso or Box on View 1 (Scatter) or View 3 (UMAP), or click a cell in View 2. Then return to this tab to see how your selected patient cohort compares to the 30,248 hospital baseline!"),
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1.2fr 1fr", "gap": "14px"},
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div([
                                html.H3(
                                    "Clinical Complexity: Active Selection vs. Hospital Baseline",
                                    style={"margin": "0 0 2px 0", "fontSize": "15px", "fontWeight": "700", "color": "#1D3557"},
                                ),
                                html.Span(
                                    "Shows how your lassoed/clicked patient cohort differs from the overall hospital population.",
                                    style={"fontSize": "12px", "color": "#6C757D"},
                                ),
                            ]),
                            dcc.Graph(id="graph-profile-vitals", config=GRAPH_CONFIG_CLEAN),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div([
                                html.H3(
                                    "Complication Outcomes Distribution",
                                    style={"margin": "0 0 2px 0", "fontSize": "15px", "fontWeight": "700", "color": "#1D3557"},
                                ),
                                html.Span(
                                    "Compares the true complication trajectory breakdown of the active selection vs baseline.",
                                    style={"fontSize": "12px", "color": "#6C757D"},
                                ),
                            ]),
                            dcc.Graph(id="graph-profile-pathways", config=GRAPH_CONFIG_CLEAN),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_tab_story(data: DashboardData) -> html.Div:
    df = data.df
    top_drivers = data.driver_ranking.head(8)

    # Ranking table rows
    ranking_rows = []
    for _, r in top_drivers.iterrows():
        val = float(r["association"])
        ranking_rows.append(
            html.Tr(
                [
                    html.Td(r["feature"], style={"fontWeight": "600", "padding": "4px 6px", "fontFamily": "Consolas, monospace", "fontSize": "11px"}),
                    html.Td(r["kind"].split(" ")[0], style={"padding": "4px 6px", "fontSize": "10.5px", "color": "#6C757D"}),
                    html.Td(f"{val:.3f}", style={"padding": "4px 6px", "fontWeight": "700", "color": "#1D3557"}),
                    html.Td(
                        html.Div(
                            style={"backgroundColor": "#E9ECEF", "borderRadius": "3px", "width": "60px", "height": "8px", "overflow": "hidden"},
                            children=[html.Div(style={"backgroundColor": "#457B9D", "width": f"{min(100, int(val * 100))}%", "height": "100%"})],
                        ),
                        style={"padding": "4px 6px"},
                    ),
                ],
                style={"borderBottom": "1px solid #F1F3F5"},
            )
        )

    # Initial dropdown encounters
    a = df["model_a_correct"].astype(bool)
    b = df["model_b_correct"].astype(bool)
    a_wins_idx = df.index[a & ~b][:300]
    init_options = [
        {"label": f"Encounter #{eid} · {df.loc[eid, 'gender']}, {df.loc[eid, 'age']} · Actual: {df.loc[eid, 'true_next_pathway']}", "value": int(eid)}
        for eid in a_wins_idx
    ]
    init_val = int(a_wins_idx[0]) if len(a_wins_idx) > 0 else int(df.index[0])

    return html.Div(
        style={"marginTop": "10px"},
        children=[
            # Quick Jump Navigation Bar
            html.Div(
                style={
                    **CONTROL_BAR_STYLE,
                    "backgroundColor": "#FFFFFF",
                    "padding": "8px 16px",
                    "marginBottom": "14px",
                    "borderLeft": "4px solid #1D3557",
                },
                children=[
                    html.Span("🎬 5-Minute Video Walkthrough (6 Chapters):", style={"fontWeight": "700", "color": "#1D3557", "fontSize": "12.5px"}),
                    html.Div(
                        style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
                        children=[
                            html.A("1 · Error Blocks & 0% Recall", href="#chapter-performance", style={"color": "#457B9D", "textDecoration": "none", "fontWeight": "600", "fontSize": "12px"}),
                            html.A("2 · Calibration Cliff (≥8/10)", href="#chapter-calibration", style={"color": "#457B9D", "textDecoration": "none", "fontWeight": "600", "fontSize": "12px"}),
                            html.A("3 · High-D Manifold Drivers", href="#chapter-drivers", style={"color": "#457B9D", "textDecoration": "none", "fontWeight": "600", "fontSize": "12px"}),
                            html.A("4 · Counterfactual Flips", href="#chapter-shortcuts", style={"color": "#457B9D", "textDecoration": "none", "fontWeight": "600", "fontSize": "12px"}),
                            html.A("5 · Conflicts & Deconfounding", href="#chapter-disagreements", style={"color": "#457B9D", "textDecoration": "none", "fontWeight": "600", "fontSize": "12px"}),
                            html.A("6 · 10-NN Dossier (22.4% Overlap)", href="#chapter-inspector", style={"color": "#457B9D", "textDecoration": "none", "fontWeight": "600", "fontSize": "12px"}),
                        ],
                    ),
                ],
            ),

            # Chapter 1: Top-Level Summary — Reordered Confusion Matrices & 0.0% Minority Recall
            html.Div(
                id="chapter-performance",
                style={**CARD_STYLE, "marginBottom": "14px"},
                children=[
                    html.Div([
                        html.H3("Chapter 1 · Top-Level Summary: The Majority Baseline Trap & 0.0% Minority Recall", style={"margin": "0 0 2px", "fontSize": "16px", "fontWeight": "700", "color": "#1D3557"}),
                        html.P("Hierarchically reordered confusion matrices (Alsallakh et al. 2017 optimal leaf ordering) — click any cell to filter the entire dashboard.", style={"margin": "0 0 8px", "fontSize": "12px", "color": "#6C757D"}),
                    ]),
                    html.Div(
                        style={"backgroundColor": "#F3EBF8", "borderLeft": "4px solid #8E6CBF", "padding": "8px 12px", "borderRadius": "4px", "marginBottom": "10px", "fontSize": "12px", "lineHeight": "1.4", "color": "#3C1E66"},
                        children=[
                            html.Strong("💡 Why We Start Here: "),
                            html.Span("Both k=10 cosine KNN models achieve ~63.0% accuracy (Model A: 63.01%, Model B: 62.96%) — losing to the 66.36% dummy Majority Baseline ('Always Uncomplicated / Other'). Reordering by error similarity reveals a 2-state prediction collapse: both models achieve 0.0% recall on Kidney / Renal (2.5% of cohort, n=757) and Eye / Neurological (1.9%, n=563) because rare pathways almost never win a 10-neighbor majority vote."),
                        ],
                    ),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "14px"},
                        children=[
                            dcc.Graph(id="graph-story-model-a", figure=figures.confusion_heatmap(data, "model_a", True, False, None), config=GRAPH_CONFIG_CLEAN),
                            dcc.Graph(id="graph-story-model-b", figure=figures.confusion_heatmap(data, "model_b", True, False, None), config=GRAPH_CONFIG_CLEAN),
                        ],
                    ),
                ],
            ),

            # Chapter 2: The Calibration Cliff
            html.Div(
                id="chapter-calibration",
                style={**CARD_STYLE, "marginBottom": "14px"},
                children=[
                    html.Div([
                        html.H3("Chapter 2 · Confidence Drill-Down: The Calibration Cliff (≥ 8/10 Neighbor Consensus)", style={"margin": "0 0 2px", "fontSize": "16px", "fontWeight": "700", "color": "#1D3557"}),
                        html.P("Label-free test-time vote share (fraction of 10 neighbors voting for the predicted class) vs. empirical accuracy with 95% Wilson CIs.", style={"margin": "0 0 8px", "fontSize": "12px", "color": "#6C757D"}),
                    ]),
                    html.Div(
                        style={"backgroundColor": "#EBF3F8", "borderLeft": "4px solid #457B9D", "padding": "8px 12px", "borderRadius": "4px", "marginBottom": "10px", "fontSize": "12px", "lineHeight": "1.4", "color": "#1D3557"},
                        children=[
                            html.Strong("💡 Key Insight: "),
                            html.Span("In ~69.4% of hospital stays (vote share < 80%, n=21,008), neighbor consensus is weak and accuracy drops to 59.6%–59.7% — performing worse than both the 66.4% Majority Baseline and the 62.1% Random-Neighbor Baseline! Only when ≥ 8 of 10 neighbors agree (~30.6% of stays, n=9,240) does genuine clinical signal emerge, reaching 68.2%–77.6% accuracy."),
                        ],
                    ),
                    dcc.Graph(id="graph-story-calibration", figure=figures.calibration_curve(data, selected=None), config=GRAPH_CONFIG_CLEAN),
                ],
            ),

            # Chapter 3: High-Dimensional Manifold Drivers
            html.Div(
                id="chapter-drivers",
                style={**CARD_STYLE, "marginBottom": "14px"},
                children=[
                    html.Div([
                        html.H3("Chapter 3 · High-Dimensional Manifold: What Drives Patient Organization?", style={"margin": "0 0 2px", "fontSize": "16px", "fontWeight": "700", "color": "#1D3557"}),
                        html.P("2D UMAP projection strictly on 127-D clinical input features (black box closed) paired with statistical Cramér's V driver rankings.", style={"margin": "0 0 8px", "fontSize": "12px", "color": "#6C757D"}),
                    ]),
                    html.Div(
                        style={"backgroundColor": "#EAF6EF", "borderLeft": "4px solid #2A9D8F", "padding": "8px 12px", "borderRadius": "4px", "marginBottom": "10px", "fontSize": "12px", "lineHeight": "1.4", "color": "#1B4332"},
                        children=[
                            html.Strong("💡 Key Insight: "),
                            html.Span("Why don't neighbors group by future complication pathway? Objective association metrics across all 15 inputs prove the global UMAP clusters are separated by current treatment workflow flags: Medication Change (change, Cramér's V = 0.803), Diabetes Medication (diabetesMed, V = 0.796), Max Glucose (V = 0.518), Admission Source (V = 0.488), and Insulin (V = 0.486). Repeat-patient trajectories (V1 → V2 → V3) jump across clusters whenever therapy is adjusted."),
                        ],
                    ),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1.3fr 1fr", "gap": "14px", "alignItems": "center"},
                        children=[
                            dcc.Graph(id="graph-story-umap", figure=figures.umap_view(data, selected=None, color_mode="cluster", show_traj=True, n_patients=35, annotate=True), config=GRAPH_CONFIG_INTERACTIVE),
                            html.Div(
                                style={"backgroundColor": "#F8F9FA", "borderRadius": "8px", "border": "1px solid #E9ECEF", "padding": "12px"},
                                children=[
                                    html.H4("Top High-Dimensional Manifold Drivers", style={"margin": "0 0 6px", "fontSize": "13px", "color": "#1D3557"}),
                                    html.P("Cramér's V (categorical) & Correlation Ratio (numeric) against HDBSCAN clusters:", style={"fontSize": "11px", "color": "#6C757D", "margin": "0 0 8px"}),
                                    html.Table(
                                        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "11.5px"},
                                        children=[
                                            html.Thead(html.Tr([
                                                html.Th("Feature", style={"textAlign": "left", "padding": "4px 6px"}),
                                                html.Th("Type", style={"textAlign": "left", "padding": "4px 6px"}),
                                                html.Th("Assoc", style={"textAlign": "left", "padding": "4px 6px"}),
                                                html.Th("Strength", style={"textAlign": "left", "padding": "4px 6px"}),
                                            ])),
                                            html.Tbody(ranking_rows),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # Chapter 4: Perturbation Sensitivity & Shortcuts
            html.Div(
                id="chapter-shortcuts",
                style={**CARD_STYLE, "marginBottom": "14px"},
                children=[
                    html.Div([
                        html.H3("Chapter 4 · Perturbation Sensitivity: Demographic Shortcuts vs. Token Pooling Artifacts", style={"margin": "0 0 2px", "fontSize": "16px", "fontWeight": "700", "color": "#1D3557"}),
                        html.P("Mean replaced neighbors (of k=10) after single-attribute counterfactual edits across all encounters (±95% CIs).", style={"margin": "0 0 8px", "fontSize": "12px", "color": "#6C757D"}),
                    ]),
                    html.Div(
                        style={"backgroundColor": "#FFF3CD", "borderLeft": "4px solid #E76F51", "padding": "8px 12px", "borderRadius": "4px", "marginBottom": "10px", "fontSize": "12px", "lineHeight": "1.4", "color": "#856404"},
                        children=[
                            html.Strong("💡 Key Insight (Mirror Inversion): "),
                            html.Span("Each encoder suffers from an opposite representation artifact: (1) On Sex Swap (Male ↔ Female), Model A replaces 8.86 / 10 neighbors (94.7% same-sex homophily) because unscaled binary one-hot columns add Δ||x||² = 2.0 before cosine normalization, whereas Bio_ClinicalBERT replaces only 4.15 / 10. (2) Conversely, on HbA1c Elevation (None/Norm → '>8'), Bio_ClinicalBERT replaces 8.22 / 10 neighbors (vs. 4.36 / 10 in Model A) because changing 'none' → '>', '8' alters multiple WordPiece tokens in mean-pooled 768-D space."),
                        ],
                    ),
                    dcc.Graph(id="graph-story-flips", figure=figures.flips_view(data, selected=None), config=GRAPH_CONFIG_CLEAN),
                ],
            ),

            # Chapter 5: Disagreements & Subgroup Base-Rate Deconfounding
            html.Div(
                id="chapter-disagreements",
                style={**CARD_STYLE, "marginBottom": "14px"},
                children=[
                    html.Div([
                        html.H3("Chapter 5 · Model Disagreements (n=5,918) & Subgroup Base-Rate Deconfounding", style={"margin": "0 0 2px", "fontSize": "16px", "fontWeight": "700", "color": "#1D3557"}),
                        html.P("Left: Head-to-head win rates on the 19.6% of stays where A and B conflict. Right: Subgroup accuracy lift controlling for majority base-rate confounding.", style={"margin": "0 0 8px", "fontSize": "12px", "color": "#6C757D"}),
                    ]),
                    html.Div(
                        style={"backgroundColor": "#EBF3F8", "borderLeft": "4px solid #1D3557", "padding": "8px 12px", "borderRadius": "4px", "marginBottom": "10px", "fontSize": "12px", "lineHeight": "1.4", "color": "#1D3557"},
                        children=[
                            html.Strong("💡 Key Insight: "),
                            html.Span("Across 5,918 conflicts, standardized numeric features in Model A win 58.5% vs. 34.9% on younger diabetics (<50 yrs, n=482), whereas Bio_ClinicalBERT edges out Model A 51.2% vs. 46.0% on elderly multi-morbid patients (80+ yrs, n=1,321). Meanwhile, controlling for each racial group's majority-class base rate proves the raw African American vs. Caucasian gap (60.0% vs. 63.7%) is a base-rate artifact (63.1% vs. 66.9% 'Other' base rate), yielding identical relative lift (-3.1 pp vs. -3.3 pp)."),
                        ],
                    ),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "14px", "alignItems": "start"},
                        children=[
                            html.Div([
                                html.Div(
                                    style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"},
                                    children=[
                                        html.Span("Slice Disagreements By:", style={"fontSize": "11.5px", "fontWeight": "600", "color": "#495057"}),
                                        dcc.Dropdown(
                                            id="story-disparity-slice",
                                            options=[
                                                {"label": "Age Group", "value": "age"},
                                                {"label": "True Next Pathway", "value": "true_next_pathway"},
                                                {"label": "Diagnoses Count", "value": "number_diagnoses"},
                                                {"label": "Length of Stay", "value": "time_in_hospital"},
                                                {"label": "Medication Change", "value": "change"},
                                            ],
                                            value="age",
                                            clearable=False,
                                            style={"width": "210px", "fontSize": "12px"},
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="graph-story-disparity", figure=figures.disparity_battle(data, selected=None, slice_by="age"), config=GRAPH_CONFIG_CLEAN),
                            ]),
                            html.Div([
                                html.Div(
                                    style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px", "minHeight": "32px"},
                                    children=[
                                        html.Span("Subgroup Dimension:", style={"fontSize": "11.5px", "fontWeight": "600", "color": "#495057"}),
                                        dcc.RadioItems(
                                            id="story-subgroup-group",
                                            options=[
                                                {"label": " Race / Ethnicity", "value": "race"},
                                                {"label": " Admission Type", "value": "admission_type"},
                                                {"label": " Age Brackets", "value": "age"},
                                            ],
                                            value="race",
                                            inline=True,
                                            inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                                            style={"fontSize": "12px"},
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="graph-story-subgroup", figure=figures.subgroup_fairness(data, selected=None, group_by="race"), config=GRAPH_CONFIG_CLEAN),
                            ]),
                        ],
                    ),
                ],
            ),

            # Chapter 6: 10-Nearest-Neighbor Overlap Dossier
            html.Div(
                id="chapter-inspector",
                style={**CARD_STYLE, "marginBottom": "14px"},
                children=[
                    html.Div([
                        html.H3("Chapter 6 · Side-by-Side 10-NN Dossier: 22.4% Neighbor Overlap & Prompt Audit", style={"margin": "0 0 2px", "fontSize": "16px", "fontWeight": "700", "color": "#1D3557"}),
                        html.P("Direct encounter-level audit comparing retrieved neighbors, cosine similarities (1 - dist), and shared overlap.", style={"margin": "0 0 8px", "fontSize": "12px", "color": "#6C757D"}),
                    ]),
                    html.Div(
                        style={"backgroundColor": "#EAF6EF", "borderLeft": "4px solid #2A9D8F", "padding": "8px 12px", "borderRadius": "4px", "marginBottom": "10px", "fontSize": "12px", "lineHeight": "1.4", "color": "#1B4332"},
                        children=[
                            html.Strong("💡 Key Insight: "),
                            html.Span("Despite virtually identical ~63.0% accuracy, Model A and Model B share only 22.4% of their retrieved neighbors on average (2.24 of 10)! Inspect individual stays below to see how tabular feature matching and clinical text embeddings retrieve completely different cohorts."),
                        ],
                    ),
                    html.Div(
                        style={
                            "backgroundColor": "#F8F9FA",
                            "border": "1px solid #E9ECEF",
                            "borderRadius": "6px",
                            "padding": "10px 14px",
                            "display": "flex",
                            "gap": "14px",
                            "alignItems": "center",
                            "flexWrap": "wrap",
                            "marginBottom": "12px",
                        },
                        children=[
                            html.Div([
                                html.Label("Filter Examples: ", style={"fontSize": "11.5px", "fontWeight": "600", "marginRight": "6px"}),
                                dcc.Dropdown(
                                    id="story-case-type",
                                    options=[
                                        {"label": "Model A Correct, Model B Wrong (A Wins)", "value": "a_wins"},
                                        {"label": "Model B Correct, Model A Wrong (B Wins)", "value": "b_wins"},
                                        {"label": "Both Wrong", "value": "both_wrong"},
                                        {"label": "Any Encounter", "value": "all"},
                                    ],
                                    value="a_wins",
                                    clearable=False,
                                    style={"width": "260px", "fontSize": "12px"},
                                ),
                            ]),
                            html.Div(
                                style={"flex": "1"},
                                children=[
                                    html.Label("Select Encounter to Audit: ", style={"fontSize": "11.5px", "fontWeight": "600", "marginRight": "6px"}),
                                    dcc.Dropdown(
                                        id="story-encounter-dropdown",
                                        options=init_options,
                                        value=init_val,
                                        clearable=False,
                                        style={"fontSize": "12px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(id="story-retrieval-inspector"),
                ],
            ),
        ],
    )


def build_layout(data: DashboardData) -> html.Div:
    df = data.df
    metrics = data.metrics

    return html.Div(
        style={
            "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
            "backgroundColor": "#F4F6F8",
            "minHeight": "100vh",
            "padding": "14px 20px",
            "color": "#212529",
        },
        children=[
            # Header & Metrics Banner
            create_header_banner(df, metrics),

            # Global Action & Filter Status Bar
            create_global_toolbar(df),

            # State Stores for Cross-View Coordination
            dcc.Store(id="store-selected-ids", data=None),
            dcc.Store(id="store-active-cell", data=None),

            # Main Navigation Tabs
            dcc.Tabs(
                id="main-tabs",
                value="tab-story",
                style={"marginBottom": "12px"},
                children=[
                    dcc.Tab(
                        label="🎬 Curated Storyline (5-Min Video Walkthrough)",
                        value="tab-story",
                        style={"fontWeight": "600", "padding": "10px", "fontSize": "13px"},
                        selected_style={"fontWeight": "700", "color": "#1D3557", "borderTop": "3px solid #1D3557", "padding": "10px", "fontSize": "13px"},
                        children=[create_tab_story(data)],
                    ),
                    dcc.Tab(
                        label="🎛️ Primary Retrieval Studio (Views 1 – 4)",
                        value="tab-studio",
                        style={"fontWeight": "600", "padding": "10px", "fontSize": "13px"},
                        selected_style={"fontWeight": "700", "color": "#1D3557", "borderTop": "3px solid #1D3557", "padding": "10px", "fontSize": "13px"},
                        children=[
                            html.Div(style={"marginTop": "12px"}, children=[
                                # Row 1: View 1 (Left) + View 2 (Right)
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "1.08fr 1fr",
                                        "gap": "14px",
                                        "marginBottom": "14px",
                                        "alignItems": "stretch",
                                    },
                                    children=[
                                        create_view1_container(),
                                        create_view2_container(),
                                    ],
                                ),
                                # Row 2: View 3 (Left) + View 4 (Right)
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "1.08fr 1fr",
                                        "gap": "14px",
                                        "alignItems": "stretch",
                                    },
                                    children=[
                                        create_view3_container(data),
                                        create_view4_container(),
                                    ],
                                ),
                            ]),
                        ],
                    ),
                    dcc.Tab(
                        label="⚖️ Model Disparity & Reliability Audit (Calibration & Subgroups)",
                        value="tab-disparity",
                        style={"fontWeight": "600", "padding": "10px", "fontSize": "13px"},
                        selected_style={"fontWeight": "700", "color": "#1D3557", "borderTop": "3px solid #1D3557", "padding": "10px", "fontSize": "13px"},
                        children=[create_tab_disparity()],
                    ),
                    dcc.Tab(
                        label="🧬 Dynamic Cohort Profiler (Selection vs. Hospital Baseline)",
                        value="tab-profiler",
                        style={"fontWeight": "600", "padding": "10px", "fontSize": "13px"},
                        selected_style={"fontWeight": "700", "color": "#1D3557", "borderTop": "3px solid #1D3557", "padding": "10px", "fontSize": "13px"},
                        children=[create_tab_profiler()],
                    ),
                ],
            ),
        ],
    )
