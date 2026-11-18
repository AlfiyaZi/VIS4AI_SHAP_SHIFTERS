"""Phase II dashboard (Pair B, AI-aided): what did Model A and Model B learn?

Linked views around one selected stay:
  1 Map        the stays in Model A's or Model B's space (any layer), coloured by outcome / specialty / age /
               who is right / fragility; click a stay to select it
  2 Neighbours which fields the selected stay shares with its 10 neighbours in each model, and their vote
  3 Why        Model A's similarity split exactly into fields; for one neighbour, which differing field
               costs the most similarity in Model A and in Model B (Model B runs live)
  4 What-if    edit fields of the selected stay; both models re-run, the map shows where it moves
  5 Global     what each model weighs vs what predicts the outcome; groups compared
Run:  python dashboard/app.py   (from phase2/pair_B_ai_aided; needs the outputs of scripts 01-18 and
      dashboard/precompute.py). Open http://127.0.0.1:8051
"""
from __future__ import annotations

import dash
from dash import Input, Output, State, ctx, dcc, html, no_update

import figures as F
from data_layer import EDITABLE, FIELDS, GROUPS, WALKTHROUGH_STAY, load

data = load()
app = dash.Dash(__name__, title="SHAP_SHIFTERS - Phase II Explorer", suppress_callback_exceptions=True)
server = app.server

CARD = {"backgroundColor": "#FFFFFF", "borderRadius": "10px", "border": "1px solid #DEE2E6",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.04)", "padding": "14px 16px", "marginBottom": "14px"}
BAR = {"backgroundColor": "#F8F9FA", "border": "1px solid #E9ECEF", "borderRadius": "6px", "padding": "6px 10px",
       "fontSize": "12px", "display": "flex", "alignItems": "center", "gap": "12px", "flexWrap": "wrap", "margin": "6px 0"}
BTN = {"backgroundColor": "#1D3557", "color": "white", "border": "none", "borderRadius": "6px", "padding": "6px 12px",
       "fontSize": "12px", "cursor": "pointer"}
H = {"margin": "0 0 4px 0", "fontSize": "15px", "color": "#1D3557"}
NOTE = {"fontSize": "12px", "color": "#6C757D", "margin": "2px 0 6px 0"}
CLEAN = {"displaylogo": False, "modeBarButtonsToRemove": ["toImage", "sendDataToCloud", "select2d", "lasso2d"]}


def badge(title, val, sub, color):
    return html.Div([html.Span(title, style={"fontSize": "10.5px", "color": "#6C757D", "display": "block", "fontWeight": 600}),
                     html.Strong(val, style={"fontSize": "15px", "color": color, "display": "block"}),
                     html.Span(sub, style={"fontSize": "10px", "color": "#868E96"})],
                    style={"backgroundColor": "#F8F9FA", "border": "1px solid #E9ECEF", "borderRadius": "8px",
                           "padding": "5px 12px", "minWidth": "110px", "textAlign": "center"})


def card(step, title, note, *children, style=None):
    return html.Div([html.H3(f"{step} {title}", style=H), html.P(note, style=NOTE), *children], style={**CARD, **(style or {})})


df = data.df
header = html.Div(style={**CARD, "borderTop": "4px solid #1D3557", "display": "flex", "justifyContent": "space-between",
                         "flexWrap": "wrap", "gap": "12px", "alignItems": "center"}, children=[
    html.Div([html.H2("Phase II: What did Model A and Model B learn?",
                      style={"margin": "0 0 3px 0", "fontSize": "19px", "color": "#1D3557"}),
              html.P("Team SHAP_SHIFTERS | Pair B (AI-aided) · linked views 1 → 5 around one selected stay",
                     style={"margin": 0, "fontSize": "12.5px", "color": "#6C757D"})]),
    html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=[
        badge("Model A (features)", f"{df.model_a_correct.mean():.1%}", "accuracy, vote", F.CA),
        badge("Model B (BERT)", f"{df.model_b_correct.mean():.1%}", "accuracy, vote", F.CB),
        badge("Vote macro recall", "0.26", "chance 0.25", "#495057"),
        badge("Trained classifier", "0.37–0.39", "macro recall, same fields", F.NAVY),
        badge("Shared neighbours", f"{df.ab_neighbor_overlap.mean() * 10:.1f} / 10", "A vs B", "#495057"),
    ])])

controls = html.Div(style=BAR, children=[
    html.Span("Space"), dcc.Dropdown(id="space", options=list(data.maps), value="Model A (Phase I map)", clearable=False,
                                     style={"width": "210px"}),
    html.Span("Colour"), dcc.Dropdown(id="color", options=[{"label": v, "value": k} for k, v in F.COLOR_BY.items()],
                                      value="pathway", clearable=False, style={"width": "210px"}),
    html.Span("Group"), dcc.Dropdown(id="group", options=[{"label": v, "value": k} for k, v in GROUPS.items()],
                                     value="all", clearable=False, style={"width": "290px"}),
    dcc.Input(id="stay-id", type="number", placeholder="encounter id", debounce=True, style={"width": "120px"}),
    html.Button("Random stay from group", id="btn-random", n_clicks=0, style=BTN),
])

whatif_controls = html.Div(style={**BAR, "alignItems": "flex-start"}, children=[
    html.Div([html.Span(f, style={"fontSize": "11px", "display": "block", "color": "#495057"}),
              dcc.Dropdown(id=f"edit-{i}", options=data.categories[f], clearable=False, style={"width": "150px"})])
    for i, f in enumerate(EDITABLE)] + [html.Button("Run both models", id="btn-run", n_clicks=0,
                                                    style={**BTN, "alignSelf": "flex-end"})])

app.layout = html.Div(style={"backgroundColor": "#F1F3F5", "padding": "14px 18px", "fontFamily": F.FONT["family"],
                             "color": "#212529", "maxWidth": "1500px", "margin": "0 auto"}, children=[
    dcc.Store(id="store-row", data=data.pos[WALKTHROUGH_STAY]),
    dcc.Store(id="store-whatif"),
    header,
    html.Div(style={"display": "grid", "gridTemplateColumns": "3fr 2fr", "gap": "14px"}, children=[
        card("1", "Map: the stays as each model sees them",
             "Each dot is a stay (UMAP, cosine). Switch between Model A and Model B (layers 0, 6, 12). Click a stay to "
             "select it: ★ = selected, ○ = its Model A neighbours, ◇ = its Model B neighbours.",
             controls, dcc.Graph(id="map", config=CLEAN)),
        card("2", "The selected stay",
             "Its record (= Model A's input; Model B reads the same fields as one sentence) and the vote of its neighbours.",
             html.Div(id="record"), dcc.Graph(id="vote", config=CLEAN)),
    ]),
    card("2", "Neighbours: which fields does the selected stay share with them?",
         "Filled cell = the neighbour has the same value as the selected stay. Square on the left = the neighbour's "
         "next pathway (what it votes for). Model B's neighbours typically share specialty, source and payer, not the age.",
         dcc.Graph(id="match", config=CLEAN)),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"}, children=[
        card("3", "Why are they neighbours? Model A, exact",
             "Model A's cosine is a sum of one term per field (numbers: z-score products; categories: 1/0 for "
             "same/different, scaled by the vector lengths). Exact, no approximation.",
             dcc.Graph(id="decomp", config=CLEAN)),
        card("3", "Which difference costs the most? Model A vs Model B",
             "Pick one neighbour: each field where it differs from the selected stay is set to the stay's value, one at a "
             "time, and the gain in similarity is measured in both models. Model B runs Bio_ClinicalBERT live (first run "
             "loads the model, ~10 s).",
             html.Div(style=BAR, children=[dcc.Dropdown(id="nb-pick", clearable=False, style={"width": "260px"}),
                                           html.Button("Explain", id="btn-explain", n_clicks=0, style=BTN)]),
             dcc.Loading(dcc.Graph(id="swap", figure=F.empty("Pick a neighbour and press Explain"), config=CLEAN))),
    ]),
    card("4", "What-if: edit the selected stay and re-run both models",
         "Change any fields and press Run. Both models retrieve 10 new neighbours; the map draws an arrow to the centre "
         "of the new neighbours (an approximate new position) and marks them ×; the vote panel shows the new vote "
         "(white bars). Phase I's edits: sex swap, age +20 years, HbA1c → >8.",
         whatif_controls, dcc.Loading(html.Div(id="whatif-text", style={"fontSize": "13px", "marginTop": "6px"}))),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"}, children=[
        card("5", "What each model weighs",
             "Shapley: average similarity lost when a field differs (150 random pairs, same pairs for both models). "
             "Outcome: how much a classifier trained on the next pathway loses when the field is shuffled. Fields you "
             "edited in 4 are highlighted.",
             dcc.RadioItems(id="measure", options=[{"label": " Shapley vs outcome", "value": "shapley"},
                                                    {"label": " Share of the space (R²)", "value": "variance"}],
                            value="shapley", inline=True, style={"fontSize": "12px"}),
             dcc.Graph(id="weights", config=CLEAN)),
        card("5", "Groups",
             "The group chosen in 1, against all stays: accuracy of both votes, fragility under the HbA1c edit, and how "
             "many neighbours the two models share.",
             dcc.Graph(id="groups", config=CLEAN)),
    ]),
    html.P("Data: shared/data (Phase I) and phase2/pair_B_ai_aided/outputs (Phase II scripts 01-18). Built with Claude "
           "Code (AI-aided pair); see ai-interactions/phase2_ai_log.md.", style={**NOTE, "textAlign": "center"}),
])


# ---- callbacks ------------------------------------------------------------------------------
@app.callback(Output("store-row", "data"), Input("map", "clickData"), Input("btn-random", "n_clicks"),
              Input("stay-id", "value"), State("group", "value"), State("store-row", "data"), prevent_initial_call=True)
def select_row(click, _n, stay_id, group, current):
    t = ctx.triggered_id
    if t == "map" and click:
        p = click["points"][0]
        if "customdata" in p:
            return data.pos.get(int(p["customdata"][0]), current)
    if t == "btn-random":
        import numpy as np
        idx = np.flatnonzero(data.groups[group])
        return int(np.random.default_rng().choice(idx))
    if t == "stay-id" and stay_id is not None and int(stay_id) in data.pos:
        return data.pos[int(stay_id)]
    return no_update


@app.callback(Output("map", "figure"), Input("space", "value"), Input("color", "value"), Input("group", "value"),
              Input("store-row", "data"), Input("store-whatif", "data"))
def update_map(space, color, group, row, whatif):
    return F.map_figure(data, space, color, group, row, whatif)


@app.callback(Output("record", "children"), Output("match", "figure"), Output("decomp", "figure"),
              Output("nb-pick", "options"), Output("nb-pick", "value"),
              *[Output(f"edit-{i}", "value") for i in range(len(EDITABLE))], Input("store-row", "data"))
def update_stay(row):
    v = data.values[row]
    r = df.iloc[row]
    cells = [html.Tr([html.Td(f, style={"color": "#6C757D", "paddingRight": "10px"}), html.Td(v[f])]) for f in FIELDS]
    record = html.Div([
        html.Div([html.B(f"Stay {r.encounter_id}"), f" · patient {r.patient_nbr} · next pathway: ",
                  html.B(r.true_next_pathway, style={"color": F.PATHWAY_COLORS[r.true_next_pathway]})],
                 style={"fontSize": "13px", "marginBottom": "4px"}),
        html.Div(f"Model A predicts {r.model_a_pred_pathway} ({'correct' if r.model_a_correct else 'wrong'}); "
                 f"Model B predicts {r.model_b_pred_pathway} ({'correct' if r.model_b_correct else 'wrong'})",
                 style={"fontSize": "12px", "color": "#495057", "marginBottom": "6px"}),
        html.Table(cells, style={"fontSize": "12px", "columnCount": 2})])
    opts = []
    for m in "AB":
        nb, dist = data.phase1_neighbours(m, row)
        opts += [{"label": f"Model {m} neighbour #{i + 1} (stay {df.encounter_id.iat[j]})", "value": int(j)}
                 for i, j in enumerate(nb)]
    return (record, F.match_figure(data, row), F.decomposition_figure(data, row), opts, opts[10]["value"],
            *[v[f] for f in EDITABLE])


@app.callback(Output("vote", "figure"), Input("store-row", "data"), Input("store-whatif", "data"))
def update_vote(row, whatif):
    return F.vote_figure(data, row, whatif)


@app.callback(Output("swap", "figure"), Input("btn-explain", "n_clicks"), State("store-row", "data"),
              State("nb-pick", "value"), prevent_initial_call=True)
def explain(_n, row, other):
    if other is None:
        return F.empty("Pick a neighbour first")
    sw = {m: data.field_swap(m, row, other) for m in "AB"}
    if sw["A"].empty:
        return F.empty("The two stays have identical records: nothing differs")
    sw["other"] = int(df.encounter_id.iat[other])
    return F.swap_figure(sw)


@app.callback(Output("store-whatif", "data"), Output("whatif-text", "children"), Input("btn-run", "n_clicks"),
              State("store-row", "data"), *[State(f"edit-{i}", "value") for i in range(len(EDITABLE))],
              prevent_initial_call=True)
def run_whatif(_n, row, *vals):
    res = data.what_if(row, dict(zip(EDITABLE, vals)))
    if not res["changed"]:
        return None, "Nothing changed: edit at least one field."
    res["row"] = row
    rec, old = res["record"], data.values[row]
    lines = [html.Div("Edited: " + ", ".join(f"{f} {old[f]} → {rec[f]}" for f in res["changed"]))]
    for m, color in [("A", F.CA), ("B", F.CB)]:
        r = res[m]
        flip = r["vote_before"] != r["vote_after"]
        lines.append(html.Div([html.B(f"Model {m}: ", style={"color": color}),
                               f"kept {r['kept']} of 10 neighbours · moved {r['shift']:.4f} (cosine distance) · vote "
                               f"{r['vote_before']} → {r['vote_after']}",
                               html.B("  (prediction changed)", style={"color": "#9D0208"}) if flip else ""]))
    return res, lines


@app.callback(Output("weights", "figure"), Input("measure", "value"), Input("store-whatif", "data"))
def update_weights(measure, whatif):
    return F.weights_figure(data, measure, (whatif or {}).get("changed"))


@app.callback(Output("groups", "figure"), Input("group", "value"))
def update_groups(group):
    return F.groups_figure(data, group)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8051)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    print(f"Phase II dashboard: http://127.0.0.1:{a.port}")
    app.run(port=a.port, debug=a.debug)
