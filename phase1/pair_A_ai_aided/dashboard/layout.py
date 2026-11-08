"""Single-page Phase I dashboard: six agreed analyses, top to bottom."""
from dash import dcc, html
import data_layer

CARD = {"backgroundColor":"white","border":"1px solid #DEE2E6","borderRadius":"10px","padding":"18px","marginBottom":"18px"}
GRAPH = {"displaylogo":False,"modeBarButtonsToRemove":["toImage","sendDataToCloud"]}

def section(title, subtitle, children, anchor):
    return html.Section(id=anchor, style=CARD, children=[
        html.H3(title,style={"margin":"0 0 5px","color":"#1D3557","fontSize":"19px"}),
        html.P(subtitle,style={"color":"#596575","fontSize":"13px","margin":"0 0 14px"}),
        *children,
    ])

def build_layout(data: data_layer.DashboardData):
    df=data.df
    a=float(df.model_a_correct.mean()); b=float(df.model_b_correct.mean())
    baseline=float((df.majority_pred_pathway==df.true_next_pathway).mean())
    return html.Div(style={"backgroundColor":"#F4F6F8","minHeight":"100vh","padding":"22px 4%","fontFamily":"Arial, sans-serif","color":"#212529"},children=[
        html.Header(style={**CARD,"borderTop":"4px solid #1D3557"},children=[
            html.H2("SHAP_SHIFTERS · Phase I Retrieval Behavior",style={"margin":"0 0 8px","color":"#1D3557"}),
            html.P(f"{len(df):,} encounters · {df.patient_nbr.nunique():,} patients · Model A {a:.1%} · Model B {b:.1%} · Majority baseline {baseline:.1%}",style={"margin":"0 0 12px"}),
            html.Button("Clear selection",id="btn-reset",n_clicks=0,style={"padding":"8px 16px","cursor":"pointer"}),
            html.Span(id="selection-status",style={"marginLeft":"15px","fontSize":"13px"}),
            html.Div([html.A(t,href=f"#{anchor}",style={"marginRight":"18px","fontSize":"13px"}) for t,anchor in [
                ("1 Performance","performance"),("2 Disagreements","disagreement"),("3 Retrieval","retrieval"),
                ("4 Sensitivity","sensitivity"),("5 Reliability","reliability"),("6 Subgroups","subgroups")]],style={"marginTop":"14px"}),
        ]),
        dcc.Store(id="store-selected-ids"),dcc.Store(id="store-active-cell"),
        section("1 · Prediction Performance", "Row-normalized confusion matrices. Click any cell to inspect that error group; hover for exact counts.",[
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(2,minmax(0,1fr))","gap":"16px"},children=[
                dcc.Graph(id="graph-view2-model-a",config=GRAPH),dcc.Graph(id="graph-view2-model-b",config=GRAPH)]),
            html.Div(f"Majority baseline: {baseline:.1%} accuracy — always predicts Uncomplicated / Other.",style={"fontWeight":"600","fontSize":"13px"}),
        ],"performance"),
        section("2 · Where Does Each Model Perform Better?", "Only encounters where A and B disagree. Bars show Model A wins, Model B wins, or both wrong.",[
            html.Div([html.Span("Compare by: "),dcc.Dropdown(id="diag-disparity-slice",options=[
                {"label":"Age group","value":"age"},{"label":"True next pathway","value":"true_next_pathway"},
                {"label":"Diagnoses count","value":"number_diagnoses"},{"label":"Length of stay","value":"time_in_hospital"},
                {"label":"Medication change","value":"change"}],value="age",clearable=False,style={"width":"270px"})],style={"display":"flex","alignItems":"center","gap":"12px"}),
            dcc.Graph(id="graph-diag-disparity",config=GRAPH),
        ],"disagreement"),
        section("3 · Compare the 10 Retrieved Neighbors", "Select an encounter. Compare the actual neighbors, their next-visit labels, cosine similarities, and overlap.",[
            html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap","alignItems":"center"},children=[
                html.Div([html.Label("Interesting examples"),dcc.Dropdown(id="inspector-case-type",options=[
                    {"label":"Model A correct, Model B wrong","value":"a_wins"},
                    {"label":"Model B correct, Model A wrong","value":"b_wins"},
                    {"label":"Both wrong","value":"both_wrong"},
                    {"label":"Any encounter","value":"all"}],value="a_wins",clearable=False,style={"width":"270px"})]),
                html.Div([html.Label("Encounter"),dcc.Dropdown(id="encounter-inspect-dropdown",clearable=False,style={"width":"390px"})]),
            ]),
            html.Div(id="retrieval-inspector",style={"marginTop":"16px"}),
        ],"retrieval"),
        section("4 · Retrieval Sensitivity to Feature Changes", "Average number of replaced neighbors (out of 10) after synthetic edits; error bars show 95% confidence intervals.",[
            dcc.Graph(id="graph-view4-flips",config=GRAPH),
        ],"sensitivity"),
        section("5 · Does Neighbor Consensus Predict Accuracy?", "Observed accuracy versus fraction of neighbors voting for the predicted class; 95% confidence intervals.",[
            dcc.Graph(id="graph-diag-calibration",config=GRAPH),
        ],"reliability"),
        section("6 · Subgroup Performance (Secondary Analysis)", "Accuracy lift relative to each subgroup's own majority baseline; not a comprehensive fairness assessment.",[
            html.Div([html.Span("Subgroup: "),dcc.RadioItems(id="diag-subgroup-group",options=[
                {"label":"Race / ethnicity","value":"race"},{"label":"Admission type","value":"admission_type"},
                {"label":"Age","value":"age"}],value="race",inline=True)],style={"display":"flex","gap":"14px"}),
            dcc.Graph(id="graph-diag-subgroup",config=GRAPH),
        ],"subgroups"),
    ])
