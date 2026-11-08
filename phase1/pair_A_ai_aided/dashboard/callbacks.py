"""Callbacks for the six retained single-page Phase I analyses."""
from dash import Input, Output, State, ctx, html, no_update
import numpy as np
import figures
import data_layer


def register_callbacks(app, data: data_layer.DashboardData):
    df=data.df

    @app.callback(Output('store-selected-ids','data'),Output('store-active-cell','data'),
        Input('btn-reset','n_clicks'),Input('graph-view2-model-a','clickData'),Input('graph-view2-model-b','clickData'),
        State('store-active-cell','data'))
    def select_group(reset, click_a, click_b, prev):
        trig=ctx.triggered_id
        if not trig or trig=='btn-reset':
            return None,None
        model='model_a' if trig=='graph-view2-model-a' else 'model_b'
        clicked=click_a if model=='model_a' else click_b
        if not clicked or not clicked.get('points') or clicked['points'][0].get('customdata') is None:
            return no_update,no_update
        t,p=clicked['points'][0]['customdata'][:2]
        active={'model':model,'true':str(t),'pred':str(p)}
        if prev==active:
            return None,None
        mask=(df.true_next_pathway==t)&(df[f'{model}_pred_pathway']==p)
        return [int(e) for e in df.index[mask]],active

    @app.callback(Output('selection-status','children'),Input('store-selected-ids','data'))
    def status(ids):
        return f"Selected: {len(ids):,} encounters (click the same cell again or Clear selection to reset)" if ids is not None else "Showing full cohort"

    @app.callback(Output('graph-view2-model-a','figure'),Output('graph-view2-model-b','figure'),Input('store-active-cell','data'))
    def confusion(active):
        hl_a=(active['true'],active['pred']) if active and active['model']=='model_a' else None
        hl_b=(active['true'],active['pred']) if active and active['model']=='model_b' else None
        return figures.confusion_heatmap(data,'model_a',True,False,hl_a),figures.confusion_heatmap(data,'model_b',True,False,hl_b)

    @app.callback(Output('graph-diag-disparity','figure'),Input('store-selected-ids','data'),Input('diag-disparity-slice','value'))
    def disagreement(ids, slice_by):
        return figures.disparity_battle(data,set(ids) if ids is not None else None,slice_by or 'age')

    @app.callback(Output('encounter-inspect-dropdown','options'),Output('encounter-inspect-dropdown','value'),
        Input('inspector-case-type','value'),Input('store-selected-ids','data'),State('encounter-inspect-dropdown','value'))
    def inspector_options(case_type,ids,current):
        sub=df.loc[df.index.isin(ids)] if ids is not None else df
        a=sub.model_a_correct.astype(bool); b=sub.model_b_correct.astype(bool)
        if case_type=='a_wins': sub=sub[a & ~b]
        elif case_type=='b_wins': sub=sub[b & ~a]
        elif case_type=='both_wrong': sub=sub[~a & ~b]
        if sub.empty:
            return [],None
        # A searchable list of representative encounters, plus the current ID if valid.
        chosen=[int(e) for e in sub.index[:300]]
        if current is not None and current in sub.index and int(current) not in chosen:
            chosen.insert(0,int(current))
        options=[{'label':f"#{eid} · {df.loc[eid,'age']} · {df.loc[eid,'true_next_pathway']}",'value':eid} for eid in chosen]
        return options,int(current) if current in chosen else chosen[0]

    @app.callback(Output('retrieval-inspector','children'),Input('encounter-inspect-dropdown','value'))
    def inspector(eid):
        if eid is None or eid not in df.index:
            return html.P('No encounters match this selection. Clear the confusion-matrix filter or change example type.')
        row=df.loc[eid]
        ids_a=[int(v) for v in row.model_a_neighbors]
        ids_b=[int(v) for v in row.model_b_neighbors]
        overlap=set(ids_a)&set(ids_b)
        def make_table(model,neighbors):
            distances=row[f'{model}_distances']
            rows=[]
            for rank,(nid,dist) in enumerate(zip(neighbors,distances),start=1):
                nb=df.loc[nid]
                rows.append(html.Tr([
                    html.Td(str(rank)),html.Td(str(nid)),html.Td(f'{1-float(dist):.3f}'),
                    html.Td(str(nb.true_next_pathway)),html.Td('Shared' if nid in overlap else '—')],
                    style={'backgroundColor':'#EAF6EF' if nid in overlap else 'white'}))
            return html.Div(style={'minWidth':'0'},children=[
                html.H4('Model A · Tabular' if model=='model_a' else 'Model B · ClinicalBERT',style={'margin':'6px 0'}),
                html.P(f"Prediction: {row[f'{model}_pred_pathway']} · Vote share: {row[f'{model}_vote_share']:.0%}",style={'fontSize':'13px'}),
                html.Div(style={'overflowX':'auto'},children=[html.Table(style={'borderCollapse':'collapse','width':'100%','fontSize':'12px'},children=[
                    html.Thead(html.Tr([html.Th(s,style={'textAlign':'left','padding':'6px'}) for s in ['Rank','Encounter','Cosine sim.','Neighbor next pathway','Overlap']])),
                    html.Tbody(rows)])])])
        return html.Div([
            html.P([html.Strong(f'Encounter #{eid}'),f" · Patient #{int(row.patient_nbr)} · {row.gender}, {row.age} · Actual next pathway: {row.true_next_pathway}"]),
            html.P(f"Shared neighbors: {len(overlap)} / 10. Green rows appear in both lists. Similarity = 1 − saved cosine distance.",style={'fontSize':'12px','color':'#596575'}),
            html.Div(style={'display':'grid','gridTemplateColumns':'repeat(2,minmax(0,1fr))','gap':'20px'},children=[make_table('model_a',ids_a),make_table('model_b',ids_b)]),
            html.Details([html.Summary('Show ClinicalBERT input text'),html.P(str(row.model_text),style={'fontFamily':'monospace','fontSize':'12px'})],style={'marginTop':'14px'}),
        ])

    @app.callback(Output('graph-view4-flips','figure'),Input('store-selected-ids','data'))
    def sensitivity(ids):
        return figures.flips_view(data,set(ids) if ids is not None else None)

    @app.callback(Output('graph-diag-calibration','figure'),Input('store-selected-ids','data'))
    def reliability(ids):
        return figures.calibration_curve(data,set(ids) if ids is not None else None)

    @app.callback(Output('graph-diag-subgroup','figure'),Input('store-selected-ids','data'),Input('diag-subgroup-group','value'))
    def subgroup(ids,group):
        return figures.subgroup_fairness(data,set(ids) if ids is not None else None,group or 'race')
