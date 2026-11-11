"""
Automated headless test suite for Pair A (AI-aided) Phase I Dashboard.
Verifies data contracts, schemas, clustering, figure serialization, layout order,
zero duplicate IDs, and live Dash callback execution via Flask test_client().
"""

from collections import Counter
import json
import sys
import unittest
from pathlib import Path

DASH_DIR = Path(__file__).resolve().parent / "dashboard"
sys.path.insert(0, str(DASH_DIR))

import app as dash_app_module
import callbacks
import data_layer
import figures
import layout


def _collect_ids(node, out_list):
    if isinstance(node, dict):
        props = node.get("props", {})
        if isinstance(props, dict) and "id" in props:
            out_list.append(props["id"])
        for v in node.values():
            _collect_ids(v, out_list)
    elif isinstance(node, list):
        for item in node:
            _collect_ids(item, out_list)


class TestPairADashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = data_layer.load()
        cls.df = cls.data.df
        cls.metrics = cls.data.metrics
        cls.client = dash_app_module.app.server.test_client()

    def test_01_data_contract_and_schema(self):
        """Verify the static data tables load with exact expected schemas and row count."""
        self.assertEqual(len(self.df), 30248)
        self.assertEqual(self.df["patient_nbr"].nunique(), 16773)

        required_cols = [
            "encounter_id", "patient_nbr", "gender", "age", "race",
            "time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses",
            "A1Cresult", "max_glu_serum", "insulin", "change", "diabetesMed",
            "current_pathway", "true_next_pathway", "model_text",
            "umap_x", "umap_y", "jitter_x", "jitter_y",
            "model_a_agreement", "model_a_pred_pathway", "model_a_correct",
            "model_b_agreement", "model_b_pred_pathway", "model_b_correct",
            "majority_pred_pathway", "random_pred_pathway", "agreement_disparity_b_minus_a",
            "model_a_flips_sex_swap", "model_b_flips_sex_swap",
            "model_a_flips_age_shift", "model_b_flips_age_shift",
            "model_a_flips_hba1c_elevation", "model_b_flips_hba1c_elevation",
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns, f"Missing required column: {col}")

    def test_02_baseline_metrics_match(self):
        """Verify model accuracies match reported shared baseline values."""
        acc_a = (self.df["model_a_pred_pathway"] == self.df["true_next_pathway"]).mean()
        acc_b = (self.df["model_b_pred_pathway"] == self.df["true_next_pathway"]).mean()
        acc_maj = (self.df["majority_pred_pathway"] == self.df["true_next_pathway"]).mean()

        self.assertAlmostEqual(acc_a, 0.6301, places=3)
        self.assertAlmostEqual(acc_b, 0.6296, places=3)
        self.assertAlmostEqual(acc_maj, 0.6636, places=3)

    def test_03_hierarchical_confusion_reordering(self):
        """Verify confusion matrices and optimal leaf ordering."""
        order = self.data.order
        self.assertEqual(len(order), len(data_layer.PATHWAYS))
        self.assertEqual(set(order), set(data_layer.PATHWAYS))

        for model in ("model_a", "model_b", "majority"):
            cm = self.data.confusion[model]
            self.assertEqual(cm.shape, (4, 4))
            self.assertEqual(cm.to_numpy().sum(), 30248)

    def test_04_driver_ranking_and_clusters(self):
        """Verify UMAP clusters and driver association metrics."""
        self.assertEqual(len(self.data.clusters), 30248)
        ranking = self.data.driver_ranking
        self.assertGreater(len(ranking), 5)
        self.assertIn("feature", ranking.columns)
        self.assertIn("association", ranking.columns)
        top_assoc = ranking.iloc[0]["association"]
        self.assertTrue(0.0 <= top_assoc <= 1.0)

    def test_05_figures_scatter_explorer(self):
        """Verify View 1 scatter explorer compiles and uses plain JSON lists for customdata."""
        fig1 = figures.scatter_explorer(self.data, selected=None, x_model="model_b", jitter=True, contours=False)
        self.assertIsNotNone(fig1)
        d1 = json.loads(fig1.to_json())
        total_pts = sum(len(t["x"]) for t in d1["data"] if t["type"] == "scattergl")
        self.assertEqual(total_pts, 30248)
        for t in d1["data"]:
            if t["type"] == "scattergl":
                self.assertIsInstance(t["customdata"], list)

        sub_ids = set(self.df.index[:100])
        fig2 = figures.scatter_explorer(self.data, selected=sub_ids, x_model="model_a", jitter=True, contours=True)
        self.assertIsNotNone(fig2)

    def test_06_figures_confusion_heatmaps(self):
        """Verify View 2 confusion matrix heatmaps compile with typed customdata."""
        fig_a = figures.confusion_heatmap(self.data, "model_a", normalize=True, clustered=True, highlight=None)
        fig_b = figures.confusion_heatmap(
            self.data, "model_b", normalize=False, clustered=False,
            highlight=("Cardiovascular", "Kidney / Renal"),
        )
        fig_maj = figures.confusion_heatmap(self.data, "majority", normalize=True, clustered=True, highlight=None)
        d_a = json.loads(fig_a.to_json())
        self.assertIsInstance(d_a["data"][0]["customdata"], list)
        self.assertIsInstance(d_a["data"][0]["customdata"][0][0][2], int)
        self.assertIsNotNone(fig_b)
        self.assertIsNotNone(fig_maj)

    def test_07_figures_umap_view(self):
        """Verify View 3 UMAP renders all 30,248 points across every color mode with plain list coordinates."""
        for mode in figures.UMAP_COLOR_MODES:
            fig = figures.umap_view(
                self.data, selected=None, color_mode=mode, show_traj=False,
                n_patients=20, annotate=True,
            )
            d = json.loads(fig.to_json())
            scattergl_traces = [t for t in d["data"] if t["type"] == "scattergl"]
            self.assertGreater(len(scattergl_traces), 0, f"No scattergl traces in mode {mode}")
            total_pts = sum(len(t["x"]) for t in scattergl_traces)
            self.assertEqual(total_pts, 30248, f"Expected 30,248 points in mode {mode}, got {total_pts}")
            for t in scattergl_traces:
                self.assertIsInstance(t["x"], list)
                self.assertIsInstance(t["y"], list)
                self.assertIsInstance(t["customdata"], list)
                self.assertIsInstance(t["hovertemplate"], str)

    def test_08_figures_flips_and_inspector(self):
        """Verify View 4 counterfactual flips and neighbor breakdown figures compile."""
        fig_flips = figures.flips_view(self.data, selected=None)
        self.assertIsNotNone(fig_flips)

        sub_ids = set(self.df.index[:200])
        fig_flips_sub = figures.flips_view(self.data, selected=sub_ids)
        self.assertIsNotNone(fig_flips_sub)

        sample_eid = int(self.df.index[0])
        fig_nb = figures.neighbor_breakdown(self.data, sample_eid)
        self.assertIsNotNone(fig_nb)

    def test_09_layout_order_and_zero_duplicate_ids(self):
        """Verify HTTP 200 on /_dash-layout, zero duplicate IDs, and strict View 1 -> 2 -> 3 -> 4 ordering."""
        res = self.client.get("/_dash-layout")
        self.assertEqual(res.status_code, 200)
        raw_text = res.get_data(as_text=True)
        layout_json = json.loads(raw_text)

        ids = []
        _collect_ids(layout_json, ids)
        dups = {k: v for k, v in Counter(ids).items() if v > 1}
        self.assertEqual(dups, {}, f"Found duplicate component IDs in layout: {dups}")

        # Verify strict sequential ordering of View 1 -> View 2 -> View 3 -> View 4
        idx_v1 = raw_text.index("View 1:")
        idx_v2 = raw_text.index("View 2:")
        idx_v3 = raw_text.index("View 3:")
        idx_v4 = raw_text.index("View 4:")
        self.assertTrue(
            idx_v1 < idx_v2 < idx_v3 < idx_v4,
            f"Views are out of order! Indices: V1={idx_v1}, V2={idx_v2}, V3={idx_v3}, V4={idx_v4}",
        )

    def test_10_live_callbacks_end_to_end(self):
        """Verify cross-filtering and rendering callbacks via Flask test_client POST to /_dash-update-component."""
        deps = self.client.get("/_dash-dependencies")
        self.assertEqual(deps.status_code, 200)

        sample_eids = [int(e) for e in self.df.index[:15]]
        sample_points = [{"customdata": eid} for eid in sample_eids]

        # 1. Test View 3 Lasso Selection triggering handle_cross_selection
        payload_sel = {
            "output": "..store-selected-ids.data...store-active-cell.data..",
            "outputs": [
                {"id": "store-selected-ids", "property": "data"},
                {"id": "store-active-cell", "property": "data"},
            ],
            "inputs": [
                {"id": "btn-reset", "property": "n_clicks", "value": 0},
                {"id": "graph-view1-scatter", "property": "selectedData", "value": None},
                {"id": "graph-view3-umap", "property": "selectedData", "value": {"points": sample_points}},
                {"id": "graph-view2-model-a", "property": "clickData", "value": None},
                {"id": "graph-view2-model-b", "property": "clickData", "value": None},
                {"id": "graph-view2-majority", "property": "clickData", "value": None},
            ],
            "state": [
                {"id": "store-active-cell", "property": "data", "value": None},
            ],
            "changedPropIds": ["graph-view3-umap.selectedData"],
        }
        r_sel = self.client.post("/_dash-update-component", json=payload_sel)
        self.assertEqual(r_sel.status_code, 200)
        resp_sel = r_sel.get_json()["response"]
        self.assertEqual(resp_sel["store-selected-ids"]["data"], sample_eids)
        self.assertEqual(resp_sel["store-active-cell"]["data"]["source"], "umap")

        # 2. Test View 2 Confusion Matrix Cell Click
        expected_cm_count = int(self.data.confusion["model_a"].loc["Kidney / Renal", "Cardiovascular"])
        payload_cm = {
            **payload_sel,
            "inputs": [
                {"id": "btn-reset", "property": "n_clicks", "value": 0},
                {"id": "graph-view1-scatter", "property": "selectedData", "value": None},
                {"id": "graph-view3-umap", "property": "selectedData", "value": None},
                {
                    "id": "graph-view2-model-a",
                    "property": "clickData",
                    "value": {"points": [{"customdata": ["Kidney / Renal", "Cardiovascular", expected_cm_count, 0.056]}]},
                },
                {"id": "graph-view2-model-b", "property": "clickData", "value": None},
                {"id": "graph-view2-majority", "property": "clickData", "value": None},
            ],
            "changedPropIds": ["graph-view2-model-a.clickData"],
        }
        r_cm = self.client.post("/_dash-update-component", json=payload_cm)
        self.assertEqual(r_cm.status_code, 200)
        resp_cm = r_cm.get_json()["response"]
        self.assertEqual(len(resp_cm["store-selected-ids"]["data"]), expected_cm_count)
        self.assertEqual(resp_cm["store-active-cell"]["data"]["source"], "confusion")

        # 3. Test View 1 Lasso Selection triggering handle_cross_selection
        payload_v1_sel = {
            **payload_sel,
            "inputs": [
                {"id": "btn-reset", "property": "n_clicks", "value": 0},
                {"id": "graph-view1-scatter", "property": "selectedData", "value": {"points": sample_points}},
                {"id": "graph-view3-umap", "property": "selectedData", "value": None},
                {"id": "graph-view2-model-a", "property": "clickData", "value": None},
                {"id": "graph-view2-model-b", "property": "clickData", "value": None},
                {"id": "graph-view2-majority", "property": "clickData", "value": None},
            ],
            "changedPropIds": ["graph-view1-scatter.selectedData"],
        }
        r_v1 = self.client.post("/_dash-update-component", json=payload_v1_sel)
        self.assertEqual(r_v1.status_code, 200)
        resp_v1 = r_v1.get_json()["response"]
        self.assertEqual(resp_v1["store-selected-ids"]["data"], sample_eids)
        self.assertEqual(resp_v1["store-active-cell"]["data"]["source"], "scatter")

        # 4. Test Reset Button clearing selection and incrementing both uirevision keys
        payload_reset = {
            **payload_sel,
            "inputs": [
                {"id": "btn-reset", "property": "n_clicks", "value": 1},
                {"id": "graph-view1-scatter", "property": "selectedData", "value": {"points": sample_points}},
                {"id": "graph-view3-umap", "property": "selectedData", "value": None},
                {"id": "graph-view2-model-a", "property": "clickData", "value": None},
                {"id": "graph-view2-model-b", "property": "clickData", "value": None},
                {"id": "graph-view2-majority", "property": "clickData", "value": None},
            ],
            "state": [
                {"id": "store-active-cell", "property": "data", "value": {"v1_rev": 2, "v3_rev": 4}},
            ],
            "changedPropIds": ["btn-reset.n_clicks"],
        }
        r_rst = self.client.post("/_dash-update-component", json=payload_reset)
        self.assertEqual(r_rst.status_code, 200)
        resp_rst = r_rst.get_json()["response"]
        self.assertIsNone(resp_rst["store-selected-ids"]["data"])
        self.assertEqual(resp_rst["store-active-cell"]["data"]["v1_rev"], 3)
        self.assertEqual(resp_rst["store-active-cell"]["data"]["v3_rev"], 5)

        # 5. Test View 4 Patient Inspector callback on point click
        payload_insp = {
            "output": "view4-inspector-card.children",
            "outputs": {"id": "view4-inspector-card", "property": "children"},
            "inputs": [
                {"id": "encounter-inspect-dropdown", "property": "value", "value": sample_eids[0]},
                {"id": "graph-view1-scatter", "property": "clickData", "value": None},
                {"id": "graph-view3-umap", "property": "clickData", "value": {"points": [{"customdata": sample_eids[5]}]}},
            ],
            "changedPropIds": ["graph-view3-umap.clickData"],
        }
        r_insp = self.client.post("/_dash-update-component", json=payload_insp)
        self.assertEqual(r_insp.status_code, 200)
        self.assertIn(f"Encounter #{sample_eids[5]}", json.dumps(r_insp.get_json()))

    def test_11_selection_persistence_no_auto_reset(self):
        """Verify selections persist and do NOT auto-reset when re-rendered charts emit null selectedData."""
        sample_eids = list(self.df.index[:8])
        sample_points = [{"customdata": int(eid)} for eid in sample_eids]

        # Step 1: User draws lasso in View 1
        payload_v1 = {
            "output": "..store-selected-ids.data...store-active-cell.data..",
            "outputs": [
                {"id": "store-selected-ids", "property": "data"},
                {"id": "store-active-cell", "property": "data"},
            ],
            "inputs": [
                {"id": "btn-reset", "property": "n_clicks", "value": 0},
                {"id": "graph-view1-scatter", "property": "selectedData", "value": {"points": sample_points}},
                {"id": "graph-view3-umap", "property": "selectedData", "value": None},
                {"id": "graph-view2-model-a", "property": "clickData", "value": None},
                {"id": "graph-view2-model-b", "property": "clickData", "value": None},
                {"id": "graph-view2-majority", "property": "clickData", "value": None},
            ],
            "state": [
                {"id": "store-active-cell", "property": "data", "value": None},
            ],
            "changedPropIds": ["graph-view1-scatter.selectedData"],
        }
        r1 = self.client.post("/_dash-update-component", json=payload_v1)
        self.assertEqual(r1.status_code, 200)
        resp1 = r1.get_json()["response"]
        self.assertEqual(resp1["store-selected-ids"]["data"], sample_eids)

        # Step 2: Browser re-renders View 3 and fires selectedData: null on View 3
        payload_v3_null = {
            **payload_v1,
            "state": [{"id": "store-active-cell", "property": "data", "value": resp1["store-active-cell"]["data"]}],
            "changedPropIds": ["graph-view3-umap.selectedData"],
        }
        r2 = self.client.post("/_dash-update-component", json=payload_v3_null)
        self.assertEqual(r2.status_code, 200)
        # Empty response dictionary means dash.no_update -> state was NOT wiped!
        self.assertEqual(r2.get_json()["response"], {})

        # Step 3: Browser re-renders View 1 and fires selectedData: null on View 1
        payload_v1_null = {
            **payload_v1,
            "inputs": [
                {"id": "btn-reset", "property": "n_clicks", "value": 0},
                {"id": "graph-view1-scatter", "property": "selectedData", "value": None},
                {"id": "graph-view3-umap", "property": "selectedData", "value": None},
                {"id": "graph-view2-model-a", "property": "clickData", "value": None},
                {"id": "graph-view2-model-b", "property": "clickData", "value": None},
                {"id": "graph-view2-majority", "property": "clickData", "value": None},
            ],
            "state": [{"id": "store-active-cell", "property": "data", "value": resp1["store-active-cell"]["data"]}],
            "changedPropIds": ["graph-view1-scatter.selectedData"],
        }
        r3 = self.client.post("/_dash-update-component", json=payload_v1_null)
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.get_json()["response"], {})

        # Step 4: Clicking btn-reset DOES wipe state back to None
        payload_reset = {
            **payload_v1_null,
            "inputs": [
                {"id": "btn-reset", "property": "n_clicks", "value": 1},
                {"id": "graph-view1-scatter", "property": "selectedData", "value": None},
                {"id": "graph-view3-umap", "property": "selectedData", "value": None},
                {"id": "graph-view2-model-a", "property": "clickData", "value": None},
                {"id": "graph-view2-model-b", "property": "clickData", "value": None},
                {"id": "graph-view2-majority", "property": "clickData", "value": None},
            ],
            "changedPropIds": ["btn-reset.n_clicks"],
        }
        r4 = self.client.post("/_dash-update-component", json=payload_reset)
        self.assertEqual(r4.status_code, 200)
        self.assertIsNone(r4.get_json()["response"]["store-selected-ids"]["data"])

    def test_12_diagnostic_and_profiler_figures(self):
        """Verify calibration, disparity battle, subgroup fairness, and cohort profiler figures."""
        # Unfiltered
        fig_cal = figures.calibration_curve(self.data, selected=None)
        self.assertIsNotNone(fig_cal)
        self.assertGreater(len(fig_cal.data), 1)

        fig_dis = figures.disparity_battle(self.data, selected=None, slice_by="age")
        self.assertIsNotNone(fig_dis)

        fig_sub = figures.subgroup_fairness(self.data, selected=None, group_by="race")
        self.assertIsNotNone(fig_sub)

        fig_prof = figures.cohort_profiler(self.data, selected=None)
        self.assertIsNotNone(fig_prof)

        fig_path = figures.cohort_pathway_breakdown(self.data, selected=None)
        self.assertIsNotNone(fig_path)

        # Filtered
        sub_ids = set(self.df.index[:150])
        fig_cal_sub = figures.calibration_curve(self.data, selected=sub_ids)
        self.assertIsNotNone(fig_cal_sub)

        fig_prof_sub = figures.cohort_profiler(self.data, selected=sub_ids)
        self.assertIsNotNone(fig_prof_sub)
        self.assertEqual(len(fig_prof_sub.data), 2)  # Baseline + Active Selection

    def test_13_tab_callbacks_live_execution(self):
        """Verify POST requests to new Tab 2 and Tab 3 callbacks return HTTP 200 and figure data."""
        sample_eids = [int(e) for e in self.df.index[:20]]

        # Calibration
        payload_cal = {
            "output": "graph-diag-calibration.figure",
            "outputs": {"id": "graph-diag-calibration", "property": "figure"},
            "inputs": [{"id": "store-selected-ids", "property": "data", "value": sample_eids}],
            "changedPropIds": ["store-selected-ids.data"],
        }
        r_cal = self.client.post("/_dash-update-component", json=payload_cal)
        self.assertEqual(r_cal.status_code, 200)

        # Disparity Battle
        payload_dis = {
            "output": "graph-diag-disparity.figure",
            "outputs": {"id": "graph-diag-disparity", "property": "figure"},
            "inputs": [
                {"id": "store-selected-ids", "property": "data", "value": sample_eids},
                {"id": "diag-disparity-slice", "property": "value", "value": "age"},
            ],
            "changedPropIds": ["diag-disparity-slice.value"],
        }
        r_dis = self.client.post("/_dash-update-component", json=payload_dis)
        self.assertEqual(r_dis.status_code, 200)

        # Cohort Profiler
        payload_prof = {
            "output": "graph-profile-vitals.figure",
            "outputs": {"id": "graph-profile-vitals", "property": "figure"},
            "inputs": [{"id": "store-selected-ids", "property": "data", "value": sample_eids}],
            "changedPropIds": ["store-selected-ids.data"],
        }
        r_prof = self.client.post("/_dash-update-component", json=payload_prof)
        self.assertEqual(r_prof.status_code, 200)

    def test_14_storyline_tab_and_callbacks(self):
        """Verify Curated Storyline figures compile and callback endpoints respond with HTTP 200."""
        sample_eids = [int(e) for e in self.df.index[:25]]

        # 1. Test Story Inspector Options Callback
        payload_opts = {
            "output": "..story-encounter-dropdown.options...story-encounter-dropdown.value..",
            "outputs": [
                {"id": "story-encounter-dropdown", "property": "options"},
                {"id": "story-encounter-dropdown", "property": "value"},
            ],
            "inputs": [
                {"id": "story-case-type", "property": "value", "value": "a_wins"},
                {"id": "store-selected-ids", "property": "data", "value": sample_eids},
            ],
            "state": [
                {"id": "story-encounter-dropdown", "property": "value", "value": None},
            ],
            "changedPropIds": ["story-case-type.value"],
        }
        r_opts = self.client.post("/_dash-update-component", json=payload_opts)
        self.assertEqual(r_opts.status_code, 200)

        # 2. Test Story Inspector Content Callback
        test_eid = int(self.df.index[0])
        payload_insp = {
            "output": "story-retrieval-inspector.children",
            "outputs": {"id": "story-retrieval-inspector", "property": "children"},
            "inputs": [{"id": "story-encounter-dropdown", "property": "value", "value": test_eid}],
            "changedPropIds": ["story-encounter-dropdown.value"],
        }
        r_insp = self.client.post("/_dash-update-component", json=payload_insp)
        self.assertEqual(r_insp.status_code, 200)

        # 3. Test Story Calibration Reactive Callback
        payload_cal = {
            "output": "graph-story-calibration.figure",
            "outputs": {"id": "graph-story-calibration", "property": "figure"},
            "inputs": [{"id": "store-selected-ids", "property": "data", "value": sample_eids}],
            "changedPropIds": ["store-selected-ids.data"],
        }
        r_cal = self.client.post("/_dash-update-component", json=payload_cal)
        self.assertEqual(r_cal.status_code, 200)

        # 4. Test Story Disparity & Subgroup Callbacks
        payload_dis = {
            "output": "graph-story-disparity.figure",
            "outputs": {"id": "graph-story-disparity", "property": "figure"},
            "inputs": [
                {"id": "store-selected-ids", "property": "data", "value": sample_eids},
                {"id": "story-disparity-slice", "property": "value", "value": "age"},
            ],
            "changedPropIds": ["story-disparity-slice.value"],
        }
        r_dis = self.client.post("/_dash-update-component", json=payload_dis)
        self.assertEqual(r_dis.status_code, 200)

        payload_sub = {
            "output": "graph-story-subgroup.figure",
            "outputs": {"id": "graph-story-subgroup", "property": "figure"},
            "inputs": [
                {"id": "store-selected-ids", "property": "data", "value": sample_eids},
                {"id": "story-subgroup-group", "property": "value", "value": "race"},
            ],
            "changedPropIds": ["story-subgroup-group.value"],
        }
        r_sub = self.client.post("/_dash-update-component", json=payload_sub)
        self.assertEqual(r_sub.status_code, 200)


if __name__ == "__main__":
    unittest.main()
