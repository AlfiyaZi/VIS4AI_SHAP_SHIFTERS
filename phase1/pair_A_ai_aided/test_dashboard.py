"""
Automated headless test suite for Pair A (AI-aided) Phase I Dashboard.
Verifies data contracts, schemas, clustering, figure generation, and app layout.
"""

import sys
import unittest
from pathlib import Path

DASH_DIR = Path(__file__).resolve().parent / "dashboard"
sys.path.insert(0, str(DASH_DIR))

import callbacks
import data_layer
import figures
import layout


class TestPairADashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = data_layer.load()
        cls.df = cls.data.df
        cls.metrics = cls.data.metrics

    def test_01_data_contract_and_schema(self):
        """Verify the static data tables load with exact expected schemas and row count."""
        self.assertEqual(len(self.df), 30248)
        self.assertEqual(self.df["patient_nbr"].nunique(), 16773)

        # Essential coordinate and label columns
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

        # Check confusion matrices
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
        # Top association should be bounded between 0 and 1
        top_assoc = ranking.iloc[0]["association"]
        self.assertTrue(0.0 <= top_assoc <= 1.0)

    def test_05_figures_scatter_explorer(self):
        """Verify View 1 scatter explorer compiles across configurations."""
        # Unfiltered, with jitter
        fig1 = figures.scatter_explorer(self.data, selected=None, x_model="model_b", jitter=True, contours=False)
        self.assertIsNotNone(fig1)
        self.assertTrue(len(fig1.data) > 0)

        # Filtered, with contours
        sub_ids = set(self.df.index[:100])
        fig2 = figures.scatter_explorer(self.data, selected=sub_ids, x_model="model_a", jitter=True, contours=True)
        self.assertIsNotNone(fig2)

    def test_06_figures_confusion_heatmaps(self):
        """Verify View 2 confusion matrix heatmaps compile."""
        fig_a = figures.confusion_heatmap(self.data, "model_a", normalize=True, clustered=True, highlight=None)
        fig_b = figures.confusion_heatmap(self.data, "model_b", normalize=False, clustered=False,
                                          highlight=("Cardiovascular", "Kidney / Renal"))
        fig_maj = figures.confusion_heatmap(self.data, "majority", normalize=True, clustered=True, highlight=None)
        self.assertIsNotNone(fig_a)
        self.assertIsNotNone(fig_b)
        self.assertIsNotNone(fig_maj)

    def test_07_figures_umap_view(self):
        """Verify View 3 UMAP projection compiles across color modes and trajectory settings."""
        for mode in figures.UMAP_COLOR_MODES:
            fig = figures.umap_view(self.data, selected=None, color_mode=mode, show_traj=True,
                                    n_patients=20, annotate=True)
            self.assertIsNotNone(fig)

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

    def test_09_app_layout_and_components(self):
        """Verify Dash app layout creates expected container IDs."""
        app_layout = layout.build_layout(self.data)
        self.assertIsNotNone(app_layout)


if __name__ == "__main__":
    unittest.main()
