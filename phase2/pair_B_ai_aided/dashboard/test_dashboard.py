"""Checks for the Phase II dashboard: the live models must reproduce Phase I, and every view must render.
Run from phase2/pair_B_ai_aided:  python -m pytest dashboard/test_dashboard.py -q
(needs the gitignored outputs; Model B tests load Bio_ClinicalBERT, ~30 s on CPU)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures as F  # noqa: E402
from data_layer import EDITABLE, GROUPS, WALKTHROUGH_STAY, load  # noqa: E402


@pytest.fixture(scope="module")
def d():
    return load()


def test_model_a_reproduces_phase1(d):
    for row in np.random.default_rng(0).choice(len(d.df), 50, replace=False):
        live, _ = d.knn("A", d.xa[row], row)
        p1, _ = d.phase1_neighbours("A", row)
        assert set(live) == set(p1)


def test_model_b_mostly_reproduces_phase1(d):
    rows = np.random.default_rng(1).choice(len(d.df), 50, replace=False)
    shared = [len(set(d.knn("B", d.xb[r], r)[0]) & set(d.phase1_neighbours("B", r)[0])) for r in rows]
    assert np.mean(shared) >= 9.5                       # float16 cache: near-ties may swap (notebook: 9.97)


def test_decomposition_is_exact(d):
    row = d.pos[WALKTHROUGH_STAY]
    nb, dist = d.phase1_neighbours("A", row)
    dec = d.decomposition_a(row, nb)
    assert np.allclose(dec.sum(axis=1).to_numpy(), 1 - dist, atol=1e-4)


def test_what_if_without_change_keeps_all(d):
    row = d.pos[WALKTHROUGH_STAY]
    res = d.what_if(row, {})
    assert res["changed"] == [] and res["A"]["kept"] == 10 and res["B"]["kept"] == 10


def test_what_if_hba1c_edit(d):
    row = d.pos[WALKTHROUGH_STAY]
    res = d.what_if(row, {"A1C": ">8"})
    assert res["changed"] == ["A1C"]
    assert 0 <= res["A"]["kept"] <= 10 and 0 <= res["B"]["kept"] <= 10


def test_field_swap_model_a_closes_the_gap(d):
    row = d.pos[WALKTHROUGH_STAY]
    other = int(d.phase1_neighbours("A", row)[0][0])
    sw = d.field_swap("A", row, other)
    assert (sw.cost >= -1e-6).all() or sw.empty


def test_all_views_render(d):
    row = d.pos[WALKTHROUGH_STAY]
    for space in d.maps:
        for color in F.COLOR_BY:
            assert F.map_figure(d, space, color, "all", row, None).data
    for g in GROUPS:
        assert F.groups_figure(d, g).data
        assert d.groups[g].any()
    assert F.match_figure(d, row).data and F.vote_figure(d, row).data and F.decomposition_figure(d, row).data
    for m in ("shapley", "variance"):
        assert F.weights_figure(d, m, ["A1C"]).data
    assert set(EDITABLE) <= set(d.categories)
