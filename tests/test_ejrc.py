import numpy as np

from ejrc import (
    budgeted_weight_certificate,
    einstein_scores,
    gamma_star,
    pairwise_box_certificate,
    shortlist_certificate,
    weight_only_radius,
)

PV_X = np.array([
    [0.780, 0.650, 0.600, 0.978, 0.800],
    [0.920, 0.800, 0.720, 0.968, 0.700],
    [0.880, 0.880, 0.850, 0.989, 0.650],
])
PV_W = np.array([0.1732, 0.2284, 0.2576, 0.2068, 0.1340])

LOG_X = np.array([
    [0.5241, 0.5241, 0.5241, 0.3276, 0.5241],
    [0.3931, 0.4586, 0.4586, 0.3931, 0.3931],
    [0.3931, 0.3931, 0.3276, 0.2620, 0.3276],
    [0.3931, 0.3931, 0.3931, 0.2620, 0.3931],
    [0.3931, 0.3931, 0.3931, 0.4586, 0.4586],
    [0.3276, 0.3276, 0.3276, 0.3276, 0.3931],
])
LOG_W_RAW = np.array([0.0749, 0.0457, 0.4996, 0.2548, 0.1251])
LOG_W = LOG_W_RAW / LOG_W_RAW.sum()


def test_pv_scores_and_weight_boundaries():
    scores = einstein_scores(PV_X, PV_W)
    assert np.allclose(scores, [0.817449399904, 0.862257294933, 0.908244180698], atol=2e-12)
    assert abs(weight_only_radius(PV_X, PV_W, 2, 1) - 0.6860667089703324) < 2e-12
    assert abs(weight_only_radius(PV_X, PV_W, 1, 0) - 0.5212440535060456) < 2e-12
    assert abs(weight_only_radius(PV_X, PV_W, 2, 0) - 0.8095526123177911) < 2e-12


def test_logistics_scores_order_and_winner_boundary():
    scores = einstein_scores(LOG_X, LOG_W)
    order = np.argsort(scores)[::-1]
    assert ''.join(np.array(list('ABCDEF'))[order]) == 'ABEDFC'
    assert abs(weight_only_radius(LOG_X, LOG_W, 1, 4) - 0.17877215737137908) < 2e-12
    winner_r = min(weight_only_radius(LOG_X, LOG_W, 0, j) for j in range(1, 6))
    assert abs(winner_r - 0.48470298096753306) < 2e-12


def test_logistics_top3_set_certificate_near_r_one():
    r = 0.999
    ql = np.full(5, 1-r)
    qu = np.full(5, 1+r)
    cert = shortlist_certificate(LOG_W, ql, qu, LOG_X, LOG_X, shortlist=[0, 1, 4])
    assert cert.certified


def test_general_asymmetric_box_formula_matches_manual_choice():
    v = np.array([0.2, 0.3, 0.5])
    ql = np.array([0.8, 0.7, 0.9])
    qu = np.array([1.3, 1.2, 1.1])
    Li = np.array([0.50, 0.30, 0.80])
    Uk = np.array([0.40, 0.45, 0.70])
    out = pairwise_box_certificate(v, ql, qu, Li, Uk)
    d = np.arctanh(Li) - np.arctanh(Uk)
    expected = np.sum(v * np.where(d >= 0, ql, qu) * d)
    assert abs(out.value - expected) < 1e-15


def test_budgeted_gamma_roots():
    d_log = np.arctanh(LOG_X[1]) - np.arctanh(LOG_X[4])
    g_log = gamma_star(LOG_W, d_log, 0.20)
    assert abs(g_log - 2.580335731414865) < 2e-12
    assert abs(budgeted_weight_certificate(LOG_W, d_log, 0.20, g_log)) < 2e-12

    d_pv = np.arctanh(PV_X[1]) - np.arctanh(PV_X[0])
    g_pv = gamma_star(PV_W, d_pv, 0.53)
    assert abs(g_pv - 4.8435526626717245) < 2e-12
    assert abs(budgeted_weight_certificate(PV_W, d_pv, 0.53, g_pv)) < 2e-12
