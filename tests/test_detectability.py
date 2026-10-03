import numpy as np

from ejrc import sampling_detectability

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
LOG_W = np.array([0.0749, 0.0457, 0.4996, 0.2548, 0.1251])
LOG_W = LOG_W / LOG_W.sum()


def test_pv_sampling_detectability():
    out = sampling_detectability(PV_X, PV_W, 1, 0, 0.53, n_draws=100000)
    assert abs(out.probability - 1.493229890968507e-9) < 1e-21
    assert out.active_dimension == 5
    assert out.n_for_target_detection == 2006209687
    assert abs(out.miss_probability_n - 0.9998506881589143) < 1e-15


def test_logistics_sampling_detectability():
    out = sampling_detectability(LOG_X, LOG_W, 1, 4, 0.20, n_draws=100000)
    assert abs(out.probability - 3.327433448238726e-4) < 1e-16
    assert out.active_dimension == 4
    assert out.n_for_target_detection == 9002
