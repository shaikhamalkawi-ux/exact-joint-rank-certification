"""Core numerical reproduction for the R11 EJRC manuscript.

This lightweight public script reproduces the headline numerical checks used
by the reusable EJRC package. The larger manuscript audit archive contains
additional endpoint-enumeration and sampling diagnostics.
"""

import numpy as np

from ejrc import einstein_scores, gamma_star, shortlist_certificate, weight_only_radius

PV_X = np.array([
    [0.780, 0.650, 0.600, 0.978, 0.800],
    [0.920, 0.800, 0.720, 0.968, 0.700],
    [0.880, 0.880, 0.850, 0.989, 0.650],
])
PV_W = np.array([0.1732, 0.2284, 0.2576, 0.2068, 0.1340])
PV_NAMES = np.array(["PERC", "TOPCon", "HJT"])

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
LOG_NAMES = np.array(list("ABCDEF"))

print("PV positive Einstein scores")
pv_scores = einstein_scores(PV_X, PV_W)
for name, score in zip(PV_NAMES, pv_scores):
    print(f"{name}: {score:.12f}")

print("\nPV weight-only first-tie boundaries")
for i, k in [(2, 1), (1, 0), (2, 0)]:
    print(f"{PV_NAMES[i]}>{PV_NAMES[k]}: {weight_only_radius(PV_X, PV_W, i, k):.12f}")

print("\nLogistics positive Einstein scores and order")
log_scores = einstein_scores(LOG_X, LOG_W)
order = np.argsort(log_scores)[::-1]
for i in order:
    print(f"{LOG_NAMES[i]}: {log_scores[i]:.12f}")
print("order:", ">".join(LOG_NAMES[order]))

winner_r = min(weight_only_radius(LOG_X, LOG_W, 0, j) for j in range(1, 6))
print(f"winner-A r*: {winner_r:.12f}")

r = 0.999
ql = np.full(5, 1-r)
qu = np.full(5, 1+r)
top3 = shortlist_certificate(LOG_W, ql, qu, LOG_X, LOG_X, shortlist=[0, 1, 4])
print(f"top-3 {{A,B,E}} preserved at r={r}: {top3.certified}")

d_log = np.arctanh(LOG_X[1]) - np.arctanh(LOG_X[4])
d_pv = np.arctanh(PV_X[1]) - np.arctanh(PV_X[0])
print(f"Gamma* logistics B>E at r=.20: {gamma_star(LOG_W, d_log, 0.20):.12f}")
print(f"Gamma* PV TOPCon>PERC at r=.53: {gamma_star(PV_W, d_pv, 0.53):.12f}")

assert np.allclose(pv_scores, [0.817449399904, 0.862257294933, 0.908244180698], atol=2e-12)
assert "".join(LOG_NAMES[order]) == "ABEDFC"
assert abs(winner_r - 0.48470298096753306) < 2e-12
assert top3.certified
assert abs(gamma_star(LOG_W, d_log, 0.20) - 2.580335731414865) < 2e-12
assert abs(gamma_star(PV_W, d_pv, 0.53) - 4.8435526626717245) < 2e-12

print("\nR11 CORE ASSERTIONS PASS")
