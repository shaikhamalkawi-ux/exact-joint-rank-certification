import numpy as np

from ejrc import (
    einstein_scores,
    gamma_star,
    shortlist_certificate,
    weight_only_radius,
)

X = np.array([
    [0.780, 0.650, 0.600, 0.978, 0.800],  # PERC
    [0.920, 0.800, 0.720, 0.968, 0.700],  # TOPCon
    [0.880, 0.880, 0.850, 0.989, 0.650],  # HJT
])
w = np.array([0.1732, 0.2284, 0.2576, 0.2068, 0.1340])
names = ["PERC", "TOPCon", "HJT"]

scores = einstein_scores(X, w)
print("Positive Einstein scores")
for name, score in zip(names, scores):
    print(f"  {name}: {score:.12f}")

pairs = [(2, 1), (1, 0), (2, 0)]
print("\nWeight-only first-tie boundaries")
for i, k in pairs:
    print(f"  {names[i]} > {names[k]}: r* = {weight_only_radius(X, w, i, k):.12f}")

r = 0.60
q_lo = np.full(5, 1-r)
q_hi = np.full(5, 1+r)
set_cert = shortlist_certificate(w, q_lo, q_hi, X, X, shortlist=[2])
print(f"\nHJT-winner certificate at r={r:.2f}: {set_cert.value:.12f} -> {set_cert.certified}")

d = np.arctanh(X[1]) - np.arctanh(X[0])
print(f"Budgeted Gamma* for TOPCon>PERC at r=.53: {gamma_star(w, d, 0.53):.12f}")
