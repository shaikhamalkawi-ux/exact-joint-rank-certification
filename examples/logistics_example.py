import numpy as np

from ejrc import einstein_scores, gamma_star, shortlist_certificate, weight_only_radius

names = np.array(list("ABCDEF"))
X = np.array([
    [0.5241, 0.5241, 0.5241, 0.3276, 0.5241],
    [0.3931, 0.4586, 0.4586, 0.3931, 0.3931],
    [0.3931, 0.3931, 0.3276, 0.2620, 0.3276],
    [0.3931, 0.3931, 0.3931, 0.2620, 0.3931],
    [0.3931, 0.3931, 0.3931, 0.4586, 0.4586],
    [0.3276, 0.3276, 0.3276, 0.3276, 0.3931],
])
w_raw = np.array([0.0749, 0.0457, 0.4996, 0.2548, 0.1251])
w = w_raw / w_raw.sum()

scores = einstein_scores(X, w)
order = np.argsort(scores)[::-1]
print("Order:", ">".join(names[order]))
for i in order:
    print(f"  {names[i]}: {scores[i]:.12f}")

print("\nAdjacent weight-only first-tie boundaries")
for a, b in zip(order[:-1], order[1:]):
    print(f"  {names[a]}>{names[b]}: r*={weight_only_radius(X, w, a, b):.12f}")

r = 0.40
q_lo = np.full(5, 1-r)
q_hi = np.full(5, 1+r)
a_cert = shortlist_certificate(w, q_lo, q_hi, X, X, shortlist=[0])
pair = tuple(str(names[j]) for j in a_cert.controlling_pair)
print(f"\nA remains the unique winner at r={r:.2f}: {a_cert.certified}, controlling pair={pair}")

top3 = [0, 1, 4]  # A, B, E
r = 0.99
q_lo = np.full(5, 1-r)
q_hi = np.full(5, 1+r)
top3_cert = shortlist_certificate(w, q_lo, q_hi, X, X, shortlist=top3)
top3_names = '{' + ','.join(str(names[j]) for j in top3) + '}'
print(f"Top-three membership {top3_names} preserved at r={r:.2f}: {top3_cert.certified}")

d = np.arctanh(X[1]) - np.arctanh(X[4])
print(f"Budgeted Gamma* for B>E at r=.20: {gamma_star(w, d, 0.20):.12f}")
