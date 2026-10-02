# EJRC — Exact Joint Rank Certification

**Exact Joint Rank Certification (EJRC)** is a compact Python implementation of exact decision-specific robustness certificates for **positive Einstein aggregation under simultaneous bounded uncertainty in criterion weights and normalized scores**.

The central distinction is:

> **Observed ranking stability is not the same as universal certification, and preserving a winner, a shortlist, and a complete ranking are mathematically different robustness requirements.**

This repository is method-first. It provides reusable pairwise, winner, shortlist, and budgeted coordination certificates, together with the two published-data demonstrations used in the associated manuscript.

## What EJRC provides

- **Pairwise exact certificate** over criterion-specific asymmetric score and weight boxes.
- **Winner certificate**: whether one alternative remains first for every admissible perturbation.
- **Shortlist certificate**: whether a selected set remains above every alternative outside the set, without requiring a fixed internal order.
- **Budgeted certificate**: a Bertsimas–Sim-style coordination budget for weight perturbations.
- **Constructive adverse direction**: criterion-wise multiplier choices that minimize the pairwise transformed margin.
- **Positive Einstein scores** using the additive hyperbolic generator.

For a pair of alternatives (i) and (k), let the reference weights be (v), the multiplier intervals be

```text
q_j in [q_lower_j, q_upper_j]
```

and the score intervals be

```text
x_ij in [L_ij, U_ij],   x_kj in [L_kj, U_kj].
```

The worst-case transformed difference is

```text
Delta_wc_ikj = atanh(L_ij) - atanh(U_kj).
```

The exact box certificate is

```text
C_box_ik(v) = sum_j v_j * q_adv_j * Delta_wc_ikj,
```

where `q_adv_j = q_lower_j` when `Delta_wc_ikj >= 0`, and `q_upper_j` otherwise. Strict pairwise preservation holds for every admissible perturbation **if and only if** `C_box_ik(v) > 0`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[test]
pytest -q
```

### Minimal example

```python
import numpy as np
from ejrc import einstein_scores, symmetric_joint_certificate

X = np.array([
    [0.780, 0.650, 0.600, 0.978, 0.800],  # PERC
    [0.920, 0.800, 0.720, 0.968, 0.700],  # TOPCon
    [0.880, 0.880, 0.850, 0.989, 0.650],  # HJT
])
w = np.array([0.1732, 0.2284, 0.2576, 0.2068, 0.1340])

print(einstein_scores(X, w))

cert = symmetric_joint_certificate(
    X, w, leader=2, competitor=1, r=0.20, delta=0.005
)
print(cert.value, cert.certified)
```

## Reproduce the paper examples

```bash
python examples/pv_example.py
python examples/logistics_example.py
python scripts/reproduce_r11.py
```

The examples reproduce the locked manuscript values, including:

- PV positive Einstein scores: `0.817449399904, 0.862257294933, 0.908244180698`.
- PV weight-only tie boundaries: `0.686066708970, 0.521244053506, 0.809552612318`.
- Logistics order: `A > B > E > D > F > C`.
- Logistics controlling adjacent weight-only boundary: `B > E`, `r* = 0.178772157371`.
- Logistics unique-winner boundary for `A`: `r* = 0.484702980968`.
- Logistics top-three set `{A,B,E}` remains separated from `{D,F,C}` for every `0 <= r < 1` under the stated weight-only model.
- Budgeted first-tie thresholds: `Gamma* = 2.580335731415` for logistics `B > E` at `r=0.20`, and `Gamma* = 4.843552662672` for PV `TOPCon > PERC` at `r=0.53`.

## Repository structure

```text
src/ejrc/                 Reusable EJRC implementation
examples/                 PV and logistics demonstrations
data/                     Published/derived inputs used by the examples
scripts/reproduce_r11.py  Full manuscript reproduction script
tests/                    Numerical regression tests
docs/                     Provenance and release notes
```

## Scope and interpretation

EJRC is a **decision-certification method**, not a claim that positive Einstein aggregation is universally superior to other MCDM operators. The two examples demonstrate the certificate on published data. They do not constitute new PV field measurements, new logistics observations, or universal technology/supplier rankings.

Scores are assumed to be in `[0,1)`. The endpoint `0` is valid because `atanh(0)=0`; score endpoint `1` requires limiting or regularized treatment.

## Citation and release status

This repository is public. The software citation metadata currently uses placeholder author information until the final manuscript author metadata and venue policy are confirmed. Before creating a permanent Zenodo DOI, update `CITATION.cff` with the final author list and paper DOI/URL when available.

## License

Code is released under the MIT License. Source-data citations remain governed by their original publications; this repository includes only the numerical values required to reproduce the reported demonstrations and their provenance.
