# Reproducibility

## Environment

Recommended:

- Python 3.10+
- NumPy 1.24+
- SciPy 1.10+ for the independent linear-programming budget check
- pytest 7.4+

Install:

```bash
pip install -e .[test]
```

Run regression tests:

```bash
pytest -q
```

Run examples:

```bash
python examples/pv_example.py
python examples/logistics_example.py
```

Run the full manuscript audit:

```bash
python scripts/reproduce_r11.py
```

The full reproduction script uses the locked seeds documented in the manuscript:

- PV diagnostic sampling: `20261001`
- General asymmetric-box audit: `20261002`
- Logistics sampling diagnostic: `20261004`

The manuscript's analytical exactness refers to the closed-form certificate. Numerical roots, random audits, and floating-point implementation comparisons are finite-precision checks.
