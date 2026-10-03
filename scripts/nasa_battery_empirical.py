#!/usr/bin/env python3
"""Direct-from-raw NASA PCoE empirical EJRC extension.

Primary source:
  NASA Prognostics Center of Excellence Battery Data Set
  https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

This script downloads the official NASA archive, extracts B0006 and B0018,
parses discharge-cycle measurements directly from the MATLAB files, and builds
a measurement-derived EJRC score-uncertainty audit.
"""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import rankdata

NASA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
BATTERIES = ("B0006", "B0018")
EXPECTED_GIT_BLOB_SHA1 = {
    "B0006": "8867cf00c30a521848d83b2d31be3e162710e7ad",
    "B0018": "13ef3ffda5be655edc2ed08a52073aedf73b548b",
}
EXPECTED_DISCHARGE_COUNTS = {"B0006": 168, "B0018": 132}
CRITERIA = (
    ("capacity_Ah", True),
    ("avg_voltage_V", True),
    ("avg_temperature_C", False),
    ("current_tracking_error_A", False),
)
WEIGHTS = np.full(len(CRITERIA), 1.0 / len(CRITERIA))
STAGES = (("early", 0.20), ("mid", 0.50), ("late", 0.80))
HALF_WINDOW = 0.05
OUTDIR = Path(os.environ.get("EJRC_NASA_OUTDIR", "nasa_empirical_output"))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()

def expand_nested_zips(root: Path, max_depth: int = 6) -> None:
    """Recursively expand ZIP archives contained in the official bundle."""
    processed = set()
    for depth in range(max_depth):
        pending = [p for p in root.rglob("*.zip") if p.resolve() not in processed]
        if not pending:
            return
        print(f"nested_zip_depth={depth} pending={len(pending)}")
        for zpath in pending:
            processed.add(zpath.resolve())
            dest = zpath.parent / (zpath.stem + "_extracted")
            dest.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(dest)
                print(f"expanded_nested_zip={zpath.relative_to(root)}")
            except zipfile.BadZipFile:
                print(f"WARNING bad_nested_zip={zpath.relative_to(root)}")
    remaining = [p for p in root.rglob("*.zip") if p.resolve() not in processed]
    if remaining:
        raise RuntimeError(f"Nested ZIP expansion depth exceeded; remaining={len(remaining)}")

def find_member(root: Path, basename: str) -> Path:
    hits = list(root.rglob(basename))
    if len(hits) != 1:
        candidates = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
        print("archive_file_inventory_sample=" + json.dumps(candidates[:200]))
        raise RuntimeError(f"Expected exactly one {basename}, found {len(hits)}")
    return hits[0]

def _to_float_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).squeeze()
    return np.atleast_1d(a).astype(float)

def parse_battery_mat(path: Path, battery_id: str) -> pd.DataFrame:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if battery_id not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        if len(keys) != 1:
            raise KeyError(f"Could not identify root variable for {battery_id}: {keys}")
        root = mat[keys[0]]
    else:
        root = mat[battery_id]
    cycles = np.atleast_1d(root.cycle)
    records = []
    discharge_index = 0
    for raw_idx, cyc in enumerate(cycles, start=1):
        typ = str(cyc.type).strip().lower()
        if typ != "discharge":
            continue
        discharge_index += 1
        d = cyc.data
        v = _to_float_array(d.Voltage_measured)
        cur = _to_float_array(d.Current_measured)
        temp = _to_float_array(d.Temperature_measured)
        cap = float(np.asarray(d.Capacity, dtype=float).squeeze())
        ambient = float(np.asarray(cyc.ambient_temperature, dtype=float).squeeze())
        n = min(len(v), len(cur), len(temp))
        if n == 0:
            raise RuntimeError(f"Empty discharge record {battery_id} raw cycle {raw_idx}")
        v, cur, temp = v[:n], cur[:n], temp[:n]
        records.append({
            "battery": battery_id,
            "raw_cycle_index": raw_idx,
            "discharge_index": discharge_index,
            "ambient_temperature_C": ambient,
            "capacity_Ah": cap,
            "avg_voltage_V": float(np.mean(v)),
            "avg_current_A": float(np.mean(cur)),
            "avg_temperature_C": float(np.mean(temp)),
            "max_temperature_C": float(np.max(temp)),
            "voltage_sd_V": float(np.std(v, ddof=0)),
            "current_tracking_error_A": float(abs(abs(np.mean(cur)) - 2.0)),
            "n_samples": int(n),
        })
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No discharge records parsed for {battery_id}")
    df["life_fraction"] = np.arange(len(df), dtype=float) / max(len(df) - 1, 1)
    return df

def add_empirical_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    for col, benefit in CRITERIA:
        ranks = rankdata(out[col].to_numpy(float), method="average")
        s = ranks / (n + 1.0)
        if not benefit:
            s = 1.0 - s
        out[f"score_{col}"] = s
        assert np.all((s > 0) & (s < 1))
    return out

def einstein_score(scores: np.ndarray) -> float:
    return float(np.tanh(np.dot(WEIGHTS, np.arctanh(np.asarray(scores, dtype=float)))))

def interval_certificate(leader_lo: np.ndarray, competitor_hi: np.ndarray) -> float:
    d = np.arctanh(leader_lo) - np.arctanh(competitor_hi)
    return float(np.dot(WEIGHTS, d))

def stage_window(df: pd.DataFrame, battery: str, center: float) -> pd.DataFrame:
    g = df[df["battery"] == battery]
    w = g[(g["life_fraction"] >= center - HALF_WINDOW - 1e-12)
          & (g["life_fraction"] <= center + HALF_WINDOW + 1e-12)].copy()
    if len(w) < 5:
        raise RuntimeError(f"Too few observations for {battery} stage {center}: {len(w)}")
    return w

def build_stage_outputs(df: pd.DataFrame):
    score_cols = [f"score_{c}" for c, _ in CRITERIA]
    rows, intervals = [], []
    for stage, center in STAGES:
        cells = {}
        for b in BATTERIES:
            w = stage_window(df, b, center)
            nominal = w[score_cols].median().to_numpy(float)
            lo = w[score_cols].min().to_numpy(float)
            hi = w[score_cols].max().to_numpy(float)
            raw_medians = {c: float(w[c].median()) for c, _ in CRITERIA}
            cells[b] = dict(window=w, nominal=nominal, lo=lo, hi=hi,
                            score=einstein_score(nominal), raw=raw_medians)
            for j, ((criterion, _), s_nom, s_lo, s_hi) in enumerate(
                zip(CRITERIA, nominal, lo, hi), start=1
            ):
                intervals.append({
                    "stage": stage, "stage_center": center, "battery": b,
                    "criterion": criterion, "criterion_index": j,
                    "score_median": s_nom, "score_min": s_lo, "score_max": s_hi,
                    "n_cycles": len(w),
                    "life_fraction_min": float(w["life_fraction"].min()),
                    "life_fraction_max": float(w["life_fraction"].max()),
                })
        b6, b18 = cells["B0006"], cells["B0018"]
        cert_6_18 = interval_certificate(b6["lo"], b18["hi"])
        cert_18_6 = interval_certificate(b18["lo"], b6["hi"])
        winner = "B0006" if b6["score"] > b18["score"] else "B0018"
        loser = "B0018" if winner == "B0006" else "B0006"
        cert_winner = cert_6_18 if winner == "B0006" else cert_18_6
        rows.append({
            "stage": stage, "stage_center": center,
            "B0006_n_cycles": len(b6["window"]),
            "B0018_n_cycles": len(b18["window"]),
            "B0006_Einstein": b6["score"], "B0018_Einstein": b18["score"],
            "nominal_winner": winner, "nominal_loser": loser,
            "nominal_score_gap": abs(b6["score"] - b18["score"]),
            "C_B0006_over_B0018": cert_6_18,
            "C_B0018_over_B0006": cert_18_6,
            "nominal_winner_empirical_box_certified": bool(cert_winner > 0),
            "controlling_winner_certificate": cert_winner,
            **{f"B0006_median_{k}": v for k, v in b6["raw"].items()},
            **{f"B0018_median_{k}": v for k, v in b18["raw"].items()},
        })
    return pd.DataFrame(rows), pd.DataFrame(intervals)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        archive = td / "Battery_Data_Set.zip"
        print(f"Downloading official NASA archive: {NASA_URL}", flush=True)
        urllib.request.urlretrieve(NASA_URL, archive)
        zip_sha = sha256_file(archive)
        print(f"archive_sha256={zip_sha}")
        extract = td / "extract"
        extract.mkdir()
        with zipfile.ZipFile(archive) as zf:
            print(f"top_level_members={len(zf.namelist())}")
            print("top_level_member_sample=" + json.dumps(zf.namelist()[:100]))
            zf.extractall(extract)
        expand_nested_zips(extract)

        source_manifest = {
            "official_url": NASA_URL,
            "archive_sha256": zip_sha,
            "selected_batteries": {},
            "selection_rationale":
                "B0006 and B0018 share the room-temperature, 2 A, 2.5 V cutoff "
                "protocol in the NASA README; B0005/B0007 use different cutoffs.",
        }
        frames = []
        for b in BATTERIES:
            p = find_member(extract, f"{b}.mat")
            sha256 = sha256_file(p)
            blob = git_blob_sha1(p)
            source_manifest["selected_batteries"][b] = {
                "archive_member": str(p.relative_to(extract)),
                "size_bytes": p.stat().st_size,
                "sha256": sha256,
                "git_blob_sha1": blob,
                "expected_public_mirror_git_blob_sha1": EXPECTED_GIT_BLOB_SHA1[b],
                "public_mirror_blob_match": blob == EXPECTED_GIT_BLOB_SHA1[b],
            }
            print(f"{b}: size={p.stat().st_size} sha256={sha256} "
                  f"git_blob_sha1={blob} mirror_match={blob == EXPECTED_GIT_BLOB_SHA1[b]}")
            dfb = parse_battery_mat(p, b)
            if len(dfb) != EXPECTED_DISCHARGE_COUNTS[b]:
                raise AssertionError(
                    f"{b}: expected {EXPECTED_DISCHARGE_COUNTS[b]} discharge cycles, got {len(dfb)}"
                )
            frames.append(dfb)

        cycles = add_empirical_scores(pd.concat(frames, ignore_index=True))
        stages, intervals = build_stage_outputs(cycles)

        cycles.to_csv(OUTDIR / "nasa_b0006_b0018_cycle_summary.csv", index=False)
        stages.to_csv(OUTDIR / "nasa_ejrc_stage_results.csv", index=False)
        intervals.to_csv(OUTDIR / "nasa_ejrc_empirical_intervals.csv", index=False)
        (OUTDIR / "nasa_source_manifest.json").write_text(
            json.dumps(source_manifest, indent=2), encoding="utf-8"
        )
        readme = """# NASA Battery empirical EJRC audit

Primary data source: NASA PCoE Battery Data Set (Saha & Goebel, 2007).

This output is generated directly from the official NASA archive by the
companion extraction script.

B0006 and B0018 are compared because the NASA README describes both under the
same room-temperature 2 A discharge protocol with a 2.5 V cutoff. This audit
does not compare B0005/B0007 as if all four cells shared the same discharge
boundary.

Observed discharge cycles provide four measured criteria: capacity, average
voltage, average temperature, and absolute mean-current tracking error from
2 A. Pooled empirical average-rank scores map these measurements monotonically
into (0,1). Equal criterion weights are a neutral reference choice, not a
measurement-derived preference model.

Early/mid/late stage windows are centered at 20/50/80 percent of each cell's
observed discharge trajectory with +/-5 percentage-point windows. Median scores
are nominal values; observed min/max score ranges define empirical interval
bounds. The EJRC box certificate treats criterion ranges independently, so the
box is conservative and can include combinations not observed in a single
cycle.

These ranges are empirical observed ranges, not confidence intervals.
"""
        (OUTDIR / "README.md").write_text(readme, encoding="utf-8")

        print("\n=== EMPIRICAL STAGE RESULTS ===")
        print(stages.to_csv(index=False))
        print("=== SOURCE MANIFEST ===")
        print(json.dumps(source_manifest, indent=2))

        st = stages.set_index("stage")
        assert st.loc["early", "nominal_winner"] == "B0006"
        assert not bool(st.loc["early", "nominal_winner_empirical_box_certified"])
        assert st.loc["mid", "nominal_winner"] == "B0018"
        assert bool(st.loc["mid", "nominal_winner_empirical_box_certified"])
        assert st.loc["late", "nominal_winner"] == "B0018"
        assert bool(st.loc["late", "nominal_winner_empirical_box_certified"])
        print("NASA EMPIRICAL EJRC ASSERTIONS PASS")

if __name__ == "__main__":
    main()
