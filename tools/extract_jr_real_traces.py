#!/usr/bin/env python3
"""Extract exact r=10 arrival permutations from JR-DIGITAL/ooo-dataset.

This script is intentionally dependency-free.  It downloads only the nine
recorded WLAN/UMTS sessions fixed by the SCLS practical-application protocol,
verifies their Git blob SHA-1 values, and emits all admitted blocks (ordered
and reordered) plus audit summaries.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import pathlib
import statistics
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

COMMIT = "812f45a59faf9df36f1254d0b3fa9ed7d461fa1a"
REPO = "JR-DIGITAL/ooo-dataset"
R = 10
SESSIONS: Sequence[Tuple[str, str, str, str]] = (
    ("D-1", "UMTS", "dataset-d/d-1.csv", "66a4c8561393817235cfb7967d5ecb61d4c08e5b"),
    ("D-2", "UMTS", "dataset-d/d-2.csv", "cf8f114f3382ad0b3ef2a4567d5cb789ac8e33c3"),
    ("D-3", "UMTS", "dataset-d/d-3.csv", "9221ff491697188918fef932e335db1f07e66cf1"),
    ("D-4", "UMTS", "dataset-d/d-4.csv", "64f4864aac51f0cfa06a96d50a769301333ad35c"),
    ("D-5", "UMTS", "dataset-d/d-5.csv", "c06481234b43e8de47c291b14cf77e89c396d9d8"),
    ("S-7", "WLAN", "dataset-s/s-7.csv", "1f6a533508f5e13af62c26d9072764c931d8ef11"),
    ("S-8", "WLAN", "dataset-s/s-8.csv", "887ae058920cc5779458c5a1dcd9f98ef11e5a81"),
    ("S-9", "WLAN", "dataset-s/s-9.csv", "8f1749e7747682cde7d127450b0af29248814d93"),
    ("S-10", "WLAN", "dataset-s/s-10.csv", "149b37d5fffe9063ae0a8c15b10b98a482e5ffd9"),
)


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def inversion_count(p: Sequence[int]) -> int:
    return sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))


def inverse_permutation(p: Sequence[int]) -> List[int]:
    inv = [0] * len(p)
    for arrival_pos, transmitted_pos in enumerate(p):
        inv[transmitted_pos] = arrival_pos
    return inv


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    pos = q * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def summarize_blocks(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "admitted_blocks": 0,
            "reordered_blocks": 0,
            "reordered_fraction": None,
        }
    moved = [float(r["moved_position_fraction"]) for r in rows]
    invr = [float(r["inversion_rate"]) for r in rows]
    maxd = [float(r["maximum_displacement"]) for r in rows]
    reordered = sum(int(r["reordered"]) for r in rows)
    return {
        "admitted_blocks": len(rows),
        "ordered_blocks": len(rows) - reordered,
        "reordered_blocks": reordered,
        "reordered_fraction": reordered / len(rows),
        "unique_permutations": len({r["permutation_arrival_to_transmitted"] for r in rows}),
        "mean_moved_position_fraction": statistics.fmean(moved),
        "median_moved_position_fraction": statistics.median(moved),
        "p90_moved_position_fraction": quantile(moved, 0.90),
        "mean_inversion_rate": statistics.fmean(invr),
        "median_inversion_rate": statistics.median(invr),
        "p90_inversion_rate": quantile(invr, 0.90),
        "mean_maximum_displacement": statistics.fmean(maxd),
        "median_maximum_displacement": statistics.median(maxd),
        "p90_maximum_displacement": quantile(maxd, 0.90),
    }


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SCLS-JR-real-trace-extractor/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> int:
    out_dir = pathlib.Path(os.environ.get("OUTPUT_DIR", "output"))
    raw_dir = out_dir / "raw_source_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_blocks: List[dict] = []
    session_audits: List[dict] = []
    source_manifest: List[dict] = []

    for session, network, relpath, expected_blob_sha1 in SESSIONS:
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{relpath}"
        data = download(url)
        actual_blob_sha1 = git_blob_sha1(data)
        if actual_blob_sha1 != expected_blob_sha1:
            raise RuntimeError(
                f"Git blob SHA-1 mismatch for {session}: "
                f"expected {expected_blob_sha1}, got {actual_blob_sha1}"
            )
        raw_sha256 = hashlib.sha256(data).hexdigest()
        raw_path = raw_dir / pathlib.Path(relpath).name
        raw_path.write_bytes(data)

        text = data.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text), delimiter=";", quotechar='"')
        rows_by_device: Dict[str, List[dict]] = defaultdict(list)
        total_events = 0
        for source_row_index, row in enumerate(reader, start=2):
            total_events += 1
            device = row["S.Device.ID"]
            seq = int(row["S.Message.ID"])
            receive_ms = int(row["S.Message.received.time.ms"])
            rows_by_device[device].append(
                {
                    "device": device,
                    "sequence_id": seq,
                    "server_receive_ms": receive_ms,
                    "source_row_index": source_row_index,
                }
            )

        candidate_blocks = 0
        excluded_missing = 0
        excluded_duplicate = 0
        timestamp_tie_blocks = 0
        session_blocks: List[dict] = []
        device_audit: List[dict] = []

        for device in sorted(rows_by_device):
            device_rows = rows_by_device[device]
            grouped: Dict[int, List[dict]] = defaultdict(list)
            for event in device_rows:
                grouped[event["sequence_id"] // R].append(event)

            admitted_device = 0
            reordered_device = 0
            excluded_missing_device = 0
            excluded_duplicate_device = 0
            for block_id in sorted(grouped):
                candidate_blocks += 1
                events = grouped[block_id]
                first_sequence = block_id * R
                counts = Counter(e["sequence_id"] for e in events)
                has_duplicate = any(v > 1 for v in counts.values())
                if has_duplicate:
                    excluded_duplicate += 1
                    excluded_duplicate_device += 1
                    continue
                expected = list(range(first_sequence, first_sequence + R))
                if len(events) != R or sorted(counts) != expected:
                    excluded_missing += 1
                    excluded_missing_device += 1
                    continue

                admitted_device += 1
                timestamp_counts = Counter(e["server_receive_ms"] for e in events)
                has_timestamp_tie = any(v > 1 for v in timestamp_counts.values())
                if has_timestamp_tie:
                    timestamp_tie_blocks += 1

                # Timestamp is the primary arrival-order key.  Source row index is
                # used only as a deterministic tie-breaker; every tie is recorded.
                arrival = sorted(
                    events,
                    key=lambda e: (e["server_receive_ms"], e["source_row_index"]),
                )
                p = [e["sequence_id"] - first_sequence for e in arrival]
                invp = inverse_permutation(p)
                moved = sum(int(v != j) for j, v in enumerate(p))
                inversions = inversion_count(p)
                max_displacement = max(abs(v - j) for j, v in enumerate(p))
                reordered = int(inversions > 0)
                reordered_device += reordered

                record = {
                    "trace_block_uid": f"{session}|{device}|{block_id}",
                    "network": network,
                    "session": session,
                    "device_id": device,
                    "block_id": block_id,
                    "first_sequence_id": first_sequence,
                    "last_sequence_id": first_sequence + R - 1,
                    "block_size": R,
                    "permutation_arrival_to_transmitted": ",".join(map(str, p)),
                    "inverse_permutation_transmitted_to_arrival": ",".join(map(str, invp)),
                    "arrival_sequence_ids": ",".join(str(e["sequence_id"]) for e in arrival),
                    "arrival_receive_ms": ",".join(str(e["server_receive_ms"]) for e in arrival),
                    "moved_position_count": moved,
                    "moved_position_fraction": moved / R,
                    "inversion_count": inversions,
                    "inversion_rate": inversions / (R * (R - 1) / 2),
                    "maximum_displacement": max_displacement,
                    "reordered": reordered,
                    "timestamp_tie": int(has_timestamp_tie),
                }
                all_blocks.append(record)
                session_blocks.append(record)

            device_audit.append(
                {
                    "device_id": device,
                    "events": len(device_rows),
                    "candidate_blocks": len(grouped),
                    "admitted_blocks": admitted_device,
                    "reordered_blocks": reordered_device,
                    "excluded_missing_blocks": excluded_missing_device,
                    "excluded_duplicate_blocks": excluded_duplicate_device,
                }
            )

        source_manifest.append(
            {
                "session": session,
                "network": network,
                "relative_path": relpath,
                "source_url": url,
                "locked_commit": COMMIT,
                "git_blob_sha1": actual_blob_sha1,
                "sha256": raw_sha256,
                "bytes": len(data),
            }
        )
        session_audits.append(
            {
                "session": session,
                "network": network,
                "events": total_events,
                "devices": len(rows_by_device),
                "candidate_blocks": candidate_blocks,
                "admitted_blocks": len(session_blocks),
                "excluded_missing_blocks": excluded_missing,
                "excluded_duplicate_blocks": excluded_duplicate,
                "timestamp_tie_blocks": timestamp_tie_blocks,
                **summarize_blocks(session_blocks),
                "devices_detail": device_audit,
            }
        )

    fieldnames = [
        "trace_block_uid",
        "network",
        "session",
        "device_id",
        "block_id",
        "first_sequence_id",
        "last_sequence_id",
        "block_size",
        "permutation_arrival_to_transmitted",
        "inverse_permutation_transmitted_to_arrival",
        "arrival_sequence_ids",
        "arrival_receive_ms",
        "moved_position_count",
        "moved_position_fraction",
        "inversion_count",
        "inversion_rate",
        "maximum_displacement",
        "reordered",
        "timestamp_tie",
    ]
    full_csv = out_dir / "jr_real_trace_blocks_r10.csv"
    with full_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_blocks)

    with full_csv.open("rb") as src, gzip.open(
        out_dir / "jr_real_trace_blocks_r10.csv.gz", "wb", compresslevel=9
    ) as dst:
        dst.write(src.read())

    reordered_csv = out_dir / "jr_real_reordered_blocks_r10.csv"
    with reordered_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([r for r in all_blocks if int(r["reordered"])])

    by_network = {
        network: summarize_blocks([r for r in all_blocks if r["network"] == network])
        for network in ("WLAN", "UMTS")
    }
    combined = summarize_blocks(all_blocks)
    audit = {
        "schema_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPO,
        "locked_commit": COMMIT,
        "block_size": R,
        "arrival_order_definition": "ascending S.Message.received.time.ms within each device and sequence-defined block; source row index is deterministic tie-breaker",
        "transmitted_order_definition": "ascending S.Message.ID within each device",
        "block_definition": "floor(S.Message.ID / 10), requiring exactly the consecutive IDs 10b,...,10b+9",
        "primary_sessions": [s[0] for s in SESSIONS],
        "simulated_G_sessions_used": False,
        "session_audits": session_audits,
        "network_summaries": by_network,
        "combined_summary": combined,
        "source_manifest": source_manifest,
    }
    (out_dir / "jr_trace_extraction_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # A small human-readable report is useful for independent review.
    report_lines = [
        "# JR-DIGITAL real-network trace extraction audit",
        "",
        f"Locked repository commit: `{COMMIT}`",
        "",
        "Only recorded WLAN S-7--S-10 and UMTS D-1--D-5 sessions are used. ",
        "Simulated G-* sessions are excluded from the primary extraction.",
        "",
        "## Network totals",
        "",
        "| Network | Admitted blocks | Reordered blocks | Reordered fraction | Unique permutations |",
        "|---|---:|---:|---:|---:|",
    ]
    for network in ("WLAN", "UMTS"):
        s = by_network[network]
        report_lines.append(
            f"| {network} | {s['admitted_blocks']} | {s['reordered_blocks']} | "
            f"{s['reordered_fraction']:.8f} | {s['unique_permutations']} |"
        )
    report_lines.extend(
        [
            "",
            "The extraction uses sequence ID for transmitted order and the server receive timestamp for arrival order. ",
            "Therefore the repository's broader A.ooo.Event.Sent.Time percentages are not substituted for the exact arrival permutations required by the replay protocol.",
            "",
        ]
    )
    (out_dir / "JR_TRACE_EXTRACTION_AUDIT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    # Manifest all emitted files except cached raw source rows, which are retained
    # only inside the short-lived workflow workspace and are not uploaded.
    manifest_rows = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            payload = path.read_bytes()
            manifest_rows.append(
                {
                    "path": path.name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    (out_dir / "EXTRACTION_OUTPUT_SHA256.json").write_text(
        json.dumps(manifest_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps({"network_summaries": by_network, "combined": combined}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
