#!/usr/bin/env python3
"""Validate the bridge FEA mesh-requirements corpus with the standard library."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "data" / "bridge_fea_mesh_cases"
EXPECTED_FIELDS = (
    "id",
    "evidence_tier",
    "source_type",
    "publisher",
    "year",
    "component_family",
    "analysis_context",
    "engineering_qoi",
    "mesh_need_class",
    "mesh_question",
    "mesh_attention",
    "ai_intent_signal",
    "source_title",
    "url",
)
EXPECTED_TIERS = {"T1": 37, "T2": 48, "T3": 5, "T4": 10}


def main() -> None:
    rows: list[dict[str, str]] = []
    files = sorted(CASE_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No corpus CSV files found in {CASE_DIR}")

    for path in files:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
                raise SystemExit(f"Unexpected fields in {path}: {reader.fieldnames}")
            rows.extend(dict(row) for row in reader)

    ids = [row["id"] for row in rows]
    duplicates = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate catalog IDs: {duplicates}")

    tiers = Counter(row["evidence_tier"] for row in rows)
    if dict(tiers) != EXPECTED_TIERS:
        raise SystemExit(f"Unexpected evidence-tier counts: {dict(tiers)}")

    bad_urls = [row["id"] for row in rows if not row["url"].startswith(("https://", "http://"))]
    if bad_urls:
        raise SystemExit(f"Rows with invalid URLs: {bad_urls}")

    for row in rows:
        for field in EXPECTED_FIELDS:
            if not row[field].strip():
                raise SystemExit(f"Blank field {field!r} in row {row['id']}")

    json.loads((ROOT / "data" / "bridge_mesh_question_templates.json").read_text(encoding="utf-8"))
    json.loads((ROOT / "schemas" / "bridge_mesh_intent.schema.json").read_text(encoding="utf-8"))

    print(f"validated_rows={len(rows)}")
    print(f"evidence_tiers={dict(tiers)}")
    print(f"catalog_files={[path.name for path in files]}")


if __name__ == "__main__":
    main()
