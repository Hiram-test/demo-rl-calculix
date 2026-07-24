#!/usr/bin/env python3
"""Validate the bridge FEA case corpus, direct-question summary and JSON contracts."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "data" / "bridge_fea_mesh_cases"
QUESTION_DIR = ROOT / "data" / "bridge_engineer_questions"

CASE_FIELDS = (
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
EXPECTED_QUESTION_COUNTS = {
    "atomic_question_count": 174,
    "source_thread_or_faq_count": 81,
    "unique_url_count": 81,
}


def require_nonblank(rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    for row in rows:
        for field in fields:
            if not row[field].strip():
                raise SystemExit(f"Blank field {field!r} in case row {row['id']}")


def validate_case_corpus() -> None:
    rows: list[dict[str, str]] = []
    files = sorted(CASE_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No case-corpus CSV files found in {CASE_DIR}")

    for path in files:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CASE_FIELDS:
                raise SystemExit(f"Unexpected fields in {path}: {reader.fieldnames}")
            rows.extend(dict(row) for row in reader)

    ids = [row["id"] for row in rows]
    duplicates = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate case IDs: {duplicates}")

    tiers = Counter(row["evidence_tier"] for row in rows)
    if dict(tiers) != EXPECTED_TIERS:
        raise SystemExit(f"Unexpected evidence-tier counts: {dict(tiers)}")

    bad_urls = [row["id"] for row in rows if not row["url"].startswith(("https://", "http://"))]
    if bad_urls:
        raise SystemExit(f"Case rows with invalid URLs: {bad_urls}")

    require_nonblank(rows, CASE_FIELDS)
    print(f"validated_case_rows={len(rows)}")
    print(f"evidence_tiers={dict(tiers)}")


def validate_question_research() -> None:
    summary_path = QUESTION_DIR / "summary.json"
    readme_path = QUESTION_DIR / "README.md"
    findings_path = ROOT / "docs" / "bridge_fea_mesh" / "ENGINEER_QUESTION_FINDINGS.md"
    coding_path = ROOT / "docs" / "bridge_fea_mesh" / "QUESTION_CODING_METHOD.md"

    for path in (summary_path, readme_path, findings_path, coding_path):
        if not path.exists():
            raise SystemExit(f"Missing direct-question research file: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for field, expected in EXPECTED_QUESTION_COUNTS.items():
        actual = int(summary.get(field, -1))
        if actual != expected:
            raise SystemExit(f"Unexpected {field}: expected {expected}, found {actual}")

    source_counts = summary.get("source_type_counts", {})
    if sum(int(value) for value in source_counts.values()) != EXPECTED_QUESTION_COUNTS["atomic_question_count"]:
        raise SystemExit("source_type_counts do not sum to atomic_question_count")

    phase_counts = summary.get("workflow_phase_counts", {})
    if sum(int(value) for value in phase_counts.values()) != EXPECTED_QUESTION_COUNTS["atomic_question_count"]:
        raise SystemExit("workflow_phase_counts do not sum to atomic_question_count")

    if "public-question sample" not in " ".join(summary.get("interpretation_boundary", [])).lower():
        raise SystemExit("Question summary must retain the public-sample interpretation boundary")

    # A previously corrupted binary detail file must never silently reappear.
    bad_binary = QUESTION_DIR / "bridge_engineer_questions.csv.gz"
    if bad_binary.exists():
        raise SystemExit("Unreviewed binary question detail file is present")

    print(f"validated_question_summary={summary_path.name}")
    print(f"atomic_questions={summary['atomic_question_count']}")
    print(f"question_sources={summary['source_thread_or_faq_count']}")


def main() -> None:
    validate_case_corpus()
    validate_question_research()
    json.loads((ROOT / "data" / "bridge_mesh_question_templates.json").read_text(encoding="utf-8"))
    json.loads((ROOT / "schemas" / "bridge_mesh_intent.schema.json").read_text(encoding="utf-8"))
    print("json_contracts=valid")


if __name__ == "__main__":
    main()
