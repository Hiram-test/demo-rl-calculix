#!/usr/bin/env python3
"""Validate bridge FEA evidence, engineer-question and mesh-only corpora."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "data" / "bridge_fea_mesh_cases"
QUESTION_DIR = ROOT / "data" / "bridge_engineer_questions"
MESH_QUESTION_DIR = ROOT / "data" / "bridge_mesh_questions"

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
MESH_QUESTION_FIELDS = (
    "question_id",
    "source_group_id",
    "source_scope",
    "channel",
    "year",
    "component_or_context",
    "mesh_need_group",
    "question_paraphrase",
    "decision_needed",
    "qoi_or_result",
    "source_title",
    "url",
)
EXPECTED_TIERS = {"T1": 37, "T2": 48, "T3": 5, "T4": 10}
EXPECTED_QUESTION_COUNTS = {
    "atomic_question_count": 174,
    "source_thread_or_faq_count": 81,
    "unique_url_count": 81,
}
EXPECTED_MESH_QUESTION_COUNTS = {
    "atomic_question_count": 103,
    "source_page_count": 36,
    "unique_url_count": 36,
}
EXPECTED_MESH_SCOPE_COUNTS = {
    "bridge_specific": 81,
    "structural_mesh_transfer": 22,
}


def require_nonblank(
    rows: list[dict[str, str]], fields: tuple[str, ...], id_field: str
) -> None:
    for row in rows:
        for field in fields:
            if not row[field].strip():
                raise SystemExit(
                    f"Blank field {field!r} in row {row.get(id_field, '<unknown>')}"
                )


def read_csv_parts(
    directory: Path, pattern: str, expected_fields: tuple[str, ...]
) -> tuple[list[dict[str, str]], list[Path]]:
    rows: list[dict[str, str]] = []
    files = sorted(directory.glob(pattern))
    if not files:
        raise SystemExit(f"No CSV files matching {pattern!r} in {directory}")

    for path in files:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != expected_fields:
                raise SystemExit(f"Unexpected fields in {path}: {reader.fieldnames}")
            rows.extend(dict(row) for row in reader)
    return rows, files


def validate_case_corpus() -> None:
    rows, files = read_csv_parts(CASE_DIR, "*.csv", CASE_FIELDS)
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

    require_nonblank(rows, CASE_FIELDS, "id")
    print(f"validated_case_rows={len(rows)}")
    print(f"case_files={[path.name for path in files]}")
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

    bad_binary = QUESTION_DIR / "bridge_engineer_questions.csv.gz"
    if bad_binary.exists():
        raise SystemExit("Unreviewed binary question detail file is present")

    print(f"validated_question_summary={summary_path.name}")
    print(f"atomic_questions={summary['atomic_question_count']}")
    print(f"question_sources={summary['source_thread_or_faq_count']}")


def validate_mesh_only_questions() -> None:
    rows, files = read_csv_parts(
        MESH_QUESTION_DIR, "mesh_only_questions_*.csv", MESH_QUESTION_FIELDS
    )
    summary_path = MESH_QUESTION_DIR / "summary.json"
    readme_path = MESH_QUESTION_DIR / "README.md"
    findings_path = ROOT / "docs" / "bridge_fea_mesh" / "MESH_ONLY_QUESTION_FINDINGS.md"
    for path in (summary_path, readme_path, findings_path):
        if not path.exists():
            raise SystemExit(f"Missing mesh-only question file: {path}")

    require_nonblank(rows, MESH_QUESTION_FIELDS, "question_id")

    ids = [row["question_id"] for row in rows]
    expected_ids = [f"MQ{index:03d}" for index in range(1, 104)]
    if ids != expected_ids:
        missing = sorted(set(expected_ids) - set(ids))
        extra = sorted(set(ids) - set(expected_ids))
        raise SystemExit(
            f"Mesh question IDs are not the ordered MQ001-MQ103 sequence; "
            f"missing={missing}, extra={extra}"
        )

    duplicates = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate mesh-question IDs: {duplicates}")

    bad_urls = [
        row["question_id"]
        for row in rows
        if not row["url"].startswith(("https://", "http://"))
    ]
    if bad_urls:
        raise SystemExit(f"Mesh-question rows with invalid URLs: {bad_urls}")

    source_groups = {row["source_group_id"] for row in rows}
    urls = {row["url"] for row in rows}
    if len(source_groups) != EXPECTED_MESH_QUESTION_COUNTS["source_page_count"]:
        raise SystemExit(f"Unexpected mesh source-group count: {len(source_groups)}")
    if len(urls) != EXPECTED_MESH_QUESTION_COUNTS["unique_url_count"]:
        raise SystemExit(f"Unexpected mesh unique-URL count: {len(urls)}")

    group_to_urls: dict[str, set[str]] = {}
    for row in rows:
        group_to_urls.setdefault(row["source_group_id"], set()).add(row["url"])
    inconsistent_groups = sorted(group for group, group_urls in group_to_urls.items() if len(group_urls) != 1)
    if inconsistent_groups:
        raise SystemExit(f"Source groups map to multiple URLs: {inconsistent_groups}")

    scopes = Counter(row["source_scope"] for row in rows)
    if dict(scopes) != EXPECTED_MESH_SCOPE_COUNTS:
        raise SystemExit(f"Unexpected mesh source-scope counts: {dict(scopes)}")

    need_groups = Counter(row["mesh_need_group"] for row in rows)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for field, expected in EXPECTED_MESH_QUESTION_COUNTS.items():
        actual = int(summary.get(field, -1))
        if actual != expected:
            raise SystemExit(
                f"Unexpected mesh summary {field}: expected {expected}, found {actual}"
            )
    if summary.get("source_scope_counts") != dict(scopes):
        raise SystemExit("Mesh summary source_scope_counts do not match CSV rows")
    if summary.get("mesh_need_group_counts") != dict(need_groups):
        raise SystemExit("Mesh summary mesh_need_group_counts do not match CSV rows")
    if sum(int(value) for value in summary.get("atomic_question_channel_counts", {}).values()) != len(rows):
        raise SystemExit("Mesh summary channel counts do not sum to CSV row count")
    if "not the global population" not in " ".join(summary.get("interpretation_boundary", [])).lower():
        raise SystemExit("Mesh-only summary must retain the sample interpretation boundary")

    print(f"validated_mesh_question_rows={len(rows)}")
    print(f"mesh_question_files={[path.name for path in files]}")
    print(f"mesh_question_source_pages={len(source_groups)}")
    print(f"mesh_question_scopes={dict(scopes)}")
    print(f"mesh_need_groups={dict(need_groups)}")


def main() -> None:
    validate_case_corpus()
    validate_question_research()
    validate_mesh_only_questions()
    json.loads((ROOT / "data" / "bridge_mesh_question_templates.json").read_text(encoding="utf-8"))
    json.loads((ROOT / "schemas" / "bridge_mesh_intent.schema.json").read_text(encoding="utf-8"))
    print("json_contracts=valid")


if __name__ == "__main__":
    main()
