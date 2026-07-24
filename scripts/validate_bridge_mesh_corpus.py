#!/usr/bin/env python3
"""Validate the bridge FEA source corpus and direct engineer-question corpus."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
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

QUESTION_FIELDS = (
    "question_id",
    "source_group_id",
    "channel",
    "source_type",
    "year",
    "software",
    "component_family",
    "analysis_context",
    "source_title",
    "question_paraphrase",
    "explicit_need",
    "latent_need",
    "need_family",
    "engineer_job",
    "workflow_phase",
    "engineering_qoi",
    "failure_or_risk",
    "desired_answer_type",
    "answer_mode",
    "mesh_role_group",
    "mesh_role",
    "url",
    "review_status",
)
EXPECTED_QUESTION_COUNT = 174
EXPECTED_SOURCE_GROUP_COUNT = 81
ALLOWED_PHASES = {
    "problem_definition_and_modeling",
    "verification_and_acceptance",
    "meshing",
    "postprocessing",
    "workflow_and_scale",
    "solution",
}
ALLOWED_MESH_ROLES = {
    "primary",
    "joint",
    "indirect",
    "model_before_mesh",
    "workflow",
}
ALLOWED_ANSWER_MODES = {
    "choose_or_configure_model",
    "diagnose_and_debug",
    "verify_and_accept",
    "automate_and_execute",
    "interpret_results",
    "make_decision",
}


def require_nonblank(rows: list[dict[str, str]], fields: tuple[str, ...], id_field: str) -> None:
    for row in rows:
        for field in fields:
            if not row[field].strip():
                raise SystemExit(f"Blank field {field!r} in row {row[id_field]}")


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
        raise SystemExit(f"Duplicate case-catalog IDs: {duplicates}")

    tiers = Counter(row["evidence_tier"] for row in rows)
    if dict(tiers) != EXPECTED_TIERS:
        raise SystemExit(f"Unexpected evidence-tier counts: {dict(tiers)}")

    bad_urls = [row["id"] for row in rows if not row["url"].startswith(("https://", "http://"))]
    if bad_urls:
        raise SystemExit(f"Case rows with invalid URLs: {bad_urls}")

    require_nonblank(rows, CASE_FIELDS, "id")
    print(f"validated_case_rows={len(rows)}")
    print(f"evidence_tiers={dict(tiers)}")


def validate_question_corpus() -> None:
    question_path = QUESTION_DIR / "bridge_engineer_questions.csv.gz"
    summary_path = QUESTION_DIR / "summary.json"

    if not question_path.exists():
        raise SystemExit(f"Direct-question corpus not found: {question_path}")

    with gzip.open(question_path, mode="rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != QUESTION_FIELDS:
            raise SystemExit(f"Unexpected question fields: {reader.fieldnames}")
        rows = [dict(row) for row in reader]

    if len(rows) != EXPECTED_QUESTION_COUNT:
        raise SystemExit(f"Expected {EXPECTED_QUESTION_COUNT} questions, found {len(rows)}")

    ids = [row["question_id"] for row in rows]
    duplicates = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate question IDs: {duplicates}")

    source_groups = {row["source_group_id"] for row in rows}
    if len(source_groups) != EXPECTED_SOURCE_GROUP_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_SOURCE_GROUP_COUNT} source groups, found {len(source_groups)}"
        )

    bad_urls = [
        row["question_id"]
        for row in rows
        if not row["url"].startswith(("https://", "http://"))
    ]
    if bad_urls:
        raise SystemExit(f"Question rows with invalid URLs: {bad_urls}")

    require_nonblank(rows, QUESTION_FIELDS, "question_id")

    for row in rows:
        if row["workflow_phase"] not in ALLOWED_PHASES:
            raise SystemExit(
                f"Unknown workflow phase {row['workflow_phase']!r} in {row['question_id']}"
            )
        if row["mesh_role_group"] not in ALLOWED_MESH_ROLES:
            raise SystemExit(
                f"Unknown mesh role group {row['mesh_role_group']!r} in {row['question_id']}"
            )
        if row["answer_mode"] not in ALLOWED_ANSWER_MODES:
            raise SystemExit(
                f"Unknown answer mode {row['answer_mode']!r} in {row['question_id']}"
            )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["atomic_question_count"]) != len(rows):
        raise SystemExit("Question summary row count does not match CSV")
    if int(summary["source_thread_or_faq_count"]) != len(source_groups):
        raise SystemExit("Question summary source-group count does not match CSV")

    print(f"validated_question_rows={len(rows)}")
    print(f"validated_question_sources={len(source_groups)}")
    print(f"question_file={question_path.name}")
    print(f"workflow_phases={dict(Counter(row['workflow_phase'] for row in rows))}")


def main() -> None:
    validate_case_corpus()
    validate_question_corpus()
    json.loads((ROOT / "data" / "bridge_mesh_question_templates.json").read_text(encoding="utf-8"))
    json.loads((ROOT / "schemas" / "bridge_mesh_intent.schema.json").read_text(encoding="utf-8"))
    print("json_contracts=valid")


if __name__ == "__main__":
    main()
