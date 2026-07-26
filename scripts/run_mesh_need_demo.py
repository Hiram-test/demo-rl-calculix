#!/usr/bin/env python3
"""Run one model-aware mesh-need diagnosis example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mesh_need import diagnose_question  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        nargs="?",
        default=str(REPO_ROOT / "examples/mesh_need/fatigue_weld_hotspot_question.json"),
    )
    args = parser.parse_args()

    case_path = Path(args.case)
    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read case {case_path}: {exc}")

    diagnosis = diagnose_question(case)
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
