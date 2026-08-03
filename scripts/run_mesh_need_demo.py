#!/usr/bin/env python3
"""Build an AI analysis packet for one engineering question."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mesh_need import build_analysis_packet  # noqa: E402


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

    packet = build_analysis_packet(case)
    print(json.dumps(packet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
