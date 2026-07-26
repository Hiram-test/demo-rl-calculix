#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mesh_need.core import run_pipeline
from mesh_need.web_ui import serve


def _load_json(path: str | None, default: Any) -> Any:
    if not path:
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_json(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    return json.loads(raw)


def ask(args: argparse.Namespace) -> None:
    question = args.question or input("请输入你的网格/有限元问题：\n> ").strip()
    case = {
        "question": question,
        "context": args.context or "",
        "intended_use": args.intended_use or "",
        "current_claim": args.current_claim or "",
        "calculix_inp": args.calculix_inp or "",
        "qoi": {"name": args.qoi_name or "", "location": args.qoi_location or "", "extraction_method": args.qoi_method or "", "tolerance": args.qoi_tolerance},
        "mesh_series": _parse_json(args.mesh_series, []),
        "energy_history": _parse_json(args.energy_history, []),
    }
    result = run_pipeline(case, args.output_dir, _load_json(args.ai_proposal, None))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def pipeline(args: argparse.Namespace) -> None:
    case = _load_json(args.case, {})
    result = run_pipeline(case, args.output_dir, _load_json(args.ai_proposal, None))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-aware FEA mesh-need diagnosis and evidence ledger")
    sub = parser.add_subparsers(dest="command", required=True)
    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("--question")
    ask_parser.add_argument("--context")
    ask_parser.add_argument("--intended-use")
    ask_parser.add_argument("--current-claim")
    ask_parser.add_argument("--calculix-inp")
    ask_parser.add_argument("--qoi-name")
    ask_parser.add_argument("--qoi-location")
    ask_parser.add_argument("--qoi-method")
    ask_parser.add_argument("--qoi-tolerance", type=float, default=0.02)
    ask_parser.add_argument("--mesh-series", help="JSON array")
    ask_parser.add_argument("--energy-history", help="JSON array")
    ask_parser.add_argument("--ai-proposal", help="path to candidate diagnosis JSON")
    ask_parser.add_argument("--output-dir", default="mesh-need-runs/latest")
    ask_parser.set_defaults(func=ask)
    pipeline_parser = sub.add_parser("pipeline")
    pipeline_parser.add_argument("--case", required=True)
    pipeline_parser.add_argument("--ai-proposal")
    pipeline_parser.add_argument("--output-dir", required=True)
    pipeline_parser.set_defaults(func=pipeline)
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.set_defaults(func=lambda args: serve(args.host, args.port))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
