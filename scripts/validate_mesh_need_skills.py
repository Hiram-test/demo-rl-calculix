#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jsonschema import Draft202012Validator

from mesh_need.core import run_pipeline



def main() -> None:
    diagnosis_schema = json.loads((ROOT / "schemas/mesh_need_diagnosis.schema.json").read_text(encoding="utf-8"))
    ledger_schema = json.loads((ROOT / "schemas/evidence_ledger.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(diagnosis_schema)
    Draft202012Validator.check_schema(ledger_schema)
    examples = [
        "crack_tip_case.json",
        "bounded_hotspot_case.json",
        "qoi_indicator_case.json",
        "multi_hotspot_pso_case.json",
        "disconnected_case.json",
    ]
    with tempfile.TemporaryDirectory() as temp:
        for index, filename in enumerate(examples):
            case = json.loads((ROOT / "examples/mesh_need" / filename).read_text(encoding="utf-8"))
            if case.get("calculix_inp"):
                case["calculix_inp"] = str(ROOT / case["calculix_inp"])
            proposal = None
            if filename == "crack_tip_case.json":
                proposal = json.loads((ROOT / "examples/mesh_need/unsafe_ai_hotspot_proposal.json").read_text(encoding="utf-8"))
            out = Path(temp) / str(index)
            run_pipeline(case, out, proposal)
            diagnosis = json.loads((out / "diagnosis.json").read_text(encoding="utf-8"))
            ledger = json.loads((out / "evidence_ledger.json").read_text(encoding="utf-8"))
            Draft202012Validator(diagnosis_schema).validate(diagnosis)
            Draft202012Validator(ledger_schema).validate(ledger)
            assert ledger["final_verdict"] is None
    print(f"validated {len(examples)} runnable cases and both Draft 2020-12 schemas")


if __name__ == "__main__":
    main()
