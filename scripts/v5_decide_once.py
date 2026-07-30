#!/usr/bin/env python3
"""Run one live V5 DeepSeek decision and always leave a trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engineering_agent.decision_loop import DecisionLoop, TraceRecorder
from engineering_agent.deepseek_client import DeepSeekChatClient
from engineering_agent.problem_manifest import ProblemManifest
from engineering_agent.skill_contract import SkillLibrary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--skills", default="skills/engineering")
    parser.add_argument(
        "--trace-dir",
        default="artifacts/v5-live-decision",
    )
    args = parser.parse_args()

    manifest_payload = json.loads(
        Path(args.manifest).read_text(encoding="utf-8")
    )
    manifest = ProblemManifest.from_mapping(manifest_payload)
    skills = SkillLibrary.load_json_directory(args.skills)
    trace = TraceRecorder(Path(args.trace_dir))

    try:
        model = DeepSeekChatClient.from_environment()
    except Exception as exc:
        error_message = str(exc)

        class ConfigurationFailureModel:
            def __init__(self, message: str) -> None:
                self.message = message

            def complete_json(self, messages):
                raise RuntimeError(self.message)

        model = ConfigurationFailureModel(error_message)

    loop = DecisionLoop(
        model=model,
        manifest=manifest,
        skills=skills,
        trace=trace,
    )
    decision = loop.decide_next()
    print(
        json.dumps(
            decision.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
