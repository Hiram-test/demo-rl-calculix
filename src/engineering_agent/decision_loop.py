"""Evidence-driven DeepSeek decision loop for V5.

The loop has no required-artifact checklist. Each model call receives only the
current problem manifest, reusable engineering-method skills, and observations
from this run. It may ask the user, generate code, run or revise code, record a
finding, or finish. Errors are traced and returned as observations rather than
used to suppress later reports.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .problem_manifest import ProblemManifest
from .skill_contract import SkillLibrary


class DecisionAction(str, Enum):
    ASK_USER = "ask_user"
    PROPOSE_ASSUMPTION = "propose_assumption"
    WRITE_CODE = "write_code"
    RUN_COMMAND = "run_command"
    INSPECT_RESULT = "inspect_result"
    REVISE_CODE = "revise_code"
    RECORD_FINDING = "record_finding"
    FINISH = "finish"


@dataclass(frozen=True, slots=True)
class GeneratedFile:
    path: str
    content: str
    purpose: str = ""


@dataclass(frozen=True, slots=True)
class ModelResult:
    payload: Mapping[str, Any]
    raw_response: Mapping[str, Any] | str | None = None
    provider: str = "unknown"
    model: str = "unknown"


class DecisionModel(Protocol):
    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> ModelResult: ...


@dataclass(slots=True)
class AgentDecision:
    iteration: int
    action: DecisionAction
    rationale: str
    selected_skill_ids: list[str] = field(default_factory=list)
    questions_for_user: list[str] = field(default_factory=list)
    assumptions: list[dict[str, Any]] = field(default_factory=list)
    files: list[GeneratedFile] = field(default_factory=list)
    command: str | None = None
    expected_observation: str | None = None
    next_objective: str | None = None
    completion_note: str | None = None
    issues: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


@dataclass(slots=True)
class TraceRecorder:
    root: Path
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "model_io").mkdir(exist_ok=True)

    def record_request(
        self,
        iteration: int,
        payload: Mapping[str, Any],
    ) -> None:
        self._write_json(
            self.root / "model_io" / f"iteration_{iteration:04d}_request.json",
            payload,
        )

    def record_response(self, iteration: int, payload: Any) -> None:
        self._write_json(
            self.root / "model_io" / f"iteration_{iteration:04d}_response.json",
            payload,
        )

    def append_event(self, event: Mapping[str, Any]) -> None:
        row = dict(event)
        row.setdefault(
            "timestamp_utc",
            datetime.now(timezone.utc).isoformat(),
        )
        self.events.append(row)
        self._write_json(self.root / "agent_trace.json", self.events)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )


_SYSTEM_PROMPT = """You are the code-generating engineering decision agent for V5.
Return one JSON object only.

Principles:
- Use only facts and sources in the current problem manifest.
- Never copy geometry, material, load, boundary-condition, mesh, or optimisation values from old cases.
- A skill is an engineering reasoning process, not a prewritten case function.
- If a necessary fact is absent, ask the user or state a clearly labelled assumption pending confirmation.
- Generate code for the current problem when code is the appropriate next action.
- Select PSO, nonlinear analysis, hotspot analysis, theory checks, or any other method only when evidence makes it useful.
- There is no required-artifact checklist and no solver-call quota.
- Errors, failed calculations, incomplete evidence, and uncertainty must be recorded; they do not prohibit reporting.
- Choose exactly one next action.

Allowed action values:
ask_user, propose_assumption, write_code, run_command, inspect_result,
revise_code, record_finding, finish.

JSON shape example:
{
  "action": "ask_user",
  "rationale": "The load definition is missing.",
  "selected_skill_ids": ["problem-definition-source-audit"],
  "questions_for_user": ["Provide the load magnitude, direction and source."],
  "assumptions": [],
  "files": [],
  "command": null,
  "expected_observation": null,
  "next_objective": "Establish the physical problem before generating a model.",
  "completion_note": null
}
"""


@dataclass(slots=True)
class DecisionLoop:
    model: DecisionModel
    manifest: ProblemManifest
    skills: SkillLibrary
    trace: TraceRecorder
    workspace_summary: str = ""
    observations: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[AgentDecision] = field(default_factory=list)
    prompt_version: str = "v5-decision-1"

    def decide_next(self) -> AgentDecision:
        iteration = len(self.decisions) + 1
        request_payload = {
            "prompt_version": self.prompt_version,
            "iteration": iteration,
            "problem_manifest": self.manifest.to_dict(),
            "engineering_skills": self.skills.prompt_catalog(),
            "skill_load_observations": list(self.skills.observations),
            "workspace_summary": self.workspace_summary,
            "recent_observations": self.observations[-8:],
            "previous_decisions": [
                item.to_dict() for item in self.decisions[-5:]
            ],
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Current V5 decision state as JSON:\n"
                + json.dumps(
                    request_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        self.trace.record_request(
            iteration,
            {"messages": messages, **request_payload},
        )
        try:
            result = self.model.complete_json(messages)
            response_record = {
                "provider": result.provider,
                "model": result.model,
                "parsed_payload": dict(result.payload),
                "raw_response": result.raw_response,
            }
            self.trace.record_response(iteration, response_record)
            decision = _parse_decision(iteration, result.payload)
        except Exception as exc:
            response_record = {
                "error": type(exc).__name__,
                "message": str(exc),
            }
            self.trace.record_response(iteration, response_record)
            decision = AgentDecision(
                iteration=iteration,
                action=DecisionAction.RECORD_FINDING,
                rationale=(
                    "The model call or response parsing failed; record it "
                    "and keep the run reportable."
                ),
                issues=[f"{type(exc).__name__}: {exc}"],
                next_objective=(
                    "Repair configuration or response format, then retry "
                    "from the same manifest."
                ),
                raw_payload=response_record,
            )
        self.decisions.append(decision)
        self.trace.append_event(
            {"type": "decision", "decision": decision.to_dict()}
        )
        return decision

    def record_execution_observation(
        self,
        observation: Mapping[str, Any],
    ) -> None:
        row = dict(observation)
        self.observations.append(row)
        self.trace.append_event(
            {"type": "execution_observation", "observation": row}
        )

    def run_until_pause(
        self,
        *,
        max_steps: int = 1,
        executor: Callable[
            [AgentDecision], Mapping[str, Any] | None
        ]
        | None = None,
    ) -> list[AgentDecision]:
        """Run a bounded amount of work.

        max_steps is an operational budget, not a correctness or publication
        gate. The loop pauses for user input, on finish, or when no executor is
        supplied.
        """
        for _ in range(max_steps):
            decision = self.decide_next()
            if decision.action in {
                DecisionAction.ASK_USER,
                DecisionAction.FINISH,
            }:
                break
            if executor is None:
                break
            try:
                observation = executor(decision)
                if observation is not None:
                    self.record_execution_observation(observation)
            except Exception as exc:
                self.record_execution_observation(
                    {
                        "status": "error",
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        return list(self.decisions)


def _parse_decision(
    iteration: int,
    payload: Mapping[str, Any],
) -> AgentDecision:
    raw = dict(payload)
    issues: list[str] = []
    try:
        action = DecisionAction(
            str(payload.get("action", "record_finding"))
        )
    except ValueError:
        action = DecisionAction.RECORD_FINDING
        issues.append(f"unknown action: {payload.get('action')!r}")

    files: list[GeneratedFile] = []
    for index, item in enumerate(payload.get("files", []) or []):
        if not isinstance(item, Mapping):
            issues.append(f"files[{index}] is not an object")
            continue
        path = str(item.get("path", "")).strip()
        content = str(item.get("content", ""))
        if not path:
            issues.append(f"files[{index}] has no path")
            continue
        files.append(
            GeneratedFile(
                path=path,
                content=content,
                purpose=str(item.get("purpose", "")),
            )
        )

    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        issues.append("rationale is empty")
        rationale = (
            "No rationale was returned; this is recorded for the next revision."
        )

    return AgentDecision(
        iteration=iteration,
        action=action,
        rationale=rationale,
        selected_skill_ids=[
            str(item)
            for item in payload.get("selected_skill_ids", []) or []
        ],
        questions_for_user=[
            str(item)
            for item in payload.get("questions_for_user", []) or []
        ],
        assumptions=[
            dict(item)
            for item in payload.get("assumptions", []) or []
            if isinstance(item, Mapping)
        ],
        files=files,
        command=(
            None
            if payload.get("command") in (None, "")
            else str(payload.get("command"))
        ),
        expected_observation=(
            None
            if payload.get("expected_observation") in (None, "")
            else str(payload.get("expected_observation"))
        ),
        next_objective=(
            None
            if payload.get("next_objective") in (None, "")
            else str(payload.get("next_objective"))
        ),
        completion_note=(
            None
            if payload.get("completion_note") in (None, "")
            else str(payload.get("completion_note"))
        ),
        issues=issues,
        raw_payload=raw,
    )
