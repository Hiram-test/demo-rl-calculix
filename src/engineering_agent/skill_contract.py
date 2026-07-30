"""Contracts for engineering-experience skills.

A skill describes a reusable engineering reasoning process. It is not an
executable case builder and must not carry current-case geometry, material,
load, boundary-condition, mesh, or optimisation values.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

_FORBIDDEN_KEYS = {
    "builder",
    "callable",
    "case_parameters",
    "finish_requirements",
    "fixed_parameters",
    "function",
    "required_artifacts",
}


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    name: str
    purpose: str
    acceptable_sources: tuple[str, ...] = ()
    optional: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvidenceRequirement":
        return cls(
            name=str(payload.get("name", "")).strip(),
            purpose=str(payload.get("purpose", "")).strip(),
            acceptable_sources=tuple(
                str(item) for item in payload.get("acceptable_sources", [])
            ),
            optional=bool(payload.get("optional", False)),
        )


@dataclass(frozen=True, slots=True)
class EngineeringSkill:
    skill_id: str
    title: str
    purpose: str
    engineering_questions: tuple[str, ...] = ()
    evidence_requirements: tuple[EvidenceRequirement, ...] = ()
    procedure: tuple[str, ...] = ()
    tool_guidance: tuple[str, ...] = ()
    output_records: tuple[str, ...] = ()
    non_goals: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    version: str = "1.0"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EngineeringSkill":
        forbidden = sorted(_FORBIDDEN_KEYS.intersection(payload))
        if forbidden:
            raise ValueError(
                "engineering skill contains case-execution keys: "
                + ", ".join(forbidden)
            )
        return cls(
            skill_id=str(payload.get("skill_id", "")).strip(),
            title=str(payload.get("title", "")).strip(),
            purpose=str(payload.get("purpose", "")).strip(),
            engineering_questions=tuple(
                str(item) for item in payload.get("engineering_questions", [])
            ),
            evidence_requirements=tuple(
                EvidenceRequirement.from_mapping(item)
                for item in payload.get("evidence_requirements", [])
            ),
            procedure=tuple(str(item) for item in payload.get("procedure", [])),
            tool_guidance=tuple(
                str(item) for item in payload.get("tool_guidance", [])
            ),
            output_records=tuple(
                str(item) for item in payload.get("output_records", [])
            ),
            non_goals=tuple(str(item) for item in payload.get("non_goals", [])),
            tags=tuple(str(item) for item in payload.get("tags", [])),
            version=str(payload.get("version", "1.0")),
        )

    def issues(self) -> list[str]:
        issues: list[str] = []
        if not self.skill_id:
            issues.append("skill_id is empty")
        if not self.title:
            issues.append(f"{self.skill_id or '<unknown>'}: title is empty")
        if not self.purpose:
            issues.append(f"{self.skill_id or '<unknown>'}: purpose is empty")
        if not self.procedure:
            issues.append(f"{self.skill_id or '<unknown>'}: procedure is empty")
        for index, item in enumerate(self.evidence_requirements):
            if not item.name or not item.purpose:
                issues.append(
                    f"{self.skill_id or '<unknown>'}: "
                    f"evidence requirement {index} is incomplete"
                )
        return issues

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "title": self.title,
            "purpose": self.purpose,
            "engineering_questions": list(self.engineering_questions),
            "evidence_requirements": [
                asdict(item) for item in self.evidence_requirements
            ],
            "procedure": list(self.procedure),
            "tool_guidance": list(self.tool_guidance),
            "output_records": list(self.output_records),
            "non_goals": list(self.non_goals),
            "tags": list(self.tags),
            "version": self.version,
        }


@dataclass(slots=True)
class SkillLibrary:
    skills: dict[str, EngineeringSkill] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)

    def register(self, skill: EngineeringSkill) -> None:
        if skill.skill_id in self.skills:
            self.observations.append(f"duplicate skill replaced: {skill.skill_id}")
        self.skills[skill.skill_id] = skill
        self.observations.extend(skill.issues())

    def prompt_catalog(self) -> list[dict[str, Any]]:
        return [
            self.skills[key].to_prompt_dict()
            for key in sorted(self.skills)
        ]

    @classmethod
    def load_json_directory(cls, directory: str | Path) -> "SkillLibrary":
        library = cls()
        root = Path(directory)
        if not root.exists():
            library.observations.append(
                f"skill directory does not exist: {root}"
            )
            return library
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                library.register(EngineeringSkill.from_mapping(payload))
            except Exception as exc:
                library.observations.append(
                    f"could not load {path.name}: {exc}"
                )
        return library
