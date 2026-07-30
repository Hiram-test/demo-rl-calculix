"""Problem-manifest and provenance contracts.

The manifest is the first boundary in the rewritten system:

* engineering facts must come from the current user/model or be explicitly
  derived by the agent;
* missing required facts remain missing and are turned into questions;
* no geometry, material, load, boundary condition, or engineering parameter is
  silently supplied by code defaults;
* algorithm configuration is kept separate from physical model facts.

These contracts record state. They do not block later reporting or research
output; callers decide how to continue an exploratory run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class ProvenanceKind(str, Enum):
    """Allowed origins for a value used by the engineering agent."""

    PARSED_FROM_USER_MODEL = "parsed_from_user_model"
    PROVIDED_BY_USER = "provided_by_user"
    DERIVED_BY_AGENT = "derived_by_agent"
    ASSUMPTION_PENDING_CONFIRMATION = "assumption_pending_confirmation"
    ALGORITHM_CONFIGURATION = "algorithm_configuration"
    LEGACY_FIXTURE = "legacy_fixture"


@dataclass(frozen=True, slots=True)
class FactRecord:
    """One fact, decision, assumption, or algorithm setting with provenance."""

    path: str
    value: Any
    provenance: ProvenanceKind
    source_ref: str | None = None
    derivation: str | None = None
    user_confirmed: bool = False
    notes: str | None = None

    def provenance_issues(self) -> list[str]:
        issues: list[str] = []
        if not self.path.strip():
            issues.append("fact path is empty")
        if self.provenance in {
            ProvenanceKind.PARSED_FROM_USER_MODEL,
            ProvenanceKind.PROVIDED_BY_USER,
        } and not self.source_ref:
            issues.append(f"{self.path}: source_ref is required for {self.provenance.value}")
        if self.provenance is ProvenanceKind.DERIVED_BY_AGENT and not self.derivation:
            issues.append(f"{self.path}: derivation is required for agent-derived values")
        if (
            self.provenance is ProvenanceKind.ASSUMPTION_PENDING_CONFIRMATION
            and self.user_confirmed
        ):
            issues.append(
                f"{self.path}: pending assumption cannot already be marked user_confirmed"
            )
        return issues


@dataclass(frozen=True, slots=True)
class MissingFact:
    """A required fact that the current inputs do not establish."""

    path: str
    reason: str
    question: str
    acceptable_sources: tuple[str, ...] = ()


@dataclass(slots=True)
class ProblemManifest:
    """Source-of-truth manifest for one current engineering problem."""

    manifest_version: str = "1.0"
    task_id: str = ""
    user_goal: str = ""
    input_files: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, FactRecord] = field(default_factory=dict)
    missing_facts: list[MissingFact] = field(default_factory=list)
    algorithm_configuration: dict[str, FactRecord] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)

    def add_fact(self, fact: FactRecord) -> None:
        if fact.provenance is ProvenanceKind.ALGORITHM_CONFIGURATION:
            raise ValueError(
                "algorithm configuration must be added with add_algorithm_configuration"
            )
        self.facts[fact.path] = fact
        self.missing_facts = [row for row in self.missing_facts if row.path != fact.path]

    def add_algorithm_configuration(self, setting: FactRecord) -> None:
        if setting.provenance is not ProvenanceKind.ALGORITHM_CONFIGURATION:
            raise ValueError("algorithm setting must use algorithm_configuration provenance")
        self.algorithm_configuration[setting.path] = setting

    def require_fact(
        self,
        path: str,
        *,
        reason: str,
        question: str,
        acceptable_sources: Iterable[str] = (),
    ) -> None:
        if path in self.facts:
            return
        row = MissingFact(
            path=path,
            reason=reason,
            question=question,
            acceptable_sources=tuple(acceptable_sources),
        )
        self.missing_facts = [item for item in self.missing_facts if item.path != path]
        self.missing_facts.append(row)

    def questions_for_user(self) -> list[dict[str, Any]]:
        return [
            {
                "field": row.path,
                "reason": row.reason,
                "question": row.question,
                "acceptable_sources": list(row.acceptable_sources),
            }
            for row in self.missing_facts
        ]

    def provenance_report(self) -> dict[str, Any]:
        issues: list[str] = []
        for fact in [*self.facts.values(), *self.algorithm_configuration.values()]:
            issues.extend(fact.provenance_issues())

        return {
            "task_id": self.task_id,
            "fact_count": len(self.facts),
            "algorithm_setting_count": len(self.algorithm_configuration),
            "missing_fact_count": len(self.missing_facts),
            "issues": issues,
            "has_legacy_fixture_values": any(
                item.provenance is ProvenanceKind.LEGACY_FIXTURE
                for item in self.facts.values()
            ),
            "has_pending_assumptions": any(
                item.provenance is ProvenanceKind.ASSUMPTION_PENDING_CONFIRMATION
                for item in self.facts.values()
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "task_id": self.task_id,
            "user_goal": self.user_goal,
            "input_files": list(self.input_files),
            "facts": {key: _record_to_dict(value) for key, value in self.facts.items()},
            "missing_facts": [asdict(item) for item in self.missing_facts],
            "algorithm_configuration": {
                key: _record_to_dict(value)
                for key, value in self.algorithm_configuration.items()
            },
            "observations": list(self.observations),
            "provenance_report": self.provenance_report(),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProblemManifest":
        manifest = cls(
            manifest_version=str(payload.get("manifest_version", "1.0")),
            task_id=str(payload.get("task_id", "")),
            user_goal=str(payload.get("user_goal", "")),
            input_files=list(payload.get("input_files", [])),
            observations=[str(item) for item in payload.get("observations", [])],
        )
        for path, row in dict(payload.get("facts", {})).items():
            manifest.add_fact(_record_from_mapping(path, row))
        for row in payload.get("missing_facts", []):
            manifest.require_fact(
                str(row["path"]),
                reason=str(row["reason"]),
                question=str(row["question"]),
                acceptable_sources=row.get("acceptable_sources", []),
            )
        for path, row in dict(payload.get("algorithm_configuration", {})).items():
            manifest.add_algorithm_configuration(_record_from_mapping(path, row))
        return manifest


def _record_to_dict(record: FactRecord) -> dict[str, Any]:
    return {
        "value": record.value,
        "provenance": record.provenance.value,
        "source_ref": record.source_ref,
        "derivation": record.derivation,
        "user_confirmed": record.user_confirmed,
        "notes": record.notes,
    }


def _record_from_mapping(path: str, payload: Mapping[str, Any]) -> FactRecord:
    return FactRecord(
        path=path,
        value=payload.get("value"),
        provenance=ProvenanceKind(str(payload["provenance"])),
        source_ref=payload.get("source_ref"),
        derivation=payload.get("derivation"),
        user_confirmed=bool(payload.get("user_confirmed", False)),
        notes=payload.get("notes"),
    )
