"""Flexible numerical evidence comparison for model-selected JSON paths."""
from __future__ import annotations

import re
from typing import Any, Mapping

from .contracts import SkillManifest, SkillResult

_SEGMENT_RE = re.compile(r"^(?P<key>[^\[]*)(?P<selectors>(?:\[[^\]]*\])*)$")
_SELECTOR_RE = re.compile(r"\[([^\]]*)\]")


def _segments(path: str) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for raw in path.split(".") if path else []:
        match = _SEGMENT_RE.fullmatch(raw)
        if not match:
            raise ValueError(f"invalid path segment {raw!r}")
        rows.append((match.group("key"), _SELECTOR_RE.findall(match.group("selectors"))))
    return rows


def flexible_path_values(payload: Any, path: str) -> list[Any]:
    """Resolve dotted paths, dotted JSON keys, list indices, and wildcards."""
    segments = _segments(path)

    def walk(item: Any, pos: int) -> list[Any]:
        if pos >= len(segments):
            return list(item) if isinstance(item, list) else [item]
        key, _selectors = segments[pos]
        if not isinstance(item, Mapping):
            raise KeyError(f"expected object before segment {key!r} in {path}")

        chosen: str | None = None
        consume = 0
        for end in range(pos, len(segments)):
            if end > pos and segments[end - 1][1]:
                break
            candidate = ".".join(segment[0] for segment in segments[pos : end + 1])
            if candidate in item:
                chosen = candidate
                consume = end - pos + 1
            if segments[end][1]:
                break
        if chosen is None:
            raise KeyError(
                f"path {path!r} not found at segment {key!r}; "
                f"available keys={list(item)[:30]}"
            )

        value = item[chosen]
        final_selectors = segments[pos + consume - 1][1]
        for selector in final_selectors:
            if selector in ("", "*"):
                if not isinstance(value, list):
                    raise TypeError(f"{chosen!r} is not a list in {path}")
                results: list[Any] = []
                for child in value:
                    results.extend(walk(child, pos + consume))
                return results
            if not isinstance(value, list):
                raise TypeError(f"{chosen!r} is not a list in {path}")
            value = value[int(selector)]
        return walk(value, pos + consume)

    return walk(payload, 0)


def compare_evidence(
    context: Any, arguments: dict[str, Any], manifest: SkillManifest
) -> SkillResult:
    del manifest
    left_id = str(arguments["left_artifact"])
    right_id = str(arguments["right_artifact"])
    left = context.read_artifact_json(left_id)
    right = context.read_artifact_json(right_id)
    metrics: list[dict[str, Any]] = []

    try:
        for query in arguments["metric_queries"]:
            name = str(query["name"])
            common_path = query.get("path")
            left_path = query.get("left_path") or common_path
            right_path = query.get("right_path") or common_path
            if not left_path or not right_path:
                raise ValueError(
                    f"metric {name} requires path or both left_path and right_path"
                )
            left_values = flexible_path_values(left, str(left_path))
            right_values = flexible_path_values(right, str(right_path))
            operation = str(query.get("operation", "series"))
            row: dict[str, Any] = {
                "name": name,
                "operation": operation,
                "left_path": str(left_path),
                "right_path": str(right_path),
                "left_values": left_values,
                "right_values": right_values,
            }
            if operation in {"difference", "ratio", "relative_difference"}:
                if len(left_values) != len(right_values):
                    raise ValueError(f"metric {name} requires equally sized series")
                numeric_left = [float(value) for value in left_values]
                numeric_right = [float(value) for value in right_values]
                if operation == "difference":
                    row["result"] = [
                        right_value - left_value
                        for left_value, right_value in zip(numeric_left, numeric_right)
                    ]
                elif operation == "ratio":
                    row["result"] = [
                        right_value / left_value if left_value != 0 else None
                        for left_value, right_value in zip(numeric_left, numeric_right)
                    ]
                else:
                    row["result"] = [
                        abs(right_value - left_value) / max(abs(left_value), 1.0e-15)
                        for left_value, right_value in zip(numeric_left, numeric_right)
                    ]
            metrics.append(row)
    except (KeyError, TypeError, ValueError) as exc:
        return SkillResult(
            skill_name="compare_evidence",
            status="invalid_arguments",
            summary="A requested numerical evidence path could not be resolved.",
            errors=[str(exc)],
            metadata={"left_artifact": left_id, "right_artifact": right_id},
        )

    comparison = {
        "kind": "evidence_comparison",
        "left_artifact": left_id,
        "right_artifact": right_id,
        "metrics": metrics,
        "comparison_purpose": arguments["comparison_purpose"],
        "interpretations": [],
    }
    artifact = context.add_json_artifact(
        kind="comparison",
        stem="comparison",
        payload=comparison,
        summary="Requested numerical comparisons with no embedded physical classification.",
        metadata={"left_artifact": left_id, "right_artifact": right_id},
    )
    return SkillResult(
        skill_name="compare_evidence",
        status="completed",
        summary=f"Computed {len(metrics)} evidence comparisons.",
        artifacts=[artifact],
        observations=[
            {
                "observation": "Numerical evidence sets were compared.",
                "artifact_ref": artifact.artifact_id,
                "metric_names": [row["name"] for row in metrics],
            }
        ],
    )
