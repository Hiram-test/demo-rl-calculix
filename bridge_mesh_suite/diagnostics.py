from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class SkillRecord:
    name: str
    purpose: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    passed: bool


@dataclass
class DiagnosticResult:
    user_question: str
    diagnosis: str
    plain_explanation: str
    applied_plan: list[str]
    supported_use: list[str]
    unsupported_use: list[str]
    evidence: dict[str, Any]
    skill_trace: list[SkillRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def relative_change(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0e-30)
    return abs(a - b) / denom


def energy_consistency_skill(level_rows: list[dict[str, Any]], *, tolerance: float = 1.0e-8) -> SkillRecord:
    balances = [float(row["energy_balance_rel"]) for row in level_rows]
    energies = [float(row["strain_energy"]) for row in level_rows]
    mesh_change = relative_change(energies[-1], energies[-2]) if len(energies) >= 2 else float("nan")
    passed = bool(max(balances, default=0.0) <= tolerance)
    return SkillRecord(
        name="energy_consistency",
        purpose="确认每档网格满足线性静力能量平衡，并判断整体柔度是否已稳定。",
        inputs={"levels": len(level_rows), "balance_tolerance": tolerance},
        outputs={
            "balance_relative_errors": balances,
            "strain_energies": energies,
            "last_mesh_energy_change": mesh_change,
        },
        passed=passed,
    )


def theory_cross_check_skill(
    *,
    name: str,
    numerical: float,
    theoretical: float,
    tolerance: float,
    quantity: str,
) -> SkillRecord:
    error = relative_change(numerical, theoretical)
    return SkillRecord(
        name="theory_cross_check",
        purpose=f"用{name}对有限元的{quantity}进行独立校核。",
        inputs={"theory": name, "quantity": quantity, "tolerance": tolerance},
        outputs={"numerical": numerical, "theoretical": theoretical, "relative_error": error},
        passed=bool(error <= tolerance),
    )


def peak_growth_skill(
    level_rows: list[dict[str, Any]],
    *,
    peak_key: str,
    stable_keys: Iterable[str],
    peak_growth_threshold: float = 0.08,
    stable_change_threshold: float = 0.05,
) -> SkillRecord:
    peak_last = float(level_rows[-1][peak_key])
    peak_prev = float(level_rows[-2][peak_key])
    peak_growth = (peak_last - peak_prev) / max(abs(peak_prev), 1.0e-30)
    stable_changes = {
        key: relative_change(float(level_rows[-1][key]), float(level_rows[-2][key])) for key in stable_keys
    }
    localized = peak_growth > peak_growth_threshold and all(
        value <= stable_change_threshold for value in stable_changes.values()
    )
    return SkillRecord(
        name="peak_growth_diagnostic",
        purpose="区分‘整体仍未收敛’与‘局部峰值因理想化或奇异性持续增长’。",
        inputs={
            "peak_key": peak_key,
            "stable_keys": list(stable_keys),
            "peak_growth_threshold": peak_growth_threshold,
            "stable_change_threshold": stable_change_threshold,
        },
        outputs={
            "last_peak_growth": peak_growth,
            "stable_quantity_changes": stable_changes,
            "localized_peak_growth": localized,
        },
        passed=True,
    )


def physical_variant_skill(
    *,
    baseline: dict[str, Any],
    variant: dict[str, Any],
    peak_key: str,
    preserved_keys: Iterable[str],
    description: str,
    preserved_tolerance: float = 0.08,
) -> SkillRecord:
    peak_ratio = float(variant[peak_key]) / max(abs(float(baseline[peak_key])), 1.0e-30)
    preserved_changes = {
        key: relative_change(float(variant[key]), float(baseline[key])) for key in preserved_keys
    }
    passed = peak_ratio < 1.0 and all(v <= preserved_tolerance for v in preserved_changes.values())
    return SkillRecord(
        name="physical_variant_check",
        purpose=description,
        inputs={"peak_key": peak_key, "preserved_keys": list(preserved_keys)},
        outputs={"peak_ratio_variant_to_baseline": peak_ratio, "preserved_quantity_changes": preserved_changes},
        passed=passed,
    )


def contour_spread(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float((arr.max() - arr.min()) / max(abs(arr.mean()), 1.0e-30))
