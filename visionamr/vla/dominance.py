# Pre-registered matched-cost dominance gates for WM-VLA versus exact Dörfler AFEM.  # Module purpose.
from __future__ import annotations  # Enable postponed annotations for record-like inputs.
from dataclasses import dataclass  # Provide a compact typed gate configuration.
from typing import Any, Iterable  # Accept SolveRecord objects or JSON-compatible record dictionaries.
import math  # Compute stable logarithmic curve comparisons.


@dataclass  # Configure the release criterion without tuning it after seeing results.
class DominanceConfig:  # Define what "not weaker than Dörfler" means operationally.
    energy_final_tolerance: float = 0.02  # Allow at most two percent final energy-error degradation.
    energy_curve_tolerance: float = 0.03  # Allow at most three percent geometric-mean curve degradation.
    qoi_final_tolerance: float = 0.05  # Allow a wider five percent QoI-error tolerance.
    minimum_common_solves: int = 2  # Require more than one matched real-solve point.
    require_all_budget_feasible: bool = True  # Reject a controller that crosses the declared equation cap.


def _value(record: Any, name: str, default: Any = None) -> Any:  # Read fields uniformly from dataclasses and JSON dictionaries.
    return record.get(name, default) if isinstance(record, dict) else getattr(record, name, default)  # Preserve both repository representations.


def _method_records(records: Iterable[Any], method: str, budget: int) -> list[Any]:  # Select and order one independent method trajectory.
    selected = [record for record in records if str(_value(record, "method", "")) == str(method)]  # Filter by exact method provenance.
    selected.sort(key=lambda record: int(_value(record, "solve_index", 0)))  # Restore honest expensive-solve order.
    return selected  # Return the complete trajectory including over-budget diagnostics.


def _best_feasible_prefix(records: list[Any], metric: str, budget: int, max_k: int) -> list[dict[str, Any] | None]:  # Build a hold-last best measured envelope.
    output: list[dict[str, Any] | None] = []  # Accumulate one envelope point per real solve count.
    best: dict[str, Any] | None = None  # Hold the best feasible measured point seen so far.
    for k in range(1, int(max_k) + 1):  # Traverse matched expensive-solve counts.
        eligible = [record for record in records if int(_value(record, "solve_index", 0)) <= k and int(_value(record, "n_equations", 0)) <= int(budget) and _value(record, metric, None) is not None]  # Restrict to measured feasible prefixes.
        if eligible:  # Update the envelope only from real feasible results.
            chosen = min(eligible, key=lambda record: float(_value(record, metric)))  # Select the lowest measured error.
            best = {"k": k, "error": float(_value(chosen, metric)), "n_equations": int(_value(chosen, "n_equations")), "source_solve_index": int(_value(chosen, "solve_index"))}  # Preserve the selected real point.
        output.append(None if best is None else dict(best))  # Hold the best point forward for fair solve-budget comparison.
    return output  # Return the complete measured envelope.


def _geometric_curve_ratio(wm_curve: list[dict[str, Any] | None], dorfler_curve: list[dict[str, Any] | None]) -> tuple[float | None, list[dict[str, float]]]:  # Compare matched-solve error curves robustly.
    logs: list[float] = []  # Accumulate logarithmic error ratios.
    rows: list[dict[str, float]] = []  # Preserve every common matched point.
    for index, (wm_point, dorfler_point) in enumerate(zip(wm_curve, dorfler_curve), start=1):  # Traverse aligned solve counts.
        if wm_point is None or dorfler_point is None:  # Skip counts before either method has a feasible measured point.
            continue  # Move to the next matched count.
        wm_error = max(float(wm_point["error"]), 1.0e-15)  # Bound the WM error away from zero.
        dorfler_error = max(float(dorfler_point["error"]), 1.0e-15)  # Bound the Dörfler error away from zero.
        ratio = wm_error / dorfler_error  # Compute the pointwise matched-solve degradation ratio.
        logs.append(math.log(ratio))  # Accumulate logarithms for a geometric mean.
        rows.append({"k": float(index), "wm": wm_error, "dorfler": dorfler_error, "ratio": ratio})  # Preserve the transparent comparison row.
    if not logs:  # Handle trajectories without a common feasible point.
        return None, rows  # Return an unavailable curve ratio explicitly.
    return float(math.exp(sum(logs) / len(logs))), rows  # Return the geometric-mean matched-solve ratio.


def evaluate_dorfler_floor(records: Iterable[Any], wm_method: str, dorfler_method: str, budget: int, config: DominanceConfig | None = None) -> dict[str, Any]:  # Evaluate the pre-registered non-inferiority release gate.
    cfg = config or DominanceConfig()  # Use explicit or default tolerances.
    wm_records = _method_records(records, wm_method, budget)  # Read the independent WM-VLA trajectory.
    dorfler_records = _method_records(records, dorfler_method, budget)  # Read the independent exact Dörfler trajectory.
    max_k = min(len(wm_records), len(dorfler_records))  # Restrict comparisons to common expensive-solve counts.
    energy_wm = _best_feasible_prefix(wm_records, "e_energy", budget, max_k)  # Build the WM energy-error envelope.
    energy_dorfler = _best_feasible_prefix(dorfler_records, "e_energy", budget, max_k)  # Build the Dörfler energy-error envelope.
    qoi_wm = _best_feasible_prefix(wm_records, "e_qoi", budget, max_k)  # Build the WM QoI-error envelope.
    qoi_dorfler = _best_feasible_prefix(dorfler_records, "e_qoi", budget, max_k)  # Build the Dörfler QoI-error envelope.
    curve_ratio, curve_rows = _geometric_curve_ratio(energy_wm, energy_dorfler)  # Compare the complete common energy curve.
    common_energy = [row for row in curve_rows if row]  # Count actual common measured points.
    final_energy_wm = next((point for point in reversed(energy_wm) if point is not None), None)  # Read the final common WM envelope point.
    final_energy_dorfler = next((point for point in reversed(energy_dorfler) if point is not None), None)  # Read the final common Dörfler envelope point.
    final_qoi_wm = next((point for point in reversed(qoi_wm) if point is not None), None)  # Read the final common WM QoI point.
    final_qoi_dorfler = next((point for point in reversed(qoi_dorfler) if point is not None), None)  # Read the final common Dörfler QoI point.
    final_energy_ratio = None if final_energy_wm is None or final_energy_dorfler is None else float(final_energy_wm["error"] / max(float(final_energy_dorfler["error"]), 1.0e-15))  # Compute final energy non-inferiority.
    final_qoi_ratio = None if final_qoi_wm is None or final_qoi_dorfler is None else float(final_qoi_wm["error"] / max(float(final_qoi_dorfler["error"]), 1.0e-15))  # Compute final QoI non-inferiority.
    common_pass = len(common_energy) >= int(cfg.minimum_common_solves)  # Require enough matched evidence for release.
    final_energy_pass = final_energy_ratio is not None and final_energy_ratio <= 1.0 + float(cfg.energy_final_tolerance)  # Apply the pre-registered final energy margin.
    curve_pass = curve_ratio is not None and curve_ratio <= 1.0 + float(cfg.energy_curve_tolerance)  # Apply the pre-registered full-curve margin.
    qoi_pass = final_qoi_ratio is not None and final_qoi_ratio <= 1.0 + float(cfg.qoi_final_tolerance)  # Apply the pre-registered final QoI margin.
    wm_budget_ok = all(int(_value(record, "n_equations", 0)) <= int(budget) for record in wm_records) if wm_records else False  # Verify every executed WM mesh against the hard cap.
    budget_pass = wm_budget_ok if cfg.require_all_budget_feasible else True  # Apply or waive the strict budget condition explicitly.
    passed = bool(common_pass and final_energy_pass and curve_pass and qoi_pass and budget_pass)  # Combine all declared release conditions.
    return {  # Return a complete machine-readable gate report.
        "pass": passed,  # State whether WM-VLA may be claimed non-inferior to Dörfler.
        "wm_method": str(wm_method),  # Preserve exact WM provenance.
        "dorfler_method": str(dorfler_method),  # Preserve exact classical provenance.
        "budget": int(budget),  # Preserve the shared hard resource cap.
        "common_solves": int(max_k),  # Report the matched expensive-solve horizon.
        "common_energy_points": int(len(common_energy)),  # Report actual evidence depth.
        "final_energy_ratio": final_energy_ratio,  # Report WM divided by Dörfler final energy error.
        "energy_curve_geometric_ratio": curve_ratio,  # Report WM divided by Dörfler curve error.
        "final_qoi_ratio": final_qoi_ratio,  # Report WM divided by Dörfler final QoI error.
        "checks": {"common": common_pass, "final_energy": final_energy_pass, "energy_curve": curve_pass, "final_qoi": qoi_pass, "budget": budget_pass},  # Disclose every gate component.
        "tolerances": dict(cfg.__dict__),  # Preserve the pre-registered margins.
        "energy_curve": curve_rows,  # Preserve every matched measured comparison.
        "wm_energy_envelope": energy_wm,  # Preserve the WM hold-last envelope.
        "dorfler_energy_envelope": energy_dorfler,  # Preserve the Dörfler hold-last envelope.
        "wm_qoi_envelope": qoi_wm,  # Preserve the WM QoI envelope.
        "dorfler_qoi_envelope": qoi_dorfler,  # Preserve the Dörfler QoI envelope.
    }  # Finish the release-gate report.
