#!/usr/bin/env python3  # Execute this one-shot source finalizer with the repository interpreter.
"""Apply idempotent WM-VLA contract fixes before the final solver-backed gate."""  # State the narrow maintenance purpose of this temporary script.

from __future__ import annotations  # Enable postponed annotation evaluation.

from pathlib import Path  # Import portable repository-path handling.


def _replace_once(text: str, old: str, new: str, label: str) -> str:  # Replace one known source fragment without silently duplicating it.
    if new in text:  # Detect an already finalized source tree.
        return text  # Preserve the idempotent final form.
    if old not in text:  # Reject an unknown source revision rather than applying a speculative patch.
        raise RuntimeError(f"missing expected source fragment: {label}")  # Surface the exact incompatible fragment.
    return text.replace(old, new, 1)  # Apply exactly one deterministic replacement.


def _finalize_controller(root: Path) -> None:  # Harden the canonical controller's budget and constructor contracts.
    path = root / "visionamr" / "vla" / "world_controller.py"  # Select the canonical multi-step controller source.
    text = path.read_text(encoding="utf-8")  # Read the exact branch source.
    text = text.replace("if not receipt.budget_pass and receipt.fallback_used:", "if not receipt.budget_pass:")  # Stop every uncertified over-budget next mesh, including pure Dörfler.
    import_old = "import hashlib  # Import hashing for immutable action and mesh receipts.\n"  # Identify the stable import insertion point.
    import_new = "import hashlib  # Import hashing for immutable action and mesh receipts.\nimport inspect  # Import constructor signature inspection for repository compatibility.\n"  # Add one standard-library compatibility dependency.
    text = _replace_once(text, import_old, import_new, "controller inspect import")  # Insert signature inspection exactly once.
    helper_marker = "\n\ndef semantic_persistence(name: str) -> float:"  # Identify the first stable helper insertion point.
    helper = '''\n\ndef _make_partition(seeds, problem, gradation: float, drawings):  # Construct the fixed semantic partition across repository constructor revisions.\n    parameters = inspect.signature(Partition).parameters  # Read the installed partition constructor contract.\n    kwargs: dict[str, object] = {}  # Allocate only keyword arguments supported by this repository revision.\n    if "gradation" in parameters:  # Supply the deterministic size-gradation limit when supported.\n        kwargs["gradation"] = float(gradation)  # Store the normalized gradation parameter.\n    if "assign_mode" in parameters:  # Select geometry-drawing assignment when the constructor exposes that switch.\n        kwargs["assign_mode"] = "drawn"  # Freeze the semantic partition to the vision drawings.\n    if "drawings" in parameters:  # Supply cached drawings only when accepted by the constructor.\n        kwargs["drawings"] = list(drawings)  # Copy the drawing sequence to avoid later mutation.\n    return Partition(seeds, problem, **kwargs)  # Return the repository-compatible immutable semantic partition.\n'''
    if "def _make_partition(" not in text:  # Insert the compatibility helper only once.
        if helper_marker not in text:  # Reject an unexpected controller layout.
            raise RuntimeError("missing controller helper insertion point")  # Stop before a speculative source edit.
        text = text.replace(helper_marker, helper + helper_marker, 1)  # Insert the constructor adapter before model helpers.
    direct = 'partition = Partition(seeds, problem, gradation=cfg.gradation, assign_mode="drawn", drawings=drawings)  # Freeze the semantic partition across the adaptive trajectory.'  # Identify the original constructor call.
    adapted = 'partition = _make_partition(seeds, problem, cfg.gradation, drawings)  # Freeze the semantic partition through the repository-compatible constructor adapter.'  # Define the compatibility-safe constructor call.
    text = text.replace(direct, adapted)  # Replace the call when the original form remains.
    path.write_text(text, encoding="utf-8")  # Write the finalized canonical controller source.


def _finalize_bridge(root: Path) -> None:  # Remove positional ambiguity from bridge semantic anchors.
    path = root / "visionamr" / "bridge_diaphragm.py"  # Select the medium-complexity bridge benchmark source.
    text = path.read_text(encoding="utf-8")  # Read the exact branch source.
    import_old = "from __future__ import annotations  # Enable postponed evaluation of annotations.\n"  # Identify the stable import insertion point.
    import_new = "from __future__ import annotations  # Enable postponed evaluation of annotations.\n\nimport inspect  # Import signature inspection for FeatureAnchor compatibility.\n"  # Add one standard-library compatibility dependency.
    text = _replace_once(text, import_old, import_new, "bridge inspect import")  # Insert signature inspection exactly once.
    marker = "\n\ndef _box_predicate("  # Identify the first stable bridge helper insertion point.
    helper = '''\n\ndef _feature_anchor(name: str, x_value: float, y_value: float, z_value: float, kind: str, radius: float = 180.0) -> FeatureAnchor:  # Construct one semantic anchor across repository signature revisions.\n    parameters = inspect.signature(FeatureAnchor).parameters  # Read the installed immutable anchor constructor contract.\n    kwargs: dict[str, object] = {}  # Allocate keyword arguments without relying on positional field order.\n    point = (float(x_value), float(y_value), float(z_value))  # Normalize the three-dimensional anchor location.\n    if "name" in parameters:  # Supply the stable semantic region name when supported.\n        kwargs["name"] = str(name)  # Store the semantic name explicitly.\n    coordinate_key = next((key for key in ("xyz", "point", "coord", "coords", "location") if key in parameters), None)  # Discover the repository coordinate-field spelling.\n    if coordinate_key is not None:  # Prefer the compact coordinate tuple contract.\n        kwargs[coordinate_key] = point  # Store the complete anchor point.\n    else:  # Support a legacy constructor exposing separate Cartesian fields.\n        for key, value in zip(("x", "y", "z"), point):  # Visit each Cartesian coordinate deterministically.\n            if key in parameters:  # Supply only fields accepted by the installed contract.\n                kwargs[key] = value  # Store the corresponding coordinate value.\n    if "kind" in parameters:  # Supply the structural semantic class when supported.\n        kwargs["kind"] = str(kind)  # Store the semantic class explicitly.\n    elif "role" in parameters:  # Support the equivalent legacy role spelling.\n        kwargs["role"] = str(kind)  # Store the semantic class under the legacy field name.\n    if "r" in parameters:  # Supply the anchor influence radius used by regional assignment.\n        kwargs["r"] = float(radius)  # Store the normalized influence radius.\n    elif "radius" in parameters:  # Support an explicit radius field spelling.\n        kwargs["radius"] = float(radius)  # Store the normalized influence radius.\n    return FeatureAnchor(**kwargs)  # Construct the immutable repository anchor without positional ambiguity.\n'''
    if "def _feature_anchor(" not in text:  # Insert the compatibility helper only once.
        if marker not in text:  # Reject an unexpected benchmark layout.
            raise RuntimeError("missing bridge helper insertion point")  # Stop before a speculative source edit.
        text = text.replace(marker, helper + marker, 1)  # Insert the anchor adapter before geometry predicates.
    text = text.replace("features = [FeatureAnchor(", "features = [_feature_anchor(")  # Replace the first positional anchor construction.
    text = text.replace(', FeatureAnchor("', ', _feature_anchor("')  # Replace all remaining positional anchor constructions.
    path.write_text(text, encoding="utf-8")  # Write the finalized bridge benchmark source.


def main() -> None:  # Apply both idempotent contract fixes from the repository root.
    root = Path(__file__).resolve().parents[1]  # Resolve the checked-out repository root deterministically.
    _finalize_controller(root)  # Harden controller budget and partition construction.
    _finalize_bridge(root)  # Harden bridge semantic-anchor construction.


if __name__ == "__main__":  # Execute only when launched as the one-shot maintenance script.
    main()  # Apply the deterministic finalization edits.
