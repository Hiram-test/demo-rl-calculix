#!/usr/bin/env python3  # Execute this one-shot scientific-split finalizer with the repository interpreter.
"""Freeze independent held-out WM-VLA evaluation from one training snapshot."""  # State the exact scientific correction applied by this temporary script.

from __future__ import annotations  # Enable postponed annotation evaluation.

from pathlib import Path  # Import portable repository-path handling.


def main() -> None:  # Apply the deterministic training and test isolation edits.
    root = Path(__file__).resolve().parents[1]  # Resolve the checked-out repository root.
    path = root / "scripts" / "run_world_model_vla_bridge.py"  # Select the held-out campaign runner.
    text = path.read_text(encoding="utf-8")  # Read the exact branch source.
    text = text.replace("world = RegionalWorldModel(config)  # Initialize one cross-case regional transition library.", "world = RegionalWorldModel(training_config)  # Initialize the independent training transition library with the exploration contract.")  # Align the learned prior and uncertainty settings with transition collection.
    old = 'world_summary = run_world_model_vla(world_runner, _make_partitioner(), args.n_eq_cap, config=config, model=world, method="wm_vla")  # Run the held-out multi-step controller.'  # Identify the state-sharing held-out call.
    new = 'test_world = RegionalWorldModel.load(model_path, config=config)  # Restore the same frozen training snapshot independently for this held-out case.\n        world_summary = run_world_model_vla(world_runner, _make_partitioner(), args.n_eq_cap, config=config, model=test_world, method="wm_vla")  # Run within-case online correction without cross-test leakage.'  # Define independent per-case evaluation.
    if old in text:  # Apply the correction only while the original state-sharing call remains.
        text = text.replace(old, new, 1)  # Replace exactly one held-out execution call.
    elif "test_world = RegionalWorldModel.load(model_path, config=config)" not in text:  # Reject an unknown campaign revision rather than guessing.
        raise RuntimeError("held-out world-model call was not found")  # Surface the incompatible source location.
    path.write_text(text, encoding="utf-8")  # Write the corrected scientific campaign source.


if __name__ == "__main__":  # Execute only when launched as the one-shot finalizer.
    main()  # Apply the deterministic held-out isolation edit.
