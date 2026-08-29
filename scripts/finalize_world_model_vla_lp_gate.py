#!/usr/bin/env python3  # Execute this one-shot local-prediction evaluation finalizer with the repository interpreter.
"""Add an independent LP baseline and explicit overtake gate to the bridge campaign."""  # State the exact scientific edit applied by this temporary script.

from __future__ import annotations  # Enable postponed annotation evaluation.

from pathlib import Path  # Import portable repository-path handling.


def main() -> None:  # Patch the held-out campaign without altering the WM-VLA controller.
    root = Path(__file__).resolve().parents[1]  # Resolve the checked-out repository root.
    path = root / "scripts" / "run_world_model_vla_bridge.py"  # Select the canonical bridge campaign source.
    text = path.read_text(encoding="utf-8")  # Read the exact branch source.
    import_marker = "from visionamr.indicators import zz_indicator  # Import the shared element-wise ZZ estimator.\n"  # Identify the stable baseline import insertion point.
    lp_import = "from visionamr.local_prediction_runner import run_local_prediction_multistep  # Import the independent repeated local-prediction baseline.\n"  # Define the evaluation-only LP import.
    if lp_import not in text:  # Insert the independent baseline import only once.
        if import_marker not in text:  # Reject an incompatible campaign revision.
            raise RuntimeError("campaign indicator import marker was not found")  # Stop before a speculative edit.
        text = text.replace(import_marker, import_marker + lp_import, 1)  # Add the LP runner beside other evaluation methods.
    parser_marker = '    parser.add_argument("--allow-weaker", action="store_true")  # Permit exploratory artifacts without enforcing the Dörfler performance gate.\n'  # Identify the stable acceptance-option insertion point.
    parser_new = parser_marker + '    parser.add_argument("--require-lp-overtake", action="store_true")  # Require WM-VLA to be LP-noninferior on all tests and advantageous on at least one.\n'  # Define the explicit strong LP acceptance option.
    if "--require-lp-overtake" not in text:  # Add the option only once.
        if parser_marker not in text:  # Reject an incompatible parser revision.
            raise RuntimeError("campaign parser marker was not found")  # Stop before a speculative edit.
        text = text.replace(parser_marker, parser_new, 1)  # Add the explicit LP overtake gate.
    aggregate_marker = "    any_world_advantage = False  # Initialize the aggregate world-model advantage flag.\n"  # Identify the stable aggregate-state insertion point.
    aggregate_new = aggregate_marker + "    all_lp_noninferior = True  # Initialize the aggregate independent-LP noninferiority gate.\n    any_lp_overtake = False  # Initialize the aggregate independent-LP advantage flag.\n"  # Define the two LP aggregate acceptance states.
    if "all_lp_noninferior = True" not in text:  # Insert the aggregate states only once.
        if aggregate_marker not in text:  # Reject an incompatible campaign loop revision.
            raise RuntimeError("campaign aggregate marker was not found")  # Stop before a speculative edit.
        text = text.replace(aggregate_marker, aggregate_new, 1)  # Add independent-LP aggregate tracking.
    runner_marker = '        world_runner = _make_runner(problem, args.output / "test" / f"case_{index:02d}" / "world")  # Isolate the world-model execution.\n'  # Identify the stable world-runner creation point.
    runner_new = '        lp_runner = _make_runner(problem, args.output / "test" / f"case_{index:02d}" / "local_prediction")  # Isolate the independent local-prediction execution.\n        lp_summary = run_local_prediction_multistep(lp_runner, args.n_eq_cap, max_solves=config.max_solves, gradation=config.gradation, budget_safety=config.budget_safety, exponent=2.5, require_reference=config.require_reference, method="local_prediction")  # Run the repository LP formula without exposing it to WM-VLA.\n' + runner_marker  # Define the independent baseline execution before WM-VLA.
    if "lp_summary = run_local_prediction_multistep" not in text:  # Insert the LP execution only once.
        if runner_marker not in text:  # Reject an incompatible held-out loop revision.
            raise RuntimeError("world runner marker was not found")  # Stop before a speculative edit.
        text = text.replace(runner_marker, runner_new, 1)  # Add the isolated LP runner and trajectory.
    points_marker = '        world_points = _method_points(world_runner, "wm_vla")  # Normalize the world-model trajectory.\n'  # Identify the stable world-point normalization line.
    points_new = '        lp_points = _method_points(lp_runner, "local_prediction")  # Normalize the independent local-prediction trajectory.\n' + points_marker  # Define LP normalization immediately before WM normalization.
    if 'lp_points = _method_points(lp_runner, "local_prediction")' not in text:  # Insert LP point normalization only once.
        if points_marker not in text:  # Reject an incompatible held-out result layout.
            raise RuntimeError("world point marker was not found")  # Stop before a speculative edit.
        text = text.replace(points_marker, points_new, 1)  # Add the independent LP trajectory points.
    comparison_marker = '        comparison = compare_to_dorfler(dorfler_points, world_points)  # Enforce the same-solves and same-equations performance floor.\n'  # Identify the stable Dörfler comparison line.
    comparison_new = comparison_marker + '        lp_comparison = compare_to_dorfler(lp_points, world_points)  # Compare WM-VLA against the independent LP best-so-far Pareto envelope.\n'  # Define the independent LP comparison.
    if "lp_comparison = compare_to_dorfler" not in text:  # Insert the LP comparison only once.
        if comparison_marker not in text:  # Reject an incompatible comparison layout.
            raise RuntimeError("Dörfler comparison marker was not found")  # Stop before a speculative edit.
        text = text.replace(comparison_marker, comparison_new, 1)  # Add the independent LP comparison.
    update_marker = '        any_world_advantage = any_world_advantage or bool(comparison.get("world_advantage", False))  # Record whether a held-out case demonstrates world-model value.\n'  # Identify the stable Dörfler aggregate update.
    update_new = update_marker + '        all_lp_noninferior = all_lp_noninferior and bool(lp_comparison["not_weaker"])  # Require WM-VLA not to fall below the LP envelope on any held-out case.\n        any_lp_overtake = any_lp_overtake or bool(lp_comparison.get("world_advantage", False))  # Require at least one held-out LP overtake mechanism.\n'  # Define aggregate independent-LP acceptance updates.
    if "all_lp_noninferior = all_lp_noninferior" not in text:  # Insert aggregate updates only once.
        if update_marker not in text:  # Reject an incompatible aggregate update layout.
            raise RuntimeError("world advantage update marker was not found")  # Stop before a speculative edit.
        text = text.replace(update_marker, update_new, 1)  # Add the independent LP aggregate gates.
    append_old = '        tests.append({"index": index, "problem": problem.params, "dorfler": dorfler_summary, "world": asdict(world_summary), "dorfler_points": dorfler_points, "world_points": world_points, "comparison": comparison})  # Preserve the complete held-out audit record.\n'  # Identify the original per-case audit payload.
    append_new = '        tests.append({"index": index, "problem": problem.params, "dorfler": dorfler_summary, "local_prediction": asdict(lp_summary), "world": asdict(world_summary), "dorfler_points": dorfler_points, "local_prediction_points": lp_points, "world_points": world_points, "comparison": comparison, "local_prediction_comparison": lp_comparison})  # Preserve all independent trajectories and comparisons without method leakage.\n'  # Define the expanded per-case audit payload.
    if append_old in text:  # Replace the original payload exactly once.
        text = text.replace(append_old, append_new, 1)  # Add LP evidence to each held-out case.
    elif '"local_prediction_comparison": lp_comparison' not in text:  # Reject an unknown per-case payload revision.
        raise RuntimeError("held-out audit payload was not found")  # Surface the incompatible source location.
    payload_old = '    payload = {"schema": 1, "seed": args.seed, "config": asdict(config), "n_eq_cap": args.n_eq_cap, "training": training, "tests": tests, "world_model_rows": world.sample_count, "world_model_snapshot": str(model_path), "dorfler_floor_pass": bool(all_floor_pass), "world_model_advantage_observed": bool(any_world_advantage), "local_prediction_used": False}  # Build the final machine-readable campaign result.\n'  # Identify the original campaign payload.
    payload_new = '    payload = {"schema": 1, "seed": args.seed, "config": asdict(config), "n_eq_cap": args.n_eq_cap, "training": training, "tests": tests, "world_model_rows": world.sample_count, "world_model_snapshot": str(model_path), "dorfler_floor_pass": bool(all_floor_pass), "world_model_advantage_observed": bool(any_world_advantage), "local_prediction_noninferior_all": bool(all_lp_noninferior), "local_prediction_overtaken_observed": bool(any_lp_overtake), "local_prediction_overtake_gate": bool(all_lp_noninferior and any_lp_overtake), "local_prediction_used": False, "local_prediction_baseline_executed": True}  # Build the final machine-readable campaign result with independent LP evidence.\n'  # Define the expanded campaign payload.
    if payload_old in text:  # Replace the original payload exactly once.
        text = text.replace(payload_old, payload_new, 1)  # Add independent LP aggregate results.
    elif '"local_prediction_overtake_gate"' not in text:  # Reject an unknown campaign payload revision.
        raise RuntimeError("campaign payload marker was not found")  # Surface the incompatible source location.
    print_old = '    print(json.dumps({"result": str(result_path), "dorfler_floor_pass": all_floor_pass, "world_model_advantage_observed": any_world_advantage, "world_model_rows": world.sample_count}, indent=1))  # Print only the concise final gate summary.\n'  # Identify the original terminal summary.
    print_new = '    print(json.dumps({"result": str(result_path), "dorfler_floor_pass": all_floor_pass, "world_model_advantage_observed": any_world_advantage, "local_prediction_noninferior_all": all_lp_noninferior, "local_prediction_overtaken_observed": any_lp_overtake, "local_prediction_overtake_gate": all_lp_noninferior and any_lp_overtake, "world_model_rows": world.sample_count}, indent=1))  # Print the concise Dörfler and independent-LP gate summary.\n'  # Define the expanded terminal summary.
    if print_old in text:  # Replace the original terminal summary exactly once.
        text = text.replace(print_old, print_new, 1)  # Add LP acceptance fields to stdout.
    elif '"local_prediction_overtake_gate": all_lp_noninferior and any_lp_overtake' not in text:  # Reject an unknown summary revision.
        raise RuntimeError("campaign print marker was not found")  # Surface the incompatible source location.
    exit_marker = '    if not all_floor_pass and not args.allow_weaker:  # Enforce the user\'s non-weaker requirement by default.\n        return 2  # Fail the campaign when WM-VLA falls below the Dörfler Pareto envelope.\n    return 0  # Report successful execution and gate completion.\n'  # Identify the original terminal acceptance block.
    exit_new = '    if not all_floor_pass and not args.allow_weaker:  # Enforce the Dörfler non-weaker requirement by default.\n        return 2  # Fail the campaign when WM-VLA falls below the Dörfler Pareto envelope.\n    if args.require_lp_overtake and not (all_lp_noninferior and any_lp_overtake):  # Enforce the explicit independent-LP overtake requirement when requested.\n        return 3  # Distinguish an LP overtake failure from a Dörfler-floor failure.\n    return 0  # Report successful execution and all requested gate completion.\n'  # Define the combined terminal acceptance block.
    if exit_marker in text:  # Replace the original block exactly once.
        text = text.replace(exit_marker, exit_new, 1)  # Add the optional strong LP overtake gate.
    elif "return 3  # Distinguish an LP overtake failure" not in text:  # Reject an unknown acceptance revision.
        raise RuntimeError("campaign exit marker was not found")  # Surface the incompatible source location.
    path.write_text(text, encoding="utf-8")  # Write the expanded independent-baseline campaign source.


if __name__ == "__main__":  # Execute only when launched as the one-shot finalizer.
    main()  # Apply the deterministic independent-LP evaluation patch.
