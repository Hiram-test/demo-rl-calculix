"""Deterministic non-learning baselines for the elastoplastic mesh-control task.

The baselines answer the central scientific objection to mesh RL: why not simply
refine the current stress or plastic-strain hotspot?  Every policy receives the
same one-action-per-load-increment budget as the DQN and is evaluated against
exactly the same fine-mesh reference path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Mapping

import numpy as np

from calculix_plastic_backend import PlasticPlateConfig, StateAwareCalculixPlasticEnv
from mesh_goal import GoalCondition
from state_aware_dqn_agent import ACTION_NAMES, KEEP, REFINE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plastic-config", required=True)
    parser.add_argument("--goal-file", required=True)
    parser.add_argument("--gmsh-cmd", required=True)
    parser.add_argument("--ccx-cmd", required=True)
    parser.add_argument("--simulations-root", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-mesh-size", type=float, default=0.06)
    parser.add_argument("--global-mesh-size", type=float, default=0.75)
    parser.add_argument("--cell-min-mesh-size", type=float, default=0.12)
    parser.add_argument("--cell-max-mesh-size", type=float, default=1.50)
    parser.add_argument("--min-elements", type=int, default=100)
    parser.add_argument("--max-elements", type=int, default=1500)
    parser.add_argument("--refine-step-size", type=float, default=0.18)
    parser.add_argument("--coarsen-step-size", type=float, default=0.18)
    parser.add_argument("--solver-timeout", type=int, default=300)
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> StateAwareCalculixPlasticEnv:
    env = StateAwareCalculixPlasticEnv(
        plate=PlasticPlateConfig.from_json(args.plastic_config),
        simulations_root=args.simulations_root,
        gmsh_cmd=args.gmsh_cmd,
        ccx_cmd=args.ccx_cmd,
        global_mesh_size=args.global_mesh_size,
        cell_min_mesh_size=args.cell_min_mesh_size,
        cell_max_mesh_size=args.cell_max_mesh_size,
        min_elements=args.min_elements,
        max_elements=args.max_elements,
        refine_step_size=args.refine_step_size,
        coarsen_step_size=args.coarsen_step_size,
        solver_timeout_seconds=args.solver_timeout,
    )
    env.compute_baseline(
        cache_dir=args.baseline_cache,
        use_cache=True,
        baseline_mesh_size=args.baseline_mesh_size,
    )
    return env


def keep_candidate(mask: Mapping[int, list[bool]]) -> tuple[int, int]:
    for cell_id, row in sorted(mask.items()):
        if len(row) > KEEP and row[KEEP]:
            return int(cell_id), KEEP
    raise RuntimeError("No valid KEEP candidate")


def choose_keep(env: StateAwareCalculixPlasticEnv, mask: Mapping[int, list[bool]]) -> tuple[int, int]:
    del env
    return keep_candidate(mask)


def choose_stress_hotspot(
    env: StateAwareCalculixPlasticEnv, mask: Mapping[int, list[bool]]
) -> tuple[int, int]:
    result = env._last_result
    if result is None:
        return keep_candidate(mask)
    candidates = [cell_id for cell_id, row in mask.items() if row[REFINE]]
    if not candidates:
        return keep_candidate(mask)
    stress_index = env.BASE_CELL_FEATURE_NAMES.index("max_mises_over_yield")
    scores = {
        cell_id: float(result.cell_features[cell_id][stress_index])
        for cell_id in candidates
    }
    if max(scores.values(), default=0.0) < 0.90:
        return keep_candidate(mask)
    return max(scores, key=scores.get), REFINE


def choose_peeq_hotspot(
    env: StateAwareCalculixPlasticEnv, mask: Mapping[int, list[bool]]
) -> tuple[int, int]:
    result = env._last_result
    if result is None:
        return keep_candidate(mask)
    candidates = [cell_id for cell_id, row in mask.items() if row[REFINE]]
    if not candidates:
        return keep_candidate(mask)
    max_value = max(result.cell_max_peeq.get(cell_id, 0.0) for cell_id in candidates)
    if max_value <= env.plate.plastic_threshold:
        return choose_stress_hotspot(env, mask)
    return max(candidates, key=lambda cell_id: result.cell_max_peeq.get(cell_id, 0.0)), REFINE


def choose_plastic_front(
    env: StateAwareCalculixPlasticEnv, mask: Mapping[int, list[bool]]
) -> tuple[int, int]:
    result = env._last_result
    if result is None:
        return keep_candidate(mask)
    candidates = [cell_id for cell_id, row in mask.items() if row[REFINE]]
    if not candidates:
        return keep_candidate(mask)
    if result.max_peeq <= env.plate.plastic_threshold:
        return choose_stress_hotspot(env, mask)

    def score(cell_id: int) -> float:
        own = float(result.cell_mean_peeq.get(cell_id, 0.0))
        jumps = [
            abs(own - float(result.cell_mean_peeq.get(neighbor, 0.0)))
            for neighbor in env.cell_adjacency.get(cell_id, [])
        ]
        # A small current-activity term resolves ties without turning this back
        # into a pure hotspot policy.
        return max(jumps, default=0.0) + 0.10 * own

    return max(candidates, key=score), REFINE


POLICIES: Dict[str, Callable[[StateAwareCalculixPlasticEnv, Mapping[int, list[bool]]], tuple[int, int]]] = {
    "keep_only": choose_keep,
    "stress_hotspot": choose_stress_hotspot,
    "peeq_hotspot": choose_peeq_hotspot,
    "plastic_front": choose_plastic_front,
}


def run_policy(
    env: StateAwareCalculixPlasticEnv,
    goal: GoalCondition,
    policy_name: str,
    chooser: Callable[[StateAwareCalculixPlasticEnv, Mapping[int, list[bool]]], tuple[int, int]],
) -> dict:
    env.set_goal(goal)
    env.reset(run_id=f"baseline_{policy_name}")
    path = []
    done = False
    while not done:
        mask = env.get_action_mask(goal=goal)
        cell_id, action = chooser(env, mask)
        _, reward, done, info = env.step({cell_id: action})
        path.append(
            {
                "load_step": int(info.get("load_step", env.load_step_index)),
                "load_fraction": float(info.get("load_fraction", env.load_fraction)),
                "cell_id": int(cell_id),
                "action": int(action),
                "action_name": ACTION_NAMES[action],
                "reward": float(reward),
                "resource_usage": float(env._extract_resource_usage(info)),
                "metrics": dict(info.get("error_metrics", {})),
                "reaction_force_x": info.get("reaction_force_x"),
                "plastic_zone_fraction": info.get("plastic_zone_fraction"),
                "max_peeq": info.get("max_peeq"),
            }
        )
    composite = [float(item["metrics"].get("composite_error", 0.0)) for item in path]
    resource = [float(item["resource_usage"]) for item in path]
    return {
        "policy": policy_name,
        "final_metrics": env.evaluation_metrics(),
        "path_mean_composite_error": float(np.mean(composite)),
        "path_max_composite_error": float(np.max(composite)),
        "path_integrated_resource": float(np.sum(resource)),
        "actions": path,
        "final_workdir": env._last_result.workdir if env._last_result else None,
    }


def main() -> None:
    args = parse_args()
    goal = GoalCondition.from_json(args.goal_file)
    results = []
    for name, chooser in POLICIES.items():
        env = make_env(args)
        record = run_policy(env, goal, name, chooser)
        results.append(record)
        print(
            f"[{name}] final_error={record['final_metrics']['composite_error']:.6f} "
            f"path_error={record['path_mean_composite_error']:.6f} "
            f"resource={record['final_metrics']['resource_usage']:.4f}",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
