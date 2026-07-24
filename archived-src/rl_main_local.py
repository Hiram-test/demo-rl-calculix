"""Unified local runner for the Abaqus and Gmsh/CalculiX mesh-RL backends.

Examples
--------
CalculiX preflight and one solve::

    python rl_main_local.py --backend calculix --mode preflight
    python rl_main_local.py --backend calculix --mode solve \
        --plate-config examples/calculix_plate.json

Abaqus preflight and training::

    python rl_main_local.py --backend abaqus --mode preflight \
        --abaqus-cmd "C:/SIMULIA/Commands/abaqus.bat"
    python rl_main_local.py --backend abaqus --mode train \
        --template-cae-file DEMO.cae --max-episodes 10
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch

from calculix_backend import (
    PlateConfig,
    StateAwareCalculixEnv,
    command_available,
    split_command,
)
from calculix_plastic_backend import (
    PlasticPlateConfig,
    StateAwareCalculixPlasticEnv,
)
from mesh_goal import GoalCondition
from state_aware_dqn_agent import ACTION_NAMES, GraphState, ReplayBufferV2, StateAwareDQNAgent


@dataclass
class PendingTransition:
    state: GraphState
    action_node: int
    action_type: int
    reward: float
    next_state: GraphState
    done: bool
    cell_id: int


class NStepAccumulator:
    """Convert a stream of one-step transitions into replay-safe n-step returns."""

    def __init__(self, replay: ReplayBufferV2, n_steps: int, gamma: float) -> None:
        self.replay = replay
        self.n_steps = max(1, int(n_steps))
        self.gamma = float(gamma)
        self.pending: Deque[PendingTransition] = deque()

    def append(self, transition: PendingTransition) -> None:
        self.pending.append(transition)
        if transition.done:
            self.flush()
        elif len(self.pending) >= self.n_steps:
            self._emit_one(self.n_steps)

    def _emit_one(self, horizon: int) -> None:
        items = list(self.pending)[:horizon]
        if not items:
            return
        first = items[0]
        last = items[-1]
        discounted_reward = sum(
            (self.gamma ** index) * item.reward
            for index, item in enumerate(items)
        )
        self.replay.add(
            state=first.state,
            action_node=first.action_node,
            action_type=first.action_type,
            reward=discounted_reward,
            next_state=last.next_state,
            done=any(item.done for item in items),
            cell_id=first.cell_id,
            n_steps=len(items),
        )
        self.pending.popleft()

    def flush(self) -> None:
        while self.pending:
            self._emit_one(min(self.n_steps, len(self.pending)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local state-aware graph DQN runner for Abaqus or CalculiX"
    )
    parser.add_argument(
        "--backend",
        choices=("abaqus", "calculix", "calculix-plastic"),
        required=True,
    )
    parser.add_argument("--mode", choices=("preflight", "solve", "train"), default="train")
    parser.add_argument("--goal-file", default="examples/goal_local.json")
    parser.add_argument("--sample-goals", action="store_true")
    parser.add_argument("--stop-on-target", action="store_true")

    parser.add_argument("--simulations-root", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-elements", type=int, default=None)
    parser.add_argument("--min-elements", type=int, default=None)
    parser.add_argument("--global-mesh-size", type=float, default=None)
    parser.add_argument("--cell-min-mesh-size", type=float, default=None)
    parser.add_argument("--cell-max-mesh-size", type=float, default=None)
    parser.add_argument("--baseline-mesh-size", type=float, default=None)
    parser.add_argument("--refine-step-size", type=float, default=None)
    parser.add_argument("--coarsen-step-size", type=float, default=None)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--no-baseline-cache", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--gcn-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--replay-warmup", type=int, default=64)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--updates-after-episode", type=int, default=32)
    parser.add_argument("--target-update-tau", type=float, default=0.01)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.30)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume-v2", action="store_true")
    parser.add_argument("--save-frequency", type=int, default=1)
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=0,
        help="Run a deterministic epsilon=0 validation every N episodes (0 disables)",
    )
    parser.add_argument(
        "--best-resource-weight",
        type=float,
        default=0.10,
        help="Resource term used to rank deterministic validation checkpoints",
    )
    parser.add_argument("--debug", action="store_true")

    # Abaqus backend.
    parser.add_argument(
        "--abaqus-cmd", default=os.environ.get("ABAQUS_CMD", "abaqus")
    )
    parser.add_argument("--template-cae-file", default="DEMO.cae")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--penalty-mesh-failure", type=float, default=-5.0)
    parser.add_argument("--penalty-fea-failure", type=float, default=-5.0)
    parser.add_argument("--penalty-file-missing", type=float, default=-5.0)
    parser.add_argument("--penalty-min-elements", type=float, default=-5.0)
    parser.add_argument("--penalty-max-elements", type=float, default=-5.0)

    # CalculiX backend.
    parser.add_argument(
        "--gmsh-cmd", default=os.environ.get("GMSH_CMD", "gmsh")
    )
    parser.add_argument("--ccx-cmd", default=os.environ.get("CCX_CMD", "ccx"))
    parser.add_argument("--plate-config", default="examples/calculix_plate.json")
    parser.add_argument(
        "--plastic-plate-config",
        default="examples/calculix_plastic_plate.json",
    )
    parser.add_argument("--solver-timeout", type=int, default=300)
    return parser.parse_args()


def load_goal(filepath: Optional[str]) -> GoalCondition:
    if filepath is None:
        return GoalCondition().normalized()
    return GoalCondition.from_json(filepath)


def sample_goal(base: GoalCondition, rng: np.random.Generator) -> GoalCondition:
    concentration = 1.0 + 8.0 * np.asarray(
        [
            base.accuracy_priority,
            base.resource_priority,
            base.localization_priority,
        ],
        dtype=np.float64,
    )
    priorities = rng.dirichlet(concentration)
    return GoalCondition(
        accuracy_priority=float(priorities[0]),
        resource_priority=float(priorities[1]),
        localization_priority=float(priorities[2]),
        reserve_budget_fraction=base.reserve_budget_fraction,
        target_relative_error=base.target_relative_error,
    ).normalized()


def _value(value: Optional[float | int], fallback: float | int):
    return fallback if value is None else value


def create_backend(args: argparse.Namespace, goal: GoalCondition):
    if args.backend == "calculix-plastic":
        plate = PlasticPlateConfig.from_json(args.plastic_plate_config)
        env = StateAwareCalculixPlasticEnv(
            plate=plate,
            simulations_root=args.simulations_root or "simulations_local/calculix_plastic",
            gmsh_cmd=args.gmsh_cmd,
            ccx_cmd=args.ccx_cmd,
            global_mesh_size=float(_value(args.global_mesh_size, 0.80)),
            cell_min_mesh_size=float(_value(args.cell_min_mesh_size, 0.15)),
            cell_max_mesh_size=float(_value(args.cell_max_mesh_size, 1.60)),
            max_elements=int(_value(args.max_elements, 8_000)),
            min_elements=int(_value(args.min_elements, 50)),
            refine_step_size=float(_value(args.refine_step_size, 0.20)),
            coarsen_step_size=float(_value(args.coarsen_step_size, 0.20)),
            max_consecutive_failures=args.max_consecutive_failures,
            solver_timeout_seconds=args.solver_timeout,
        )
        env.set_goal(goal)
        defaults = {
            "baseline_mesh_size": float(_value(args.baseline_mesh_size, 0.25)),
            "ckpt_dir": args.ckpt_dir or "checkpoints_local/calculix_plastic",
        }
        return env, defaults

    if args.backend == "calculix":
        plate = PlateConfig.from_json(args.plate_config)
        env = StateAwareCalculixEnv(
            plate=plate,
            simulations_root=args.simulations_root or "simulations_local/calculix",
            gmsh_cmd=args.gmsh_cmd,
            ccx_cmd=args.ccx_cmd,
            global_mesh_size=float(_value(args.global_mesh_size, 0.80)),
            cell_min_mesh_size=float(_value(args.cell_min_mesh_size, 0.15)),
            cell_max_mesh_size=float(_value(args.cell_max_mesh_size, 1.60)),
            max_elements=int(_value(args.max_elements, 20_000)),
            min_elements=int(_value(args.min_elements, 50)),
            refine_step_size=float(_value(args.refine_step_size, 0.20)),
            coarsen_step_size=float(_value(args.coarsen_step_size, 0.20)),
            max_consecutive_failures=args.max_consecutive_failures,
            solver_timeout_seconds=args.solver_timeout,
        )
        env.set_goal(goal)
        defaults = {
            "baseline_mesh_size": float(_value(args.baseline_mesh_size, 0.25)),
            "ckpt_dir": args.ckpt_dir or "checkpoints_local/calculix",
        }
        return env, defaults

    # ABAQUS_CMD is read when abaqus_env.py is imported, so set it before the
    # lazy import below.
    os.environ["ABAQUS_CMD"] = args.abaqus_cmd
    from state_aware_env import StateAwareAbaqusEnv  # pylint: disable=import-outside-toplevel

    env = StateAwareAbaqusEnv(
        template_cae_file=args.template_cae_file,
        simulations_root=args.simulations_root or "simulations_local/abaqus",
        cpus=args.cpus,
        max_elements=int(_value(args.max_elements, 30_000)),
        min_elements=int(_value(args.min_elements, 3_000)),
        cell_min_mesh_size=float(_value(args.cell_min_mesh_size, 100.0)),
        cell_max_mesh_size=float(_value(args.cell_max_mesh_size, 400.0)),
        baseline_on_reset=True,
        global_mesh_size=float(_value(args.global_mesh_size, 300.0)),
        penalty_mesh_failure=args.penalty_mesh_failure,
        penalty_fea_failure=args.penalty_fea_failure,
        penalty_file_missing=args.penalty_file_missing,
        penalty_min_elements=args.penalty_min_elements,
        penalty_max_elements=args.penalty_max_elements,
        accuracy_weight=goal.accuracy_priority,
        resource_weight=goal.resource_priority,
        refine_step_size=float(_value(args.refine_step_size, 0.05)),
        coarsen_step_size=float(_value(args.coarsen_step_size, 0.05)),
    )
    env._max_consecutive_failures = args.max_consecutive_failures
    env.set_goal(goal)
    defaults = {
        "baseline_mesh_size": float(_value(args.baseline_mesh_size, 100.0)),
        "ckpt_dir": args.ckpt_dir or "checkpoints_local/abaqus",
    }
    return env, defaults


def backend_preflight(args: argparse.Namespace, env: Any) -> dict[str, Any]:
    if args.backend in {"calculix", "calculix-plastic"}:
        return env.preflight()
    command = split_command(args.abaqus_cmd)
    cae_path = Path(args.template_cae_file)
    return {
        "backend": "abaqus",
        "abaqus_command": command,
        "abaqus_available": command_available(command),
        "template_cae_file": str(cae_path.resolve()),
        "template_cae_exists": cae_path.exists(),
        "simulation_root": str(Path(env.simulations_root).resolve()),
    }


def relative_error(env: Any) -> Optional[float]:
    method = getattr(env, "relative_qoi_error", None)
    if callable(method):
        return method()
    current = env._current_allse()
    if current is None or env.baseline_allse is None:
        return None
    return abs(float(current) - float(env.baseline_allse)) / (
        abs(float(env.baseline_allse)) + 1.0e-12
    )


def resource_usage(env: Any, info: Optional[Dict[str, Any]] = None) -> float:
    return float(env._extract_resource_usage(info))


def checkpoint_paths(ckpt_dir: str) -> Dict[str, str]:
    return {
        "agent": os.path.join(ckpt_dir, "agent_v2.pt"),
        "replay": os.path.join(ckpt_dir, "replay_v2.pt"),
        "state": os.path.join(ckpt_dir, "training_state_v2.json"),
        "history": os.path.join(ckpt_dir, "history_v2.json"),
    }


def save_training_state(
    ckpt_dir: str,
    backend: str,
    agent: StateAwareDQNAgent,
    replay: ReplayBufferV2,
    completed_episode: int,
    epsilon: float,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    paths = checkpoint_paths(ckpt_dir)
    agent.save_checkpoint(paths["agent"])
    replay.save(paths["replay"])
    with open(paths["state"], "w", encoding="utf-8") as stream:
        json.dump(
            {
                "version": 2,
                "backend": backend,
                "completed_episode": int(completed_episode),
                "epsilon": float(epsilon),
                "args": vars(args),
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
    with open(paths["history"], "w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2, ensure_ascii=False)


def load_training_state(
    ckpt_dir: str,
    backend: str,
    agent: StateAwareDQNAgent,
    replay: ReplayBufferV2,
) -> Tuple[int, float, list[dict[str, Any]]]:
    paths = checkpoint_paths(ckpt_dir)
    for path in (paths["agent"], paths["replay"], paths["state"]):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    with open(paths["state"], "r", encoding="utf-8") as stream:
        state = json.load(stream)
    if int(state.get("version", 1)) != 2:
        raise RuntimeError("Only V2 training state can be resumed")
    if state.get("backend") != backend:
        raise RuntimeError(
            f"Checkpoint backend={state.get('backend')!r} cannot be loaded by {backend!r}"
        )
    agent.load_checkpoint(paths["agent"])
    replay.load(paths["replay"])
    history: list[dict[str, Any]] = []
    if os.path.exists(paths["history"]):
        with open(paths["history"], "r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if isinstance(loaded, list):
            history = loaded
    return (
        int(state.get("completed_episode", 0)),
        float(state.get("epsilon", 0.05)),
        history,
    )


def compute_baseline(env: Any, args: argparse.Namespace, defaults: MappingLike) -> None:
    ckpt_dir = str(defaults["ckpt_dir"])
    baseline_cache = os.path.join(ckpt_dir, "baseline_cache")
    baseline = env.compute_baseline(
        cache_dir=baseline_cache,
        use_cache=not args.no_baseline_cache,
        baseline_mesh_size=float(defaults["baseline_mesh_size"]),
    )
    if baseline is None:
        print("[WARNING] Baseline quantity is unavailable; accuracy reward is incomplete.")
    else:
        print(f"[BASELINE] {baseline:.12g}")


# A small alias avoids importing typing.Mapping only for one annotation on older
# Python environments bundled with commercial solver installations.
MappingLike = Dict[str, Any]


def run_solve_mode(env: Any, goal: GoalCondition, args: argparse.Namespace) -> None:
    env.set_goal(goal)
    env.reset(run_id=f"{args.backend}_local_solve")
    state = env.build_state(goal, max_steps=max(1, args.max_steps))
    summary = {
        "backend": args.backend,
        "cells": len(state.cell_ids),
        "node_feature_dim": int(state.node_features.shape[1]),
        "global_feature_dim": int(state.global_features.shape[1]),
        "valid_cell_actions": int(state.action_mask.sum().item()),
        "resource_usage": resource_usage(env),
        "relative_error": relative_error(env),
    }
    evaluation = getattr(env, "evaluation_metrics", None)
    if callable(evaluation):
        summary["metrics"] = evaluation()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_greedy_evaluation(
    env: Any,
    goal: GoalCondition,
    args: argparse.Namespace,
    episode_number: int,
) -> dict[str, Any]:
    """Evaluate the current network without exploration or replay updates."""

    env.set_goal(goal)
    env.reset(run_id=f"{args.backend}_greedy_eval_{episode_number:03d}")
    state = env.build_state(goal, max_steps=args.max_steps)
    actions: list[dict[str, Any]] = []
    end_reason = "max_steps"
    for step_number in range(1, args.max_steps + 1):
        selection = _EVAL_AGENT.select_action(state, epsilon=0.0)
        if selection is None:
            end_reason = "no_valid_cell_action"
            break
        cell_id = int(selection["cell_id"])
        action_type = int(selection["action"])
        _, reward, env_done, info = env.step({cell_id: action_type})
        actions.append(
            {
                "step": step_number,
                "cell_id": cell_id,
                "action": action_type,
                "action_name": ACTION_NAMES.get(action_type, str(action_type)),
                "q_value": selection.get("q_value"),
                "reward": float(reward),
                "load_fraction": info.get("load_fraction"),
                "relative_error": relative_error(env),
                "resource_usage": resource_usage(env, info),
            }
        )
        state = env.build_state(goal, max_steps=args.max_steps)
        if env_done:
            end_reason = "environment_done"
            break
        if not bool(state.action_mask.any()):
            end_reason = "no_valid_cell_action"
            break
    metrics_method = getattr(env, "evaluation_metrics", None)
    metrics = metrics_method() if callable(metrics_method) else {}
    return {
        "episode": episode_number,
        "end_reason": end_reason,
        "relative_error": relative_error(env),
        "resource_usage": resource_usage(env),
        "metrics": metrics,
        "actions": actions,
    }


# Set only inside run_training while deterministic validation is executing.
_EVAL_AGENT: StateAwareDQNAgent


def run_training(
    env: Any,
    defaults: MappingLike,
    base_goal: GoalCondition,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> None:
    ckpt_dir = str(defaults["ckpt_dir"])
    agent = StateAwareDQNAgent(
        global_dim=env.global_feature_dim,
        feat_dim=env.CELL_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_actions=int(getattr(env, "num_actions", 2)),
        num_gcn_layers=args.gcn_layers,
        gamma=args.gamma,
        lr=args.learning_rate,
        dropout=args.dropout,
    )
    replay = ReplayBufferV2(capacity=args.replay_capacity)
    global _EVAL_AGENT
    _EVAL_AGENT = agent
    start_episode = 0
    epsilon = float(args.epsilon)
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_path = os.path.join(ckpt_dir, "best_evaluation_v2.json")
    if os.path.exists(best_path):
        try:
            with open(best_path, "r", encoding="utf-8") as stream:
                best_score = float(json.load(stream).get("score", math.inf))
        except Exception:
            best_score = math.inf
    if args.resume_v2:
        start_episode, epsilon, history = load_training_state(
            ckpt_dir, args.backend, agent, replay
        )
        print(
            f"[RESUME] backend={args.backend}, episode={start_episode}, "
            f"epsilon={epsilon:.4f}, replay={len(replay)}"
        )

    for episode_index in range(start_episode, args.max_episodes):
        episode_number = episode_index + 1
        goal = sample_goal(base_goal, rng) if args.sample_goals else base_goal
        env.set_goal(goal)
        env.reset(run_id=f"{args.backend}_v2_{episode_number:03d}")
        state = env.build_state(goal, max_steps=args.max_steps)
        accumulator = NStepAccumulator(replay, args.n_step, args.gamma)
        episode_reward = 0.0
        losses: list[float] = []
        action_log: list[dict[str, Any]] = []
        end_reason = "max_steps"

        print("\n" + "=" * 78)
        print(
            f"{args.backend.upper()} episode {episode_number}/{args.max_episodes} | "
            f"epsilon={epsilon:.4f} | goal={goal.to_dict()}"
        )
        print("=" * 78)

        for step_number in range(1, args.max_steps + 1):
            selection = agent.select_action(state, epsilon=epsilon)
            if selection is None:
                end_reason = "no_valid_cell_action"
                break
            action_node = int(selection["node_index"])
            action_type = int(selection["action"])
            cell_id = int(selection["cell_id"])
            previous_state = state.snapshot()
            _, reward, env_done, info = env.step({cell_id: action_type})
            next_state = env.build_state(goal, max_steps=args.max_steps)
            current_error = relative_error(env)
            target_reached = bool(
                args.stop_on_target
                and getattr(env, "allow_early_stop", True)
                and current_error is not None
                and current_error <= goal.target_relative_error
            )
            no_next_action = not bool(next_state.action_mask.any())
            time_limit = step_number >= args.max_steps
            done = bool(env_done or target_reached or no_next_action or time_limit)
            accumulator.append(
                PendingTransition(
                    state=previous_state,
                    action_node=action_node,
                    action_type=action_type,
                    reward=float(reward),
                    next_state=next_state,
                    done=done,
                    cell_id=cell_id,
                )
            )
            if len(replay) >= args.replay_warmup:
                for _ in range(max(0, args.updates_per_step)):
                    loss = agent.train_step(
                        replay,
                        batch_size=args.batch_size,
                        target_update_tau=args.target_update_tau,
                    )
                    if loss is not None:
                        losses.append(loss)
            q_value = selection.get("q_value")
            entry = {
                "step": step_number,
                "cell_id": cell_id,
                "action": action_type,
                "action_name": ACTION_NAMES.get(action_type, str(action_type)),
                "strategy": selection["strategy"],
                "q_value": q_value,
                "reward": float(reward),
                "resource_usage": resource_usage(env, info),
                "relative_error": current_error,
                "mesh_unchanged": bool(info.get("mesh_unchanged", False)),
                "state_rollback": bool(info.get("state_rollback", False)),
                "load_step": info.get("load_step"),
                "load_fraction": info.get("load_fraction"),
                "reaction_force_x": info.get("reaction_force_x"),
                "plastic_zone_fraction": info.get("plastic_zone_fraction"),
                "max_peeq": info.get("max_peeq"),
                "error_metrics": info.get("error_metrics"),
            }
            action_log.append(entry)
            episode_reward += float(reward)
            q_text = "N/A" if q_value is None else f"{float(q_value):.6f}"
            print(
                f"step={step_number:03d} cell={cell_id:04d} "
                f"action={entry['action_name']:<8} q={q_text} "
                f"reward={reward:+.6f} resource={entry['resource_usage']:.4f} "
                f"error={current_error if current_error is not None else 'N/A'}"
            )
            state = next_state
            if done:
                if env_done:
                    end_reason = "environment_done"
                elif target_reached:
                    end_reason = "target_reached"
                elif no_next_action:
                    end_reason = "no_valid_cell_action"
                else:
                    end_reason = "max_steps"
                break

        accumulator.flush()
        if len(replay) >= args.replay_warmup:
            for _ in range(max(0, args.updates_after_episode)):
                loss = agent.train_step(
                    replay,
                    batch_size=args.batch_size,
                    target_update_tau=args.target_update_tau,
                )
                if loss is not None:
                    losses.append(loss)
        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        record = {
            "backend": args.backend,
            "episode": episode_number,
            "goal": goal.to_dict(),
            "reward": episode_reward,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "epsilon": epsilon,
            "end_reason": end_reason,
            "relative_error": relative_error(env),
            "resource_usage": resource_usage(env),
            "actions": action_log,
        }
        evaluation = getattr(env, "evaluation_metrics", None)
        if callable(evaluation):
            record["final_metrics"] = evaluation()

        if args.eval_frequency > 0 and episode_number % args.eval_frequency == 0:
            validation = run_greedy_evaluation(env, goal, args, episode_number)
            score = float(validation.get("relative_error") or math.inf) + (
                float(args.best_resource_weight)
                * float(validation.get("resource_usage") or 0.0)
            )
            validation["score"] = score
            record["greedy_validation"] = validation
            print(
                f"[GREEDY EVAL] episode={episode_number}, score={score:.6f}, "
                f"error={validation.get('relative_error')}, "
                f"resource={validation.get('resource_usage')}"
            )
            if score < best_score:
                best_score = score
                agent.save_checkpoint(os.path.join(ckpt_dir, "best_agent_v2.pt"))
                with open(best_path, "w", encoding="utf-8") as stream:
                    json.dump(validation, stream, indent=2, ensure_ascii=False)
                print(f"[BEST] deterministic checkpoint updated: score={best_score:.6f}")

        history.append(record)
        print(
            f"[EPISODE] reward={episode_reward:+.6f}, end={end_reason}, "
            f"error={record['relative_error']}, resource={record['resource_usage']:.4f}"
        )
        if episode_number % max(1, args.save_frequency) == 0:
            save_training_state(
                ckpt_dir,
                args.backend,
                agent,
                replay,
                episode_number,
                epsilon,
                history,
                args,
            )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    goal = load_goal(args.goal_file)
    env, defaults = create_backend(args, goal)
    if args.mode == "preflight":
        result = backend_preflight(args, env)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        available = (
            result.get("gmsh_available", True)
            and result.get("ccx_available", True)
            and result.get("abaqus_available", True)
            and result.get("template_cae_exists", True)
        )
        if not available:
            raise SystemExit(2)
        return

    compute_baseline(env, args, defaults)
    if args.mode == "solve":
        run_solve_mode(env, goal, args)
        return
    run_training(env, defaults, goal, args, rng)


if __name__ == "__main__":
    main()
