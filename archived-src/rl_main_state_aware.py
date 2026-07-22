"""Train the state-aware graph DQN on the archived Abaqus environment.

This is a V2 entry point.  It intentionally does not load V1 checkpoints or
replay data because the Bellman target changed from a per-cell target to the
correct global ``max_(cell, action)`` target.

Example
-------
python rl_main_state_aware.py \
    --template-cae-file DEMO.cae \
    --goal-file goal_example.json \
    --max-episodes 100 --max-steps 100 --debug
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from state_aware_dqn_agent import (
    ACTION_NAMES,
    ReplayBufferV2,
    StateAwareDQNAgent,
)
from state_aware_env import GoalCondition, StateAwareAbaqusEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="State-aware global cell-action graph DQN for Abaqus AMR"
    )
    parser.add_argument("--template-cae-file", default="DEMO.cae")
    parser.add_argument("--simulations-root", default="simulations_v2")
    parser.add_argument("--ckpt-dir", default="checkpoints_v2")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-elements", type=int, default=30_000)
    parser.add_argument("--min-elements", type=int, default=3_000)
    parser.add_argument("--cell-min-mesh-size", type=float, default=100.0)
    parser.add_argument("--cell-max-mesh-size", type=float, default=400.0)
    parser.add_argument("--global-mesh-size", type=float, default=300.0)
    parser.add_argument("--baseline-mesh-size", type=float, default=100.0)
    parser.add_argument("--no-baseline-cache", action="store_true")

    parser.add_argument("--refine-step-size", type=float, default=0.05)
    parser.add_argument("--coarsen-step-size", type=float, default=0.05)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--penalty-mesh-failure", type=float, default=-5.0)
    parser.add_argument("--penalty-fea-failure", type=float, default=-5.0)
    parser.add_argument("--penalty-file-missing", type=float, default=-5.0)
    parser.add_argument("--penalty-min-elements", type=float, default=-5.0)
    parser.add_argument("--penalty-max-elements", type=float, default=-5.0)

    parser.add_argument("--goal-file", default=None)
    parser.add_argument(
        "--sample-goals",
        action="store_true",
        help="Sample task priorities each episode for goal-conditioned training",
    )
    parser.add_argument(
        "--stop-on-target",
        action="store_true",
        help="End an episode after the requested relative ALLSE error is reached",
    )

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

    parser.add_argument("--epsilon", type=float, default=0.30)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume-v2", action="store_true")
    parser.add_argument("--save-frequency", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_goal(filepath: Optional[str]) -> GoalCondition:
    if filepath is None:
        return GoalCondition().normalized()
    return GoalCondition.from_json(filepath)


def sample_goal(base: GoalCondition, rng: np.random.Generator) -> GoalCondition:
    """Sample around the requested task while retaining its hard constraints."""

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


def _relative_allse_error(env: StateAwareAbaqusEnv) -> Optional[float]:
    current = env._current_allse()
    if current is None or env.baseline_allse is None:
        return None
    return abs(float(current) - float(env.baseline_allse)) / (
        abs(float(env.baseline_allse)) + 1.0e-12
    )


def _checkpoint_paths(ckpt_dir: str) -> Dict[str, str]:
    return {
        "agent": os.path.join(ckpt_dir, "agent_v2.pt"),
        "replay": os.path.join(ckpt_dir, "replay_v2.pt"),
        "state": os.path.join(ckpt_dir, "training_state_v2.json"),
        "history": os.path.join(ckpt_dir, "history_v2.json"),
    }


def save_training_state(
    ckpt_dir: str,
    agent: StateAwareDQNAgent,
    replay: ReplayBufferV2,
    completed_episode: int,
    epsilon: float,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    paths = _checkpoint_paths(ckpt_dir)
    agent.save_checkpoint(paths["agent"])
    replay.save(paths["replay"])
    with open(paths["state"], "w", encoding="utf-8") as stream:
        json.dump(
            {
                "version": 2,
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
    agent: StateAwareDQNAgent,
    replay: ReplayBufferV2,
) -> tuple[int, float, list[dict[str, Any]]]:
    paths = _checkpoint_paths(ckpt_dir)
    for path in (paths["agent"], paths["replay"], paths["state"]):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    agent.load_checkpoint(paths["agent"])
    replay.load(paths["replay"])
    with open(paths["state"], "r", encoding="utf-8") as stream:
        state = json.load(stream)
    if int(state.get("version", 1)) != 2:
        raise RuntimeError("Only V2 training state can be resumed")
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    base_goal = load_goal(args.goal_file)
    env = StateAwareAbaqusEnv(
        template_cae_file=args.template_cae_file,
        simulations_root=args.simulations_root,
        max_elements=args.max_elements,
        min_elements=args.min_elements,
        cell_min_mesh_size=(
            args.cell_min_mesh_size if args.cell_min_mesh_size > 0 else None
        ),
        cell_max_mesh_size=(
            args.cell_max_mesh_size if args.cell_max_mesh_size > 0 else None
        ),
        baseline_on_reset=True,
        global_mesh_size=args.global_mesh_size,
        penalty_mesh_failure=args.penalty_mesh_failure,
        penalty_fea_failure=args.penalty_fea_failure,
        penalty_file_missing=args.penalty_file_missing,
        penalty_min_elements=args.penalty_min_elements,
        penalty_max_elements=args.penalty_max_elements,
        accuracy_weight=base_goal.accuracy_priority,
        resource_weight=base_goal.resource_priority,
        refine_step_size=args.refine_step_size,
        coarsen_step_size=args.coarsen_step_size,
    )
    env._max_consecutive_failures = args.max_consecutive_failures
    env.set_goal(base_goal)

    baseline_cache_dir = os.path.join(args.ckpt_dir, "baseline_cache")
    baseline = env.compute_baseline(
        cache_dir=baseline_cache_dir,
        use_cache=not args.no_baseline_cache,
        baseline_mesh_size=args.baseline_mesh_size,
    )
    if baseline is None:
        print("[WARNING] Baseline ALLSE is unavailable; accuracy reward is incomplete.")

    agent = StateAwareDQNAgent(
        global_dim=env.global_feature_dim,
        feat_dim=env.CELL_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_actions=2,
        num_gcn_layers=args.gcn_layers,
        gamma=args.gamma,
        lr=args.learning_rate,
        dropout=args.dropout,
    )
    replay = ReplayBufferV2(capacity=args.replay_capacity)

    start_episode = 0
    epsilon = float(args.epsilon)
    history: list[dict[str, Any]] = []
    if args.resume_v2:
        start_episode, epsilon, history = load_training_state(
            args.ckpt_dir, agent, replay
        )
        print(
            f"[RESUME] episode={start_episode}, epsilon={epsilon:.4f}, "
            f"replay={len(replay)}"
        )

    for episode_index in range(start_episode, args.max_episodes):
        episode_number = episode_index + 1
        goal = sample_goal(base_goal, rng) if args.sample_goals else base_goal
        env.set_goal(goal)
        env.reset(run_id=f"v2_run_{episode_number:03d}")
        state = env.build_state(goal, max_steps=args.max_steps)

        episode_reward = 0.0
        episode_losses: list[float] = []
        actions_log: list[dict[str, Any]] = []
        end_reason = "max_steps"

        print("\n" + "=" * 72)
        print(
            f"V2 Episode {episode_number}/{args.max_episodes} | "
            f"epsilon={epsilon:.4f} | goal={goal.to_dict()}"
        )
        print("=" * 72)

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

            target_error = _relative_allse_error(env)
            target_reached = bool(
                args.stop_on_target
                and target_error is not None
                and target_error <= goal.target_relative_error
            )
            no_next_action = not bool(next_state.action_mask.any())
            time_limit = step_number >= args.max_steps
            done = bool(env_done or target_reached or no_next_action or time_limit)

            replay.add(
                state=previous_state,
                action_node=action_node,
                action_type=action_type,
                reward=float(reward),
                next_state=next_state,
                done=done,
                cell_id=cell_id,
            )

            if len(replay) >= args.replay_warmup:
                for _ in range(max(0, args.updates_per_step)):
                    loss = agent.train_step(
                        replay,
                        batch_size=args.batch_size,
                        target_update_tau=args.target_update_tau,
                    )
                    if loss is not None:
                        episode_losses.append(loss)

            q_value = selection.get("q_value")
            resource_usage = env._extract_resource_usage(info)
            log_entry = {
                "step": step_number,
                "cell_id": cell_id,
                "action": action_type,
                "action_name": ACTION_NAMES.get(action_type, str(action_type)),
                "strategy": selection["strategy"],
                "q_value": q_value,
                "reward": float(reward),
                "resource_usage": resource_usage,
                "relative_allse_error": target_error,
                "mesh_unchanged": bool(info.get("mesh_unchanged", False)),
                "state_rollback": bool(info.get("state_rollback", False)),
            }
            if args.debug and previous_state.cell_ids == next_state.cell_ids:
                log_entry["q_sensitivity"] = agent.state_sensitivity(
                    previous_state, next_state
                )
            actions_log.append(log_entry)
            episode_reward += float(reward)

            q_text = "N/A" if q_value is None else f"{float(q_value):.5f}"
            error_text = "N/A" if target_error is None else f"{target_error:.5e}"
            print(
                f"step={step_number:03d} cell={cell_id} "
                f"action={ACTION_NAMES.get(action_type)} q={q_text} "
                f"reward={float(reward):+.5f} resource={resource_usage:.3f} "
                f"rel_error={error_text}"
            )
            if args.debug:
                print(
                    "  mask-valid=", int(next_state.action_mask.sum().item()),
                    "rollback=", bool(info.get("state_rollback", False)),
                    "mesh_unchanged=", bool(info.get("mesh_unchanged", False)),
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

        if len(replay) >= args.replay_warmup:
            for _ in range(max(0, args.updates_after_episode)):
                loss = agent.train_step(
                    replay,
                    batch_size=args.batch_size,
                    target_update_tau=args.target_update_tau,
                )
                if loss is not None:
                    episode_losses.append(loss)

        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        summary = {
            "episode": episode_number,
            "goal": goal.to_dict(),
            "total_reward": episode_reward,
            "steps": len(actions_log),
            "end_reason": end_reason,
            "epsilon_after_episode": epsilon,
            "mean_loss": (
                float(np.mean(episode_losses)) if episode_losses else None
            ),
            "replay_size": len(replay),
            "final_relative_allse_error": _relative_allse_error(env),
            "actions": actions_log,
        }
        history.append(summary)
        print(
            f"[EPISODE] reward={episode_reward:+.6f}, steps={len(actions_log)}, "
            f"reason={end_reason}, mean_loss={summary['mean_loss']}"
        )

        if episode_number % max(1, args.save_frequency) == 0:
            save_training_state(
                args.ckpt_dir,
                agent,
                replay,
                completed_episode=episode_number,
                epsilon=epsilon,
                history=history,
                args=args,
            )

    save_training_state(
        args.ckpt_dir,
        agent,
        replay,
        completed_episode=args.max_episodes,
        epsilon=epsilon,
        history=history,
        args=args,
    )


if __name__ == "__main__":
    main()
