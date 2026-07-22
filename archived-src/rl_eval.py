import argparse
import json
import os
from typing import Any, Dict, List, Optional

from abaqus_env import AbaqusEnv
from dqn_agent import DQNAgent
from rl_main import format_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent in Abaqus environment")
    # 环境/网格配置参数（与训练脚本保持一致，便于复现）
    parser.add_argument("--template-cae-file", type=str, default="DEMO.cae",
                        help="模板 CAE 文件路径")
    parser.add_argument("--simulations-root", type=str, default="simulations",
                        help="仿真输出目录")
    parser.add_argument("--max-elements", type=int, default=30000,
                        help="网格数量上限")
    parser.add_argument("--min-elements", type=int, default=3000,
                        help="网格数量下限")
    parser.add_argument("--cell-min-mesh-size", type=float, default=100.0,
                        help="单元级最小 mesh size（<=0 关闭，默认 100）")
    parser.add_argument("--cell-max-mesh-size", type=float, default=400.0,
                        help="单元级最大 mesh size（<=0 关闭，默认 600）")
    parser.add_argument("--global-mesh-size", type=float, default=300,
                        help="初始全局网格尺寸")
    parser.add_argument("--baseline-mesh-size", type=float, default=100,
                        help="计算 baseline 时使用的 mesh size（默认 100）")
    parser.add_argument("--no-baseline-cache", action="store_true", default=False,
                        help="禁用 baseline 计算缓存")
    parser.add_argument("--penalty-mesh-failure", type=float, default=-5.0,
                        help="网格失败惩罚")
    parser.add_argument("--penalty-fea-failure", type=float, default=-5.0,
                        help="FEA 失败惩罚")
    parser.add_argument("--penalty-file-missing", type=float, default=-5.0,
                        help="缺失文件惩罚")
    parser.add_argument("--penalty-min-elements", type=float, default=-5.0,
                        help="元素数低于阈值惩罚系数")
    parser.add_argument("--penalty-max-elements", type=float, default=-5.0,
                        help="元素数超过阈值惩罚系数")
    parser.add_argument("--accuracy-weight", type=float, default=1.0,
                        help="精度奖励权重")
    parser.add_argument("--resource-weight", type=float, default=10.0,
                        help="资源消耗项权重")
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="允许的连续失败次数")
    parser.add_argument("--refine-step-size", type=float, default=0.2,
                        help="加密动作步长（百分比）")
    parser.add_argument("--coarsen-step-size", type=float, default=0.2,
                        help="稀疏动作步长（百分比）")

    # 评估流程参数
    parser.add_argument("--episodes", type=int, default=5,
                        help="评估的 episode 数量")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="单个 episode 的最大 step 数")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="评估时使用的 epsilon（默认纯贪婪）")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints",
                        help="Checkpoint 根目录")
    parser.add_argument("--agent-checkpoint", type=str, default=None,
                        help="Agent checkpoint 文件路径（默认 {ckpt-dir}/agent_checkpoint.pt）")
    parser.add_argument("--run-prefix", type=str, default="eval",
                        help="生成 run_id 时使用的前缀（例如 eval_001）")
    parser.add_argument("--record-step-details", action="store_true", default=False,
                        help="是否记录每个 step 的详细信息到输出 JSON")
    parser.add_argument("--output-file", type=str, default=None,
                        help="可选：将评估结果写入 JSON 文件")
    parser.add_argument("--debug", action="store_true", help="输出更详细的调试日志")
    return parser.parse_args()


def load_agent(agent_ckpt: str) -> DQNAgent:
    if not os.path.exists(agent_ckpt):
        raise FileNotFoundError(f"找不到 agent checkpoint: {agent_ckpt}")
    print(f"Loading agent checkpoint from {agent_ckpt}")
    agent = DQNAgent(feat_dim=None, hidden_dim=64, num_actions=2)  # 与训练脚本保持一致
    agent.load_checkpoint(agent_ckpt)
    print("Agent checkpoint loaded successfully")
    return agent


def create_environment(args: argparse.Namespace) -> AbaqusEnv:
    env = AbaqusEnv(
        template_cae_file=args.template_cae_file,
        simulations_root=args.simulations_root,
        max_elements=args.max_elements,
        min_elements=args.min_elements,
        cell_min_mesh_size=args.cell_min_mesh_size,
        cell_max_mesh_size=args.cell_max_mesh_size,
        baseline_on_reset=True,
        global_mesh_size=args.global_mesh_size,
        penalty_mesh_failure=args.penalty_mesh_failure,
        penalty_fea_failure=args.penalty_fea_failure,
        penalty_file_missing=args.penalty_file_missing,
        penalty_min_elements=args.penalty_min_elements,
        penalty_max_elements=args.penalty_max_elements,
        accuracy_weight=args.accuracy_weight,
        resource_weight=args.resource_weight,
        refine_step_size=args.refine_step_size,
        coarsen_step_size=args.coarsen_step_size
    )
    env._max_consecutive_failures = args.max_consecutive_failures
    return env


def compute_baseline(env: AbaqusEnv, args: argparse.Namespace) -> Optional[float]:
    print(f"\n{'=' * 60}")
    print("Computing baseline ALLSE before evaluation...")
    print(f"{'=' * 60}")
    baseline_cache_dir = os.path.join(args.ckpt_dir, "baseline_cache")
    use_baseline_cache = not args.no_baseline_cache
    print(f"Baseline cache: {'enabled' if use_baseline_cache else 'disabled'}")
    baseline_mesh_size = args.baseline_mesh_size if args.baseline_mesh_size is not None else args.global_mesh_size * (1 / 3)
    baseline_allse = env.compute_baseline(
        cache_dir=baseline_cache_dir,
        use_cache=use_baseline_cache,
        baseline_mesh_size=baseline_mesh_size
    )
    if baseline_allse is None:
        print("Warning: Baseline ALLSE 计算失败，评估可能缺少参考值。")
    else:
        print(f"Baseline ALLSE: {baseline_allse}")
    print(f"{'=' * 60}\n")
    return baseline_allse


def run_episode(env: AbaqusEnv, agent: DQNAgent, args: argparse.Namespace, episode_idx: int) -> Dict[str, Any]:
    run_id = f"{args.run_prefix}_{episode_idx + 1:03d}"
    print(f"\n{'=' * 60}")
    print(f"Starting evaluation episode {episode_idx + 1}/{args.episodes} | run_id={run_id}")
    print(f"{'=' * 60}")
    env.reset(run_id=run_id)

    done = False
    step = 0
    rewards: List[float] = []
    step_details: List[Dict[str, Any]] = []

    while not done and step < args.max_steps:
        step += 1
        cell_obs = env.get_cell_observations()
        cell_adj = env.cell_adjacency

        if not cell_obs:
            print("  No available cells for action selection. Ending episode early.")
            break

        # 使用与训练相同的逻辑：选择单个cell执行动作
        selection = agent.select_single_cell_action(
            cell_observations=cell_obs,
            cell_adjacency=cell_adj,
            epsilon=args.epsilon
        )
        if selection is None:
            print("  No selectable cell at this step. Ending episode early.")
            break

        selected_cell_id = selection['cell_id']
        selected_action = selection['action']
        selection_strategy = selection.get('strategy', 'exploit')
        selected_q = selection.get('q_value')
        actions_dict = {selected_cell_id: selected_action}

        if args.debug:
            print(f"  [DEBUG] Step {step}: Cell {selected_cell_id} -> {format_action(selected_action, args.refine_step_size, args.coarsen_step_size)} "
                  f"(strategy={selection_strategy}, q={selected_q if selected_q is not None else 'N/A'})")
        else:
            q_text = f", Q={selected_q:.4f}" if selected_q is not None else ""
            strategy_text = "explore" if selection_strategy == 'explore' else "exploit"
            print(f"  Step {step} | Cell {selected_cell_id} ({strategy_text}) -> {format_action(selected_action, args.refine_step_size, args.coarsen_step_size)}{q_text}")

        _, reward, done, info = env.step(actions_dict)
        rewards.append(float(reward))

        reward_components = info.get("reward_components", {})
        if reward_components:
            resource_usage = reward_components.get("resource_usage", 0.0)
            num_elements = reward_components.get("num_elements", 0)
            delta_ratio = reward_components.get("resource_delta_ratio", 0.0)
            delta_elems = reward_components.get("resource_delta_elements", 0.0)
            delta_suffix = f", +{delta_elems:.0f} elems ({delta_ratio*100:.2f}% of max)" if delta_elems else ""
            print(f"  Step {step} | Reward={reward:.6f} | Elements {num_elements}/{args.max_elements} "
                  f"({resource_usage * 100:.1f}%{delta_suffix})")
            if args.debug:
                print(f"    Accuracy Reward (weighted): {reward_components.get('accuracy_reward', 0.0):.6f}")
                print(f"    Resource Component: {reward_components.get('resource_component', 0.0):.6f}")
                print(f"    Resource Penalty: {reward_components.get('resource_penalty', 0.0):.6f}")
                print(f"    Resource Δ Elements: {delta_elems:.0f}")
        else:
            print(f"  Step {step} | Reward={reward:.6f}")

        if args.record_step_details:
            step_details.append({
                "step": step,
                "reward": reward,
                "done": done,
                "reward_components": reward_components,
                "actions": actions_dict,
                "info": info,
            })

    episode_result = {
        "episode": episode_idx + 1,
        "run_id": run_id,
        "total_reward": sum(rewards),
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else None,
        "min_reward": min(rewards) if rewards else None,
        "num_steps": len(rewards),
        "epsilon": args.epsilon,
    }

    if args.record_step_details:
        episode_result["step_details"] = step_details

    print(f"\nEpisode {episode_idx + 1} completed | Total reward={episode_result['total_reward']:.6f} "
          f"| Mean reward={episode_result['mean_reward']:.6f} | Steps={episode_result['num_steps']}")

    return episode_result


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    total_rewards = [ep["total_reward"] for ep in results]
    mean_rewards = [ep["mean_reward"] for ep in results]
    summary = {
        "episodes": len(results),
        "avg_total_reward": sum(total_rewards) / len(total_rewards),
        "best_total_reward": max(total_rewards),
        "worst_total_reward": min(total_rewards),
        "avg_mean_reward": sum(mean_rewards) / len(mean_rewards),
    }
    print(f"\n{'=' * 60}")
    print("Evaluation summary")
    print(f"  Episodes: {summary['episodes']}")
    print(f"  Avg total reward: {summary['avg_total_reward']:.6f}")
    print(f"  Avg mean reward: {summary['avg_mean_reward']:.6f}")
    print(f"  Best total reward: {summary['best_total_reward']:.6f}")
    print(f"  Worst total reward: {summary['worst_total_reward']:.6f}")
    print(f"{'=' * 60}")
    return summary


def save_results(output_path: str, summary: Dict[str, Any], episodes: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "summary": summary,
        "episodes": episodes,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Evaluation results saved to {output_path}")


def main():
    args = parse_args()
    agent_ckpt = args.agent_checkpoint or os.path.join(args.ckpt_dir, "agent_checkpoint.pt")

    agent = load_agent(agent_ckpt)
    env = create_environment(args)
    compute_baseline(env, args)

    results: List[Dict[str, Any]] = []
    for episode_idx in range(args.episodes):
        episode_result = run_episode(env, agent, args, episode_idx)
        results.append(episode_result)

    summary = summarize_results(results)

    if args.output_file:
        save_results(args.output_file, summary, results)


if __name__ == "__main__":
    main()

