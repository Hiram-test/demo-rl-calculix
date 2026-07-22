from abaqus_env import AbaqusEnv
from dqn_agent import DQNAgent, ReplayBuffer
import argparse
import os
import json
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Run RL loop over Abaqus environment')
    parser.add_argument('--template-cae-file', type=str, default='DEMO.cae', 
                       help='Path to template CAE file')
    parser.add_argument('--simulations-root', type=str, default='simulations', 
                       help='Directory to store simulations')
    parser.add_argument('--max-episodes', type=int, default=100, 
                       help='Number of episodes to run')
    parser.add_argument('--max-elements', type=int, default=30000, 
                       help='Maximum allowed elements for meshing/analysis (default: 30000)')
    parser.add_argument('--min-elements', type=int, default=3000, 
                       help='Minimum allowed elements for meshing/analysis (default: 3000)')
    parser.add_argument('--cell-min-mesh-size', type=float, default=100.0,
                       help='Minimum allowed mesh size per cell (<=0 disables, default: 100)')
    parser.add_argument('--cell-max-mesh-size', type=float, default=400.0,
                       help='Maximum allowed mesh size per cell (<=0 disables, default: 600)')
    parser.add_argument('--global-mesh-size', type=float, default=300, 
                       help='Global mesh size parameter (default: 300)')
    parser.add_argument('--epsilon', type=float, default=0.3, 
                       help='Epsilon for epsilon-greedy policy (default: 0.3)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, 
                       help='Epsilon decay rate per episode (default: 0.995)')
    parser.add_argument('--epsilon-min', type=float, default=0.05, 
                       help='Minimum epsilon value (default: 0.05)')
    parser.add_argument('--batch-size', type=int, default=64, 
                       help='Batch size for training (default: 64)')
    parser.add_argument('--train-frequency', type=int, default=1, 
                       help='Train every N steps (default: 1) [DEPRECATED: now training happens after each episode]')
    parser.add_argument('--train-batches-per-episode', type=int, default=100, 
                       help='Number of training batches to run after each episode (default: 100)')
    parser.add_argument('--baseline-mesh-size', type=float, default=100, 
                       help='Mesh size for baseline computation (default: 100)')
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints', 
                       help='Directory to save/load checkpoints (default: checkpoints)')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume training from checkpoint in ckpt-dir')
    parser.add_argument('--save-frequency', type=int, default=1, 
                       help='Save checkpoint every N episodes (default: 1)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    parser.add_argument('--reward-history-file', type=str, default=None, 
                       help='Path to save reward history JSON file (default: {ckpt-dir}/reward_history.json)')
    parser.add_argument('--no-baseline-cache', action='store_true', default=False,
                       help='Disable baseline cache, always recompute baseline (default: False, cache is enabled)')
    # 分级惩罚配置
    parser.add_argument('--penalty-mesh-failure', type=float, default=-5.0,
                       help='Penalty for mesh generation failure (default: -50.0)')
    parser.add_argument('--penalty-fea-failure', type=float, default=-5.0,
                       help='Penalty for FEA analysis failure (default: -100.0)')
    parser.add_argument('--penalty-file-missing', type=float, default=-5.0,
                       help='Penalty for missing required files (default: -200.0)')
    parser.add_argument('--penalty-min-elements', type=float, default=-5.0,
                       help='Penalty coefficient for mesh count below minimum requirement (default: -1.0)')
    parser.add_argument('--penalty-max-elements', type=float, default=-5.0,
                       help='Penalty coefficient for mesh count above maximum requirement (default: -5.0)')
    
    # Reward平衡配置
    parser.add_argument('--accuracy-weight', type=float, default=1.0,
                       help='Weight for accuracy reward (default: 1.0)')
    parser.add_argument('--resource-weight', type=float, default=0.5,
                       help='Weight for resource usage term (default: 5.0)')
    
    # 软失败机制配置
    parser.add_argument('--max-consecutive-failures', type=int, default=5,
                       help='Maximum consecutive failures before terminating episode (default: 5)')
    
    # 动作步长配置
    parser.add_argument('--refine-step-size', type=float, default=0.05,
                       help='Step size for mesh refinement action (default: 0.1, i.e., 10%)')
    parser.add_argument('--coarsen-step-size', type=float, default=0.05,
                       help='Step size for mesh coarsening action (default: 0.1, i.e., 10%)')
    
    return parser.parse_args()


def format_action(action: int, refine_step_size: float = 0.1, coarsen_step_size: float = 0.1) -> str:
    """将action值转换为可读的字符串"""
    action_map = {
        0: f'加密 (refine, -{refine_step_size*100:.0f}%)',
        1: f'稀疏 (coarsen, +{coarsen_step_size*100:.0f}%)',
        2: 'no-op (保持不变)'
    }
    return action_map.get(action, f'unknown ({action})')


def save_training_state(ckpt_dir: str, agent: DQNAgent, replay_buffer: ReplayBuffer, 
                       episode: int, args: argparse.Namespace, reward_history: list = None,
                       loss_history: list = None):
    """保存训练状态到checkpoint目录"""
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 保存agent checkpoint
    agent_path = os.path.join(ckpt_dir, 'agent_checkpoint.pt')
    if agent._q_net is not None:
        agent.save_checkpoint(agent_path)
        print(f"  Saved agent checkpoint to {agent_path}")
    else:
        print(f"  Warning: Agent network not initialized, skipping agent checkpoint")
    
    # 保存replay buffer
    buffer_path = os.path.join(ckpt_dir, 'replay_buffer.pt')
    replay_buffer.save(buffer_path)
    print(f"  Saved replay buffer to {buffer_path}")
    
    # 保存训练状态和超参数
    state_path = os.path.join(ckpt_dir, 'training_state.json')
    training_state = {
        'episode': episode,
        'args': vars(args),
    }
    with open(state_path, 'w') as f:
        json.dump(training_state, f, indent=2)
    print(f"  Saved training state to {state_path}")
    
    # 保存reward历史
    if reward_history is not None:
        if args.reward_history_file:
            reward_history_file = args.reward_history_file
        else:
            reward_history_file = os.path.join(ckpt_dir, 'reward_history.json')
        reward_dir = os.path.dirname(reward_history_file)
        if reward_dir:
            os.makedirs(reward_dir, exist_ok=True)
        with open(reward_history_file, 'w') as f:
            json.dump(reward_history, f, indent=2)
        print(f"  Saved reward history to {reward_history_file}")
    
    # 保存loss历史
    if loss_history is not None:
        loss_history_file = os.path.join(ckpt_dir, 'loss_history.json')
        with open(loss_history_file, 'w') as f:
            json.dump(loss_history, f, indent=2)
        print(f"  Saved loss history to {loss_history_file}")


def load_training_state(ckpt_dir: str, agent: DQNAgent, replay_buffer: ReplayBuffer, 
                       reward_history_file: str = None) -> tuple:
    """从checkpoint目录加载训练状态，返回恢复的episode编号、reward历史和loss历史"""
    agent_path = os.path.join(ckpt_dir, 'agent_checkpoint.pt')
    buffer_path = os.path.join(ckpt_dir, 'replay_buffer.pt')
    state_path = os.path.join(ckpt_dir, 'training_state.json')
    
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent checkpoint not found: {agent_path}")
    if not os.path.exists(buffer_path):
        raise FileNotFoundError(f"Replay buffer checkpoint not found: {buffer_path}")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Training state not found: {state_path}")
    
    # 加载agent
    agent.load_checkpoint(agent_path)
    print(f"  Loaded agent checkpoint from {agent_path}")
    
    # 加载replay buffer
    replay_buffer.load(buffer_path)
    print(f"  Loaded replay buffer from {buffer_path} (size: {len(replay_buffer)})")
    
    # 加载训练状态
    with open(state_path, 'r') as f:
        training_state = json.load(f)
    episode = training_state.get('episode', 0)
    print(f"  Loaded training state from {state_path} (resuming from episode {episode + 1})")
    
    # 加载reward历史（如果存在）
    reward_history = []
    reward_file = reward_history_file or os.path.join(ckpt_dir, 'reward_history.json')
    if os.path.exists(reward_file):
        try:
            with open(reward_file, 'r') as f:
                reward_history = json.load(f)
            print(f"  Loaded reward history from {reward_file} ({len(reward_history)} episodes)")
        except Exception as e:
            print(f"  Warning: Failed to load reward history: {e}")
    else:
        print(f"  No reward history found at {reward_file}, starting fresh")
    
    # 加载loss历史
    loss_history = []
    loss_file = os.path.join(ckpt_dir, 'loss_history.json')
    if os.path.exists(loss_file):
        try:
            with open(loss_file, 'r') as f:
                loss_history = json.load(f)
            print(f"  Loaded loss history from {loss_file} ({len(loss_history)} records)")
        except Exception as e:
            print(f"  Warning: Failed to load loss history: {e}")
    else:
        print(f"  No loss history found at {loss_file}, starting fresh")
    
    return episode, reward_history, loss_history


def main():
    args = parse_args()
    
    # 初始化DQN Agent (feat_dim动态确定，不需要在初始化时指定)
    # num_actions=2: 0=小幅加密, 1=小幅稀疏
    agent = DQNAgent(feat_dim=None, hidden_dim=64, num_actions=2)
    
    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(capacity=100_000)
    
    # 恢复训练（如果需要）
    start_episode = 0
    reward_history = []  # 存储reward历史记录
    loss_history = []  # 存储loss历史记录
    if args.resume:
        if not os.path.exists(args.ckpt_dir):
            print(f"Warning: Checkpoint directory '{args.ckpt_dir}' does not exist. Starting from scratch.")
        else:
            try:
                start_episode, reward_history, loss_history = load_training_state(
                    args.ckpt_dir, agent, replay_buffer, args.reward_history_file
                )
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch.")
                start_episode = 0
                reward_history = []
                loss_history = []
    
    # 创建环境
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
    # 设置最大连续失败次数
    env._max_consecutive_failures = args.max_consecutive_failures

    # 在训练开始前计算一次baseline ALLSE值（使用缓存机制避免重复计算）
    print(f"\n{'='*60}")
    print(f"Computing baseline ALLSE before training starts...")
    print(f"{'='*60}")
    # 使用checkpoints目录存放baseline缓存，保持文件组织统一
    baseline_cache_dir = os.path.join(args.ckpt_dir, 'baseline_cache')
    use_baseline_cache = not args.no_baseline_cache
    if use_baseline_cache:
        print(f"Baseline cache enabled (cache dir: {baseline_cache_dir})")
    else:
        print(f"Baseline cache disabled (--no-baseline-cache flag set)")
    # 确定baseline mesh size：如果未指定，使用 global_mesh_size * 1/3
    baseline_mesh_size = args.baseline_mesh_size if args.baseline_mesh_size is not None else args.global_mesh_size * (1/3)
    print(f"Baseline mesh size: {baseline_mesh_size}")
    baseline_allse = env.compute_baseline(cache_dir=baseline_cache_dir, use_cache=use_baseline_cache, baseline_mesh_size=baseline_mesh_size)
    if baseline_allse is None:
        print("Warning: Failed to compute baseline ALLSE. Training may not work correctly.")
    else:
        print(f"Baseline ALLSE successfully computed: {baseline_allse}")
    print(f"{'='*60}\n")

    # 运行训练循环
    i = start_episode
    current_epsilon = args.epsilon
    
    # 如果恢复训练，根据已完成的episode数调整epsilon
    if start_episode > 0:
        current_epsilon = max(args.epsilon_min, args.epsilon * (args.epsilon_decay ** start_episode))
        print(f"Resuming with epsilon: {current_epsilon:.4f}")
    
    while i < args.max_episodes:
        print(f"\n{'='*60}")
        print(f"Starting Episode {i+1}/{args.max_episodes} (epsilon: {current_epsilon:.4f})")
        print(f"{'='*60}")

        # 重置环境
        env.reset(run_id=f"run_{i+1:03d}")

        # 多步交互：每步对所有cell同时进行决策
        done = False
        t = 0
        episode_rewards = []  # 记录当前episode的所有reward
        
        while not done:
            t += 1
            
            # 获取cell观测数据和邻接关系
            cell_observations = env.get_cell_observations()
            cell_adjacency = env.cell_adjacency  # 获取cell邻接关系
            
            # 保存当前状态（执行动作前的状态）用于训练
            state_cell_observations = cell_observations.copy()
            
            if not cell_observations:
                if args.debug:
                    print(f"  [DEBUG] No cells available for action selection")
                done = True
                continue
            
            # 选择单个cell执行动作
            selection = agent.select_single_cell_action(
                cell_observations=cell_observations,
                cell_adjacency=cell_adjacency,
                epsilon=current_epsilon
            )
            if selection is None:
                if args.debug:
                    print(f"  [DEBUG] No selectable cell at step {t}, terminating episode")
                done = True
                continue
            
            selected_cell_id = selection['cell_id']
            selected_action = selection['action']
            selection_strategy = selection.get('strategy', 'exploit')
            selected_q = selection.get('q_value')
            actions_dict = {selected_cell_id: selected_action}
            
            # 输出动作信息
            action_desc = format_action(selected_action, args.refine_step_size, args.coarsen_step_size)
            if args.debug:
                print(f"  [DEBUG] Step {t}: Cell {selected_cell_id} -> {action_desc} "
                      f"(strategy={selection_strategy}, q={selected_q if selected_q is not None else 'N/A'})")
            else:
                q_text = f", Q={selected_q:.4f}" if selected_q is not None else ""
                strategy_text = "explore" if selection_strategy == 'explore' else "exploit"
                print(f"  Step {t} | Cell {selected_cell_id} ({strategy_text}) -> {action_desc}{q_text}")
            
            # 执行动作（单cell执行）
            obs, reward, done, info = env.step(actions_dict)
            
            # 记录reward
            episode_rewards.append(float(reward))
            
            # 输出step信息，包括奖励组件
            reward_components = info.get('reward_components', {})
            if reward_components:
                resource_usage = reward_components.get('resource_usage', 0.0)
                num_elements = reward_components.get('num_elements', 0)
                delta_ratio = reward_components.get('resource_delta_ratio', 0.0)
                delta_elems = reward_components.get('resource_delta_elements', 0.0)
                delta_suffix = f", +{delta_elems:.0f} elems ({delta_ratio*100:.2f}% of max)" if delta_elems else ""
                print(f"  Step {t} | Reward: {reward:.6f} | Elements: {num_elements}/{args.max_elements} "
                      f"({resource_usage*100:.1f}%{delta_suffix})")
                
                if args.debug:
                    # 打印详细的奖励组件
                    print(f"    [Weighted Combination Mode]")
                    print(f"      Accuracy Reward (weighted): {reward_components.get('accuracy_reward', 0.0):.6f}")
                    print(f"      Resource Component: {reward_components.get('resource_component', 0.0):.6f}")
                    print(f"      Resource Penalty: {reward_components.get('resource_penalty', 0.0):.6f}")
                    print(f"      Resource Δ Elements: {delta_elems:.0f}")
                    print(f"      Current ALLSE: {reward_components.get('current_allse', 0.0):.6f}")
                    print(f"      Baseline ALLSE: {reward_components.get('baseline_allse', 0.0):.6f}")
            else:
                print(f"  Step {t} | Reward: {reward:.6f}")

            # 训练逻辑：保存经验并训练模型
            try:
                # 获取下一状态（执行动作后的状态）
                next_cell_observations = env.get_cell_observations()
                
                # 构建图数据（当前状态）
                s_node_feat, s_edge_idx, s_cell_id_to_index = agent.build_graph_data(
                    state_cell_observations, cell_adjacency
                )
                # 创建空的全局特征张量（保持与训练代码兼容）
                s_global_feat = torch.zeros((1, 0), dtype=torch.float32)
                
                # 构建图数据（下一状态）
                # 检测是否发生了状态回退（任何类型的失败）
                state_rollback = info.get('state_rollback', False)
                
                if next_cell_observations and not state_rollback:
                    # 正常情况：使用真实的下一状态
                    ns_node_feat, ns_edge_idx, ns_cell_id_to_index = agent.build_graph_data(
                        next_cell_observations, cell_adjacency
                    )
                    # 创建空的全局特征张量（保持与训练代码兼容）
                    ns_global_feat = torch.zeros((1, 0), dtype=torch.float32)
                else:
                    # 发生失败（状态回退）或episode结束：使用当前状态作为下一状态
                    # 
                    # 【软失败学习策略】当动作导致失败时：
                    # - 保留负reward（惩罚）
                    # - next_state = current_state（表示"状态回退"/"无进展"）
                    # - done取决于连续失败次数（软失败机制）
                    # 
                    # TD target计算：Q(s,a) ← r + γ * max Q(s', a')
                    # 当 s' = s 时：
                    #   - 如果done=False: Q(s,a) ← r + γ * max Q(s, a')（可以继续尝试其他动作）
                    #   - 如果done=True: Q(s,a) ← r（连续失败太多，episode终止）
                    # 
                    # 这样模型学到：
                    #   "在状态s采取某个动作a → 获得负reward r → 回到状态s"
                    # 模型会学习**避免**导致失败的动作，但仍有机会尝试其他动作
                    ns_node_feat, ns_edge_idx, ns_global_feat, ns_cell_id_to_index = (
                        s_node_feat.clone(), s_edge_idx.clone(), s_global_feat.clone(), s_cell_id_to_index.copy()
                    )
                    if state_rollback:
                        penalty_type = info.get('penalty_type', 'unknown')
                        consecutive_failures = info.get('consecutive_failures', 0)
                        if args.debug:
                            print(f"  [LEARNING] Failure transition saved for learning:")
                            print(f"    - Penalty type: {penalty_type}")
                            print(f"    - Reward: {reward:.2f} (penalty)")
                            print(f"    - Done: {done} (soft_failure={not done}, consecutive_failures={consecutive_failures})")
                            print(f"    - Strategy: s' = s (state rollback) + negative reward → learn to avoid")
                        else:
                            status = "will retry" if not done else "episode terminates"
                            print(f"  [LEARNING] Saving {penalty_type} experience: reward={reward:.2f}, "
                                  f"consecutive_failures={consecutive_failures}, {status}")
                
                # 【安全检查】确保维度一致（理论上现在应该总是一致的）
                if s_node_feat.shape[0] != ns_node_feat.shape[0]:
                    if args.debug:
                        print(f"  [WARNING] Unexpected dimension mismatch after handling: "
                              f"state={s_node_feat.shape[0]}, next_state={ns_node_feat.shape[0]}")
                    # 如果仍然不一致，强制使用当前状态
                    ns_node_feat, ns_edge_idx, ns_global_feat, ns_cell_id_to_index = (
                        s_node_feat.clone(), s_edge_idx.clone(), s_global_feat.clone(), s_cell_id_to_index.copy()
                    )
                
                # 将actions_dict转换为tensor（按cell索引顺序）
                if s_node_feat.shape[0] > 0:
                    # 创建actions tensor和mask，仅标记被选择的cell
                    actions_tensor = torch.zeros(s_node_feat.shape[0], dtype=torch.long)
                    action_mask = torch.zeros(s_node_feat.shape[0], dtype=torch.bool)
                    
                    selected_idx = s_cell_id_to_index.get(selected_cell_id)
                    if selected_idx is None or selected_idx >= actions_tensor.shape[0]:
                        if args.debug:
                            print(f"  [DEBUG] Selected cell {selected_cell_id} not found in graph indices, skipping experience")
                        continue
                    
                    actions_tensor[selected_idx] = selected_action
                    action_mask[selected_idx] = True
                    
                    # 使用单cell reward
                    cell_rewards = info.get('cell_rewards', {})
                    rewards_tensor = torch.zeros(s_node_feat.shape[0], dtype=torch.float32)
                    cell_reward_value = None
                    if cell_rewards:
                        cell_reward_value = cell_rewards.get(selected_cell_id)
                        if cell_reward_value is None:
                            cell_reward_value = cell_rewards.get(str(selected_cell_id))
                    if cell_reward_value is not None:
                        rewards_tensor[selected_idx] = float(cell_reward_value)
                    else:
                        rewards_tensor[selected_idx] = float(reward)
                    reward_to_save = rewards_tensor
                    
                    # 保存经验到replay buffer
                    replay_buffer.add(
                        node_features=s_node_feat,
                        edge_index=s_edge_idx,
                        global_feat=s_global_feat,
                        actions=actions_tensor,
                        reward=reward_to_save,
                        next_node_features=ns_node_feat,
                        next_edge_index=ns_edge_idx,
                        next_global_feat=ns_global_feat,
                        done=done,
                        action_mask=action_mask
                    )
            
            except Exception as e:
                if args.debug:
                    print(f"  [DEBUG] Error saving experience: {e}")
                    import traceback
                    traceback.print_exc()
                else:
                    print(f"  Warning: Failed to save experience: {type(e).__name__}")

            # 每个episode只执行一次动作后结束（可根据需要修改）
            if t >= 400:
                done = True

        # Episode结束后，集中训练模型
        print(f"\n  Training model after episode {i+1}...")
        training_losses = []
        if len(replay_buffer) >= args.batch_size:
            # 动态调整训练batch数：根据replay buffer大小
            # 确保充分利用buffer中的经验，但避免过度训练
            buffer_size = len(replay_buffer)
            if buffer_size < 1000:
                # 早期：训练次数与buffer大小成比例，避免过度重复采样
                num_batches = min(args.train_batches_per_episode, max(10, buffer_size // args.batch_size))
            else:
                # 后期：使用固定训练次数，充分利用buffer
                num_batches = args.train_batches_per_episode
            
            for train_iter in range(num_batches):
                loss = agent.train_step(replay_buffer, batch_size=args.batch_size)
                if loss is not None:
                    training_losses.append(loss)
                else:
                    break  # 如果返回None，说明样本不足，停止训练
            
            if training_losses:
                avg_loss = sum(training_losses) / len(training_losses)
                # 记录loss到历史
                loss_history.append({
                    'episode': i + 1,
                    'avg_loss': avg_loss,
                    'min_loss': min(training_losses),
                    'max_loss': max(training_losses),
                    'num_batches': len(training_losses),
                    'buffer_size': len(replay_buffer)
                })
                print(f"  Training completed: {len(training_losses)} batches, "
                      f"Average loss: {avg_loss:.6f}, Replay buffer size: {len(replay_buffer)}")
            else:
                print(f"  Training skipped: insufficient samples in replay buffer ({len(replay_buffer)} < {args.batch_size})")
        else:
            print(f"  Training skipped: insufficient samples in replay buffer ({len(replay_buffer)} < {args.batch_size})")
        
        # 计算episode的reward统计信息
        if episode_rewards:
            episode_stats = {
                'episode': i + 1,
                'total_reward': sum(episode_rewards),
                'mean_reward': sum(episode_rewards) / len(episode_rewards),
                'max_reward': max(episode_rewards),
                'min_reward': min(episode_rewards),
                'num_steps': len(episode_rewards),
                'step_rewards': episode_rewards,  # 保存每个step的reward
                'avg_training_loss': sum(training_losses) / len(training_losses) if training_losses else None,  # 保存训练损失
                'epsilon': current_epsilon  # 保存当前epsilon值
            }
            reward_history.append(episode_stats)
            
            # 计算最近N个episode的平均reward（用于查看训练趋势）
            recent_episodes = 10
            if len(reward_history) >= recent_episodes:
                recent_mean_rewards = [h['mean_reward'] for h in reward_history[-recent_episodes:]]
                avg_recent_mean = sum(recent_mean_rewards) / len(recent_mean_rewards)
                print(f"\nEpisode {i+1} reward stats: Total={episode_stats['total_reward']:.6f}, "
                      f"Mean={episode_stats['mean_reward']:.6f}, "
                      f"Max={episode_stats['max_reward']:.6f}, Min={episode_stats['min_reward']:.6f}")
                print(f"  Average mean reward over last {recent_episodes} episodes: {avg_recent_mean:.6f}")
            else:
                print(f"\nEpisode {i+1} reward stats: Total={episode_stats['total_reward']:.6f}, "
                      f"Mean={episode_stats['mean_reward']:.6f}, "
                      f"Max={episode_stats['max_reward']:.6f}, Min={episode_stats['min_reward']:.6f}")
        else:
            print(f"\nEpisode {i+1} completed with no rewards recorded")
        
        print(f"Episode {i+1} completed\n")
        
        # 衰减epsilon
        current_epsilon = max(args.epsilon_min, current_epsilon * args.epsilon_decay)
        
        # 定期保存checkpoint
        if args.ckpt_dir and (i + 1) % args.save_frequency == 0:
            print(f"  Saving checkpoint at episode {i+1}...")
            try:
                save_training_state(args.ckpt_dir, agent, replay_buffer, i + 1, args, reward_history, loss_history)
            except Exception as e:
                print(f"  Warning: Failed to save checkpoint: {e}")
        
        i += 1  # 更新episode索引
    
    # 训练结束后保存最终checkpoint
    if args.ckpt_dir:
        print(f"\nSaving final checkpoint...")
        try:
            save_training_state(args.ckpt_dir, agent, replay_buffer, args.max_episodes, args, reward_history, loss_history)
        except Exception as e:
            print(f"Warning: Failed to save final checkpoint: {e}")
    
    # 保存最终reward历史（即使没有checkpoint目录）
    if reward_history:
        if args.reward_history_file:
            reward_history_file = args.reward_history_file
        else:
            reward_history_file = os.path.join(
                args.ckpt_dir if args.ckpt_dir else '.', 'reward_history.json'
            )
        # 确保目录存在
        reward_dir = os.path.dirname(reward_history_file)
        if reward_dir:
            os.makedirs(reward_dir, exist_ok=True)
        with open(reward_history_file, 'w') as f:
            json.dump(reward_history, f, indent=2)
        print(f"Final reward history saved to {reward_history_file}")
        
        # 打印总结统计
        if len(reward_history) > 0:
            all_mean_rewards = [h['mean_reward'] for h in reward_history]
            print(f"\n{'='*60}")
            print(f"Training Summary:")
            print(f"  Total episodes: {len(reward_history)}")
            print(f"  Overall mean reward: {sum(all_mean_rewards) / len(all_mean_rewards):.6f}")
            print(f"  Best episode mean reward: {max(all_mean_rewards):.6f} (Episode {reward_history[all_mean_rewards.index(max(all_mean_rewards))]['episode']})")
            print(f"  Worst episode mean reward: {min(all_mean_rewards):.6f} (Episode {reward_history[all_mean_rewards.index(min(all_mean_rewards))]['episode']})")
            print(f"{'='*60}")


if __name__ == '__main__':
    main()
