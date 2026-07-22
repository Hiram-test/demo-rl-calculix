from abaqus_env import AbaqusEnv
from dqn_agent import DQNAgent, ReplayBuffer
import argparse
import os
import json
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import time
from typing import Dict, Tuple
import traceback
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Run RL loop over Abaqus environment with multiprocessing')
    parser.add_argument('--template-cae-file', type=str, default='DEMO.cae', 
                       help='Path to template CAE file')
    parser.add_argument('--simulations-root', type=str, default='simulations', 
                       help='Directory to store simulations')
    parser.add_argument('--max-episodes', type=int, default=100, 
                       help='Number of episodes to run')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel worker processes (default: 3)')
    parser.add_argument('--max-elements', type=int, default=30000, 
                       help='Maximum allowed elements for meshing/analysis (default: 30000)')
    parser.add_argument('--min-elements', type=int, default=6000, 
                       help='Minimum allowed elements for meshing/analysis (default: 3000)')
    parser.add_argument('--cell-min-mesh-size', type=float, default=50.0,
                       help='Minimum allowed mesh size per cell (<=0 disables, default: 100)')
    parser.add_argument('--cell-max-mesh-size', type=float, default=400.0,
                       help='Maximum allowed mesh size per cell (<=0 disables, default: 600)')
    parser.add_argument('--global-mesh-size', type=float, default=300, 
                       help='Global mesh size parameter (default: 300)')
    parser.add_argument('--epsilon', type=float, default=0.5, 
                       help='Epsilon for epsilon-greedy policy (default: 0.3)')
    parser.add_argument('--epsilon-decay', type=float, default=0.95, 
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
    parser.add_argument('--sync-frequency', type=int, default=1,
                       help='Synchronize model parameters to workers every N episodes (default: 1)')
    parser.add_argument('--sync-mode', type=str, default='episode', choices=['episode', 'training'],
                       help='Synchronization mode: "episode" (every N episodes) or "training" (every N training steps) (default: episode)')
    # 分级惩罚配置
    parser.add_argument('--penalty-mesh-failure', type=float, default=-5.0,
                       help='Penalty for mesh generation failure (default: -50.0)')
    parser.add_argument('--penalty-fea-failure', type=float, default=-5.0,
                       help='Penalty for FEA analysis failure (default: -100.0)')
    parser.add_argument('--penalty-file-missing', type=float, default=-5.0,
                       help='Penalty for missing required files (default: -200.0)')
    parser.add_argument('--penalty-min-elements', type=float, default=-5.0,
                       help='Penalty coefficient for mesh count below minimum requirement (default: -1.0)')
    
    # Reward平衡配置
    parser.add_argument('--accuracy-weight', type=float, default=1.0,
                       help='Weight for accuracy reward (default: 1.0)')
    parser.add_argument('--resource-weight', type=float, default=1.0,
                       help='Weight for resource usage term (default: 5.0)')
    
    # 软失败机制配置
    parser.add_argument('--max-consecutive-failures', type=int, default=5,
                       help='Maximum consecutive failures before terminating episode (default: 5)')
    
    # 动作步长配置
    parser.add_argument('--refine-step-size', type=float, default=0.05,
                       help='Step size for mesh refinement action (default: 0.05, i.e., 5%)')
    parser.add_argument('--coarsen-step-size', type=float, default=0.05,
                       help='Step size for mesh coarsening action (default: 0.05, i.e., 5%)')
    
    return parser.parse_args()


def format_action(action: int, refine_step_size: float = 0.05, coarsen_step_size: float = 0.05) -> str:
    """将action值转换为可读的字符串"""
    action_map = {
        0: f'加密 (refine, -{refine_step_size*100:.0f}%)',
        1: f'稀疏 (coarsen, +{coarsen_step_size*100:.0f}%)',
        2: 'no-op (保持不变)'
    }
    return action_map.get(action, f'unknown ({action})')


def prepare_worker_cae_files(template_cae_file: str, num_workers: int, work_dir: str = 'worker_cae_files') -> list:
    """
    为每个worker创建独立的CAE文件副本，避免多进程读取冲突
    
    Args:
        template_cae_file: 模板CAE文件路径
        num_workers: worker数量
        work_dir: 存放CAE副本的目录
    
    Returns:
        list: 每个worker的CAE文件路径列表
    """
    os.makedirs(work_dir, exist_ok=True)
    worker_cae_files = []
    
    for worker_id in range(num_workers):
        # 构建worker专属的CAE文件路径
        base_name = os.path.splitext(os.path.basename(template_cae_file))[0]
        worker_cae_file = os.path.join(work_dir, f'{base_name}_worker{worker_id}.cae')
        
        # 复制CAE文件
        shutil.copy2(template_cae_file, worker_cae_file)
        worker_cae_files.append(worker_cae_file)
        print(f"Created CAE copy for Worker {worker_id}: {worker_cae_file}")
    
    return worker_cae_files


def cleanup_worker_cae_files(work_dir: str = 'worker_cae_files'):
    """
    清理worker的CAE文件副本
    
    Args:
        work_dir: 存放CAE副本的目录
    """
    if os.path.exists(work_dir):
        try:
            shutil.rmtree(work_dir)
            print(f"Cleaned up worker CAE files directory: {work_dir}")
        except Exception as e:
            print(f"Warning: Failed to cleanup worker CAE files: {e}")


def worker_process(worker_id: int, 
                   experience_queue: Queue, 
                   param_queue: Queue,
                   command_queue: Queue,
                   stop_event: Event,
                   args: argparse.Namespace,
                   baseline_allse: float,
                   baseline_cell_strain_energy: dict,
                   start_episode: int,
                   worker_cae_file: str):
    """
    Worker进程：运行环境交互并收集经验
    
    Args:
        worker_id: Worker的唯一标识符
        experience_queue: 用于发送经验数据给主进程的队列
        param_queue: 用于从主进程接收模型参数的队列
        command_queue: 用于从主进程接收命令的队列（用于同步）
        stop_event: 停止信号
        args: 命令行参数
        baseline_allse: baseline ALLSE值（由主进程计算）
        start_episode: 起始episode编号
        worker_cae_file: 该worker专属的CAE文件路径
    """
    try:
        print(f"[Worker {worker_id}] Starting with CAE file: {worker_cae_file}")
        
        # 创建环境（使用worker专属的CAE文件）
        env = AbaqusEnv(
            template_cae_file=worker_cae_file,
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
            accuracy_weight=args.accuracy_weight,
            resource_weight=args.resource_weight,
            refine_step_size=args.refine_step_size,
            coarsen_step_size=args.coarsen_step_size
        )
        
        # 设置最大连续失败次数
        env._max_consecutive_failures = args.max_consecutive_failures
        
        # 设置baseline ALLSE（由主进程计算并传递）
        env.baseline_allse = baseline_allse
        # 同时设置 baseline per-cell strain energy（由主进程计算并传递），以便本地 reward 计算使用
        try:
            env.baseline_cell_strain_energy = dict(baseline_cell_strain_energy or {})
            print(f"[Worker {worker_id}] Received baseline_cell_strain_energy for {len(env.baseline_cell_strain_energy)} cells")
        except Exception:
            # 保持兼容性：若传入数据不正确则忽略
            print(f"[Worker {worker_id}] Warning: failed to set baseline_cell_strain_energy from main process")
        
        # 创建本地agent（用于决策）
        agent = DQNAgent(feat_dim=None, hidden_dim=64, num_actions=2)
        
        current_epsilon = args.epsilon
        if start_episode > 0:
            current_epsilon = max(args.epsilon_min, args.epsilon * (args.epsilon_decay ** start_episode))
        
        episode_counter = start_episode
        
        while not stop_event.is_set():
            # 检查是否有新的模型参数
            if not param_queue.empty():
                try:
                    model_data = param_queue.get_nowait()
                    if model_data is not None:
                        # 支持两种格式：完整的dict（带元数据）或仅state_dict
                        if isinstance(model_data, dict) and 'state_dict' in model_data:
                            # 新格式：包含完整的模型信息
                            state_dict = model_data['state_dict']
                            feat_dim = model_data.get('feat_dim')
                            
                            # 如果网络未初始化且有feat_dim，先初始化网络
                            if agent._q_net is None and feat_dim is not None:
                                print(f"[Worker {worker_id}] Initializing network with feat_dim={feat_dim}")
                                agent._ensure_networks(feat_dim)
                            
                            # 加载参数（确保state_dict在正确的设备上）
                            if agent._q_net is not None:
                                # 将state_dict移动到worker的设备上
                                state_dict_device = {k: v.to(agent.device) for k, v in state_dict.items()}
                                agent._q_net.load_state_dict(state_dict_device)
                                agent._tgt_net.load_state_dict(state_dict_device)
                                # 验证参数是否正确加载（检查第一个参数的范数）
                                first_param = next(iter(agent._q_net.parameters()))
                                param_norm = first_param.norm().item()
                                print(f"[Worker {worker_id}] Loaded model parameters (feat_dim={feat_dim}, param_norm={param_norm:.6f})")
                        elif isinstance(model_data, dict):
                            # 旧格式：仅state_dict（兼容性）
                            if agent._q_net is not None:
                                state_dict_device = {k: v.to(agent.device) for k, v in model_data.items()}
                                agent._q_net.load_state_dict(state_dict_device)
                                agent._tgt_net.load_state_dict(state_dict_device)
                                print(f"[Worker {worker_id}] Updated model parameters")
                            else:
                                print(f"[Worker {worker_id}] Warning: Cannot load parameters, network not initialized")
                except Exception as e:
                    print(f"[Worker {worker_id}] Error loading parameters: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 运行一个episode
            run_id = f"run_w{worker_id}_{episode_counter:03d}"
            print(f"[Worker {worker_id}] Starting episode {episode_counter} (run_id: {run_id}, epsilon: {current_epsilon:.4f})")
            
            env.reset(run_id=run_id)
            
            done = False
            t = 0
            episode_rewards = []
            
            while not done and not stop_event.is_set():
                t += 1
                
                # 获取观测
                cell_observations = env.get_cell_observations()
                cell_adjacency = env.cell_adjacency
                
                state_cell_observations = cell_observations.copy()
                
                if not cell_observations:
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
                        print(f"[Worker {worker_id}] No selectable cell at step {t}, terminating episode")
                    done = True
                    continue
                
                selected_cell_id = selection['cell_id']
                selected_action = selection['action']
                actions_dict = {selected_cell_id: selected_action}
                
                # 执行动作
                obs, reward, done, info = env.step(actions_dict)
                episode_rewards.append(float(reward))
                
                # 构建经验数据
                try:
                    next_cell_observations = env.get_cell_observations()
                    
                    # 构建图数据
                    s_node_feat, s_edge_idx, s_cell_id_to_index = agent.build_graph_data(
                        state_cell_observations, cell_adjacency
                    )
                    # 创建空的全局特征张量
                    s_global_feat = torch.zeros((1, 0), dtype=torch.float32)
                    
                    # 检查是否发生状态回退（任何类型的失败）
                    state_rollback = info.get('state_rollback', False)
                    
                    if next_cell_observations and not state_rollback:
                        # 正常情况：使用真实的下一状态
                        ns_node_feat, ns_edge_idx, ns_cell_id_to_index = agent.build_graph_data(
                            next_cell_observations, cell_adjacency
                        )
                        ns_global_feat = torch.zeros((1, 0), dtype=torch.float32)
                    else:
                        # 发生失败（状态回退）或episode结束：使用当前状态作为下一状态
                        # 软失败学习策略：next_state = current_state + 负reward → 学习避免该动作
                        ns_node_feat, ns_edge_idx, ns_cell_id_to_index = (
                            s_node_feat.clone(), s_edge_idx.clone(), s_cell_id_to_index.copy()
                        )
                        ns_global_feat = s_global_feat.clone()
                    
                    if s_node_feat.shape[0] != ns_node_feat.shape[0]:
                        ns_node_feat, ns_edge_idx, ns_cell_id_to_index = (
                            s_node_feat.clone(), s_edge_idx.clone(), s_cell_id_to_index.copy()
                        )
                        ns_global_feat = s_global_feat.clone()
                    
                    # 将actions_dict转换为tensor
                    if s_node_feat.shape[0] > 0:
                        actions_tensor = torch.zeros(s_node_feat.shape[0], dtype=torch.long)
                        action_mask = torch.zeros(s_node_feat.shape[0], dtype=torch.bool)
                        
                        selected_idx = s_cell_id_to_index.get(selected_cell_id)
                        if selected_idx is None or selected_idx >= actions_tensor.shape[0]:
                            if args.debug:
                                print(f"[Worker {worker_id}] Selected cell {selected_cell_id} not found, skipping experience")
                            continue
                        
                        actions_tensor[selected_idx] = selected_action
                        action_mask[selected_idx] = True
                        
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
                        
                        # 发送经验到主进程
                        experience = {
                            'worker_id': worker_id,
                            'episode': episode_counter,
                            'step': t,
                            'node_features': s_node_feat,
                            'edge_index': s_edge_idx,
                            'global_feat': s_global_feat,
                            'actions': actions_tensor,
                            'reward': reward_to_save,
                            'next_node_features': ns_node_feat,
                            'next_edge_index': ns_edge_idx,
                            'next_global_feat': ns_global_feat,
                            'done': done,
                            'action_mask': action_mask
                        }
                        experience_queue.put(experience)
                
                except Exception as e:
                    if args.debug:
                        print(f"[Worker {worker_id}] Error collecting experience: {e}")
                        traceback.print_exc()
                
                # 限制每个episode的最大步数
                if t >= 500:
                    done = True
            
            # Episode结束，发送统计信息
            if episode_rewards:
                episode_stats = {
                    'type': 'episode_stats',
                    'worker_id': worker_id,
                    'episode': episode_counter,
                    'total_reward': sum(episode_rewards),
                    'mean_reward': sum(episode_rewards) / len(episode_rewards),
                    'max_reward': max(episode_rewards),
                    'min_reward': min(episode_rewards),
                    'num_steps': len(episode_rewards),
                    'epsilon': current_epsilon
                }
                experience_queue.put(episode_stats)
                
                print(f"[Worker {worker_id}] Episode {episode_counter} completed: "
                      f"Total reward={episode_stats['total_reward']:.6f}, "
                      f"Mean reward={episode_stats['mean_reward']:.6f}, "
                      f"Steps={episode_stats['num_steps']}")
            
            # 等待主进程的信号再继续下一个episode
            print(f"[Worker {worker_id}] Waiting for main process signal to continue...")
            try:
                command = command_queue.get(timeout=60)  # 最多等待60秒
                if command == 'continue':
                    print(f"[Worker {worker_id}] Received continue signal, starting next episode")
                elif command == 'stop':
                    print(f"[Worker {worker_id}] Received stop signal")
                    break
            except:
                print(f"[Worker {worker_id}] Timeout waiting for signal, stopping")
                break
            
            # 更新计数器和epsilon
            episode_counter += 1
            current_epsilon = max(args.epsilon_min, current_epsilon * args.epsilon_decay)
        
        print(f"[Worker {worker_id}] Stopped")
    
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        traceback.print_exc()


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
    """从checkpoint目录加载训练状态"""
    agent_path = os.path.join(ckpt_dir, 'agent_checkpoint.pt')
    buffer_path = os.path.join(ckpt_dir, 'replay_buffer.pt')
    state_path = os.path.join(ckpt_dir, 'training_state.json')
    
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent checkpoint not found: {agent_path}")
    if not os.path.exists(buffer_path):
        raise FileNotFoundError(f"Replay buffer checkpoint not found: {buffer_path}")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Training state not found: {state_path}")
    
    agent.load_checkpoint(agent_path)
    print(f"  Loaded agent checkpoint from {agent_path}")
    
    replay_buffer.load(buffer_path)
    print(f"  Loaded replay buffer from {buffer_path} (size: {len(replay_buffer)})")
    
    with open(state_path, 'r') as f:
        training_state = json.load(f)
    episode = training_state.get('episode', 0)
    print(f"  Loaded training state from {state_path} (resuming from episode {episode + 1})")
    
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
    
    # 设置多进程启动方法（Windows需要使用spawn）
    mp.set_start_method('spawn', force=True)
    
    print(f"{'='*60}")
    print(f"Multi-Process RL Training Configuration")
    print(f"{'='*60}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Max episodes: {args.max_episodes}")
    if args.sync_mode == 'episode':
        print(f"Sync mode: Every {args.sync_frequency} episode(s)")
    else:
        print(f"Sync mode: Every {args.sync_frequency} training step(s)")
    print(f"{'='*60}\n")
    
    # 初始化DQN Agent（主进程）
    agent = DQNAgent(feat_dim=None, hidden_dim=64, num_actions=2)
    
    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(capacity=100_000)
    
    # 恢复训练（如果需要）
    start_episode = 0
    reward_history = []
    loss_history = []
    if args.resume:
        if not os.path.exists(args.ckpt_dir):
            print(f"Warning: Checkpoint directory '{args.ckpt_dir}' does not exist. Starting from scratch.")
        else:
            try:
                start_episode, reward_history, loss_history = load_training_state(
                    args.ckpt_dir, agent, replay_buffer, args.reward_history_file
                )
                # 验证checkpoint加载后的参数
                if agent._q_net is not None:
                    first_param = next(iter(agent._q_net.parameters()))
                    loaded_param_norm = first_param.norm().item()
                    print(f"  Verified loaded checkpoint param_norm: {loaded_param_norm:.6f}")
                    if loaded_param_norm < 1e-6:
                        print(f"  WARNING: Loaded parameters appear to be near zero!")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch.")
                start_episode = 0
                reward_history = []
                loss_history = []
    
    # 计算baseline ALLSE（只在主进程计算一次）
    print(f"\n{'='*60}")
    print(f"Computing baseline ALLSE before training starts...")
    print(f"{'='*60}")
    
    temp_env = AbaqusEnv(
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
        accuracy_weight=args.accuracy_weight,
        resource_weight=args.resource_weight,
        refine_step_size=args.refine_step_size,
        coarsen_step_size=args.coarsen_step_size
    )
    
    baseline_cache_dir = os.path.join(args.ckpt_dir, 'baseline_cache')
    use_baseline_cache = not args.no_baseline_cache
    if use_baseline_cache:
        print(f"Baseline cache enabled (cache dir: {baseline_cache_dir})")
    else:
        print(f"Baseline cache disabled (--no-baseline-cache flag set)")
    
    baseline_mesh_size = args.baseline_mesh_size if args.baseline_mesh_size is not None else args.global_mesh_size * (1/3)
    print(f"Baseline mesh size: {baseline_mesh_size}")
    baseline_allse = temp_env.compute_baseline(cache_dir=baseline_cache_dir, use_cache=use_baseline_cache, baseline_mesh_size=baseline_mesh_size)
    
    if baseline_allse is None:
        print("Warning: Failed to compute baseline ALLSE. Training may not work correctly.")
        baseline_allse = 1.0  # 使用默认值
    else:
        print(f"Baseline ALLSE successfully computed: {baseline_allse}")
    # 取出 baseline 时的每个 cell 的 strain energy（如果有），并传递给 workers
    baseline_cell_strain_energy = getattr(temp_env, 'baseline_cell_strain_energy', {}) or {}
    print(f"Baseline per-cell strain energy entries: {len(baseline_cell_strain_energy)}")
    print(f"{'='*60}\n")
    
    # 为每个worker准备独立的CAE文件副本（避免多进程读取冲突）
    print(f"\nPreparing CAE file copies for {args.num_workers} workers...")
    worker_cae_files = prepare_worker_cae_files(
        template_cae_file=args.template_cae_file,
        num_workers=args.num_workers,
        work_dir='worker_cae_files'
    )
    print(f"CAE files prepared.\n")
    
    # 创建进程间通信队列
    experience_queue = Queue(maxsize=1000)  # 经验数据队列
    param_queues = [Queue() for _ in range(args.num_workers)]  # 每个worker一个参数队列
    command_queues = [Queue() for _ in range(args.num_workers)]  # 每个worker一个命令队列（用于同步）
    stop_event = Event()  # 停止信号
    
    # 启动worker进程
    workers = []
    for worker_id in range(args.num_workers):
        worker = Process(
            target=worker_process,
            args=(worker_id, experience_queue, param_queues[worker_id], command_queues[worker_id],
                  stop_event, args, baseline_allse, baseline_cell_strain_energy, start_episode, worker_cae_files[worker_id])
        )
        worker.start()
        workers.append(worker)
        print(f"Started worker {worker_id} (PID: {worker.pid})")
    
    print(f"\nAll {args.num_workers} workers started. Beginning training...\n")
    
    # 【关键修复】在启动workers后立即同步模型参数（如果从checkpoint恢复训练）
    # 这确保workers使用的是加载的checkpoint参数，而不是随机初始化的参数
    if agent._q_net is not None:
        print(f"[Main] Synchronizing loaded checkpoint parameters to all workers...")
        # 验证主进程的参数是否正确加载
        first_param = next(iter(agent._q_net.parameters()))
        main_param_norm = first_param.norm().item()
        print(f"[Main] Main process model param_norm: {main_param_norm:.6f}")
        
        # 发送完整的模型信息（包括feat_dim等元数据），以便worker能正确初始化网络
        # 注意：将state_dict中的所有张量移到CPU并detach，确保多进程传递时数据正确
        state_dict_cpu = {k: v.cpu().detach().clone() for k, v in agent._q_net.state_dict().items()}
        
        # 验证CPU副本是否正确
        first_key = next(iter(state_dict_cpu.keys()))
        cpu_param_norm = state_dict_cpu[first_key].norm().item()
        print(f"[Main] CPU state_dict param_norm: {cpu_param_norm:.6f}")
        
        model_sync_data = {
            'state_dict': state_dict_cpu,
            'feat_dim': agent.feat_dim,
            'hidden_dim': agent._hidden_dim,
            'num_actions': agent._num_actions,
            'global_dim': agent._global_dim
        }
        for worker_id, param_queue in enumerate(param_queues):
            try:
                param_queue.put(model_sync_data)
                print(f"[Main] Sent initial parameters to worker {worker_id}")
            except Exception as e:
                print(f"[Main] Warning: Failed to send initial parameters to worker {worker_id}: {e}")
        print(f"[Main] Initial parameter synchronization completed\n")
    else:
        print(f"[Main] Warning: Agent network not initialized, skipping initial sync\n")
    
    # 主循环：基于episode的同步训练
    total_episodes_completed = start_episode
    episode_stats_buffer = []  # 缓存episode统计信息
    last_sync_episode = start_episode
    last_sync_training_step = 0
    training_step_counter = 0
    
    try:
        while total_episodes_completed < args.max_episodes:
            # 收集一个完整episode的经验
            print(f"\n[Main] Waiting for an episode to complete... ({total_episodes_completed}/{args.max_episodes} episodes done)")
            
            episode_experiences = []
            episode_stat = None
            completed_worker_id = None
            
            # 持续收集直到遇到episode_stats（表示一个episode完成）
            while True:
                try:
                    item = experience_queue.get(timeout=120)  # 最多等待120秒
                    
                    if isinstance(item, dict):
                        if item.get('type') == 'episode_stats':
                            # 收到episode统计信息，说明这个episode完成了
                            episode_stat = item
                            completed_worker_id = item['worker_id']
                            total_episodes_completed = max(total_episodes_completed, item['episode'] + 1)
                            
                            # 添加到reward历史
                            reward_history.append({
                                'episode': item['episode'],
                                'worker_id': item['worker_id'],
                                'total_reward': item['total_reward'],
                                'mean_reward': item['mean_reward'],
                                'max_reward': item['max_reward'],
                                'min_reward': item['min_reward'],
                                'num_steps': item['num_steps'],
                                'epsilon': item['epsilon']
                            })
                            episode_stats_buffer.append(item)
                            
                            print(f"[Main] Collected episode {item['episode']} from worker {completed_worker_id}: "
                                  f"{len(episode_experiences)} experiences, "
                                  f"total_reward={item['total_reward']:.6f}")
                            break  # episode完成，退出收集循环
                        else:
                            # 经验数据
                            episode_experiences.append(item)
                
                except Exception as e:
                    print(f"[Main] Timeout or error waiting for episode: {e}")
                    break
            
            # 如果没有收集到完整的episode，跳过
            if episode_stat is None or len(episode_experiences) == 0:
                print(f"[Main] Warning: No complete episode collected, skipping training")
                continue
            
            # 将经验添加到replay buffer
            for exp in episode_experiences:
                replay_buffer.add(
                    node_features=exp['node_features'],
                    edge_index=exp['edge_index'],
                    global_feat=exp['global_feat'],
                    actions=exp['actions'],
                    reward=exp['reward'],
                    next_node_features=exp['next_node_features'],
                    next_edge_index=exp['next_edge_index'],
                    next_global_feat=exp['next_global_feat'],
                    done=exp['done'],
                    action_mask=exp.get('action_mask')
                )
            
            print(f"[Main] Added {len(episode_experiences)} experiences to replay buffer (buffer size: {len(replay_buffer)})")
            
            # 训练模型
            if len(replay_buffer) >= args.batch_size:
                training_losses = []
                # 根据episode长度动态调整训练次数
                num_batches = max(1, len(episode_experiences) // args.batch_size)
                num_batches = min(num_batches, 20)  # 最多20个batch
                
                print(f"[Main] Starting training with {num_batches} batches...")
                for _ in range(num_batches):
                    loss = agent.train_step(replay_buffer, batch_size=args.batch_size)
                    if loss is not None:
                        training_losses.append(loss)
                
                training_step_counter += 1
                
                if training_losses:
                    avg_loss = sum(training_losses) / len(training_losses)
                    # 记录loss到历史
                    loss_history.append({
                        'training_step': training_step_counter,
                        'episode': total_episodes_completed,
                        'avg_loss': avg_loss,
                        'min_loss': min(training_losses),
                        'max_loss': max(training_losses),
                        'num_batches': num_batches,
                        'buffer_size': len(replay_buffer)
                    })
                    print(f"[Training] Step {training_step_counter}: "
                          f"Avg loss={avg_loss:.6f}, "
                          f"Batches trained={num_batches}, "
                          f"Buffer size={len(replay_buffer)}, "
                          f"Episodes completed={total_episodes_completed}/{args.max_episodes}")
            else:
                print(f"[Main] Replay buffer too small ({len(replay_buffer)} < {args.batch_size}), skipping training")
            
            # 发送信号给完成episode的worker，让它继续
            if completed_worker_id is not None:
                try:
                    command_queues[completed_worker_id].put('continue', timeout=1)
                    print(f"[Main] Sent 'continue' signal to worker {completed_worker_id}")
                except:
                    print(f"[Main] Warning: Failed to send signal to worker {completed_worker_id}")
            
            # 定期同步模型参数到workers
            should_sync = False
            if args.sync_mode == 'episode':
                # 按episode数同步
                should_sync = (total_episodes_completed - last_sync_episode >= args.sync_frequency)
            elif args.sync_mode == 'training':
                # 按训练步数同步
                should_sync = (training_step_counter - last_sync_training_step >= args.sync_frequency)
            
            if should_sync:
                if agent._q_net is not None:
                    # 使用新格式发送完整的模型信息
                    # 将state_dict中的所有张量移到CPU并detach，确保多进程传递时数据正确
                    state_dict_cpu = {k: v.cpu().detach().clone() for k, v in agent._q_net.state_dict().items()}
                    model_sync_data = {
                        'state_dict': state_dict_cpu,
                        'feat_dim': agent.feat_dim,
                        'hidden_dim': agent._hidden_dim,
                        'num_actions': agent._num_actions,
                        'global_dim': agent._global_dim
                    }
                    for param_queue in param_queues:
                        try:
                            # 非阻塞发送，如果队列满了就跳过
                            if param_queue.empty():
                                param_queue.put(model_sync_data)
                        except:
                            pass
                    if args.sync_mode == 'episode':
                        print(f"[Main] Synchronized model parameters to all workers (episode {total_episodes_completed})")
                        last_sync_episode = total_episodes_completed
                    else:
                        print(f"[Main] Synchronized model parameters to all workers (training step {training_step_counter})")
                        last_sync_training_step = training_step_counter
            
            # 打印episode统计信息
            if episode_stats_buffer:
                recent_stats = episode_stats_buffer[-min(10, len(episode_stats_buffer)):]
                avg_reward = sum(s['mean_reward'] for s in recent_stats) / len(recent_stats)
                print(f"\n[Progress] Episodes: {total_episodes_completed}/{args.max_episodes}, "
                      f"Recent avg reward: {avg_reward:.6f}, "
                      f"Buffer size: {len(replay_buffer)}\n")
                episode_stats_buffer = episode_stats_buffer[-50:]  # 保留最近50个
            
            # 定期保存checkpoint
            if args.ckpt_dir and total_episodes_completed % args.save_frequency == 0 and total_episodes_completed > start_episode:
                print(f"\n[Main] Saving checkpoint at episode {total_episodes_completed}...")
                try:
                    # 验证保存前的参数（确保已经训练更新）
                    if agent._q_net is not None:
                        first_param = next(iter(agent._q_net.parameters()))
                        save_param_norm = first_param.norm().item()
                        print(f"[Main] Saving model with param_norm: {save_param_norm:.6f}")
                    save_training_state(args.ckpt_dir, agent, replay_buffer, total_episodes_completed, args, reward_history, loss_history)
                    print(f"[Main] Checkpoint saved successfully\n")
                except Exception as e:
                    print(f"[Main] Warning: Failed to save checkpoint: {e}\n")
    
    except KeyboardInterrupt:
        print("\n[Main] Received interrupt signal, stopping workers...")
    
    finally:
        # 停止所有worker
        stop_event.set()
        
        # 发送停止信号到所有command队列
        print("\n[Main] Sending stop signals to all workers...")
        for i, cmd_queue in enumerate(command_queues):
            try:
                cmd_queue.put('stop', timeout=1)
            except:
                pass
        
        # 等待所有worker结束
        for i, worker in enumerate(workers):
            worker.join(timeout=5)
            if worker.is_alive():
                print(f"[Main] Worker {i} did not stop gracefully, terminating...")
                worker.terminate()
                worker.join()
        
        print("\n[Main] All workers stopped")
        
        # 清理worker的CAE文件副本
        print("\n[Main] Cleaning up worker CAE files...")
        cleanup_worker_cae_files('worker_cae_files')
        
        # 保存最终checkpoint
        if args.ckpt_dir:
            print(f"\n[Main] Saving final checkpoint...")
            try:
                # 验证保存前的参数
                if agent._q_net is not None:
                    first_param = next(iter(agent._q_net.parameters()))
                    final_param_norm = first_param.norm().item()
                    print(f"[Main] Saving final model with param_norm: {final_param_norm:.6f}")
                save_training_state(args.ckpt_dir, agent, replay_buffer, total_episodes_completed, args, reward_history, loss_history)
                print(f"[Main] Final checkpoint saved successfully")
            except Exception as e:
                print(f"[Main] Warning: Failed to save final checkpoint: {e}")
        
        # 打印总结
        if reward_history:
            all_mean_rewards = [h['mean_reward'] for h in reward_history]
            print(f"\n{'='*60}")
            print(f"Training Summary:")
            print(f"  Total episodes completed: {len(reward_history)}")
            print(f"  Overall mean reward: {sum(all_mean_rewards) / len(all_mean_rewards):.6f}")
            print(f"  Best episode mean reward: {max(all_mean_rewards):.6f}")
            print(f"  Worst episode mean reward: {min(all_mean_rewards):.6f}")
            print(f"  Final buffer size: {len(replay_buffer)}")
            print(f"{'='*60}")


if __name__ == '__main__':
    main()

