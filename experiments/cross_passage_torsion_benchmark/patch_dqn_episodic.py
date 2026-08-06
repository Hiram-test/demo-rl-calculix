#!/usr/bin/env python3  # 使用当前 Python 解释器执行一次性 DQN 训练流程修正
from pathlib import Path  # 读取和覆盖横向通道实验主程序
import re  # 以函数边界稳定替换旧的伪在线 DQN 实现

source_path = Path('experiments/cross_passage_torsion_benchmark/run_benchmark.py')  # 定位实验主程序
source = source_path.read_text(encoding='utf-8')  # 读取当前分支源码全文

constant_anchor = "TOP_HOTSPOT_COUNT = 4  # 使用参考解能量最大的四个图区域评价热点命中率\n"  # 定义常量插入锚点
constant_block = """TOP_HOTSPOT_COUNT = 4  # 使用参考解能量最大的四个图区域评价热点命中率
DQN_TRAIN_EPISODES = 128  # 为每个独立随机种子执行一百二十八个完整训练回合
DQN_EPISODE_STEPS = 12  # 将每个训练回合限制为十二次确定性网格状态转移
DQN_TRAIN_SEEDS = (SEED + 101, SEED + 202, SEED + 303)  # 使用三个独立随机种子评价训练稳定性
DQN_REPLAY_CAPACITY = 8192  # 保存跨回合经验并限制内存占用
DQN_BATCH_SIZE = 64  # 每次梯度更新抽取六十四条经验
DQN_REPLAY_WARMUP = 128  # 经验池达到一百二十八条后开始参数更新
DQN_TARGET_SYNC_UPDATES = 100  # 每一百次梯度更新同步目标网络
DQN_EVALUATION_SOLVE_BUDGET = 32  # 冻结策略后为每个随机种子提供三十二次独立真实求解
"""  # 定义新的正式 DQN 训练与评测常量
if 'DQN_TRAIN_EPISODES = 128' not in source:  # 检查是否尚未插入正式训练常量
    if constant_anchor not in source:  # 验证旧常量锚点存在
        raise RuntimeError('DQN constant anchor not found')  # 在未知源码上拒绝静默修改
    source = source.replace(constant_anchor, constant_block, 1)  # 插入训练回合、随机种子和回放参数

new_function = '''    def run_dqn_gcn(self) -> MethodResult:  # 先完成多回合独立训练，再冻结 GCN-DQN 并使用统一真实求解预算评测
        training_cache_start = len(self.benchmark.cache)  # 记录 DQN 训练开始前已有候选数量
        trained_states: list[dict[str, torch.Tensor]] = []  # 保存三个独立随机种子的冻结网络参数
        training_summaries: list[dict[str, Any]] = []  # 保存每个随机种子的训练回合与损失摘要
        for seed_index, training_seed in enumerate(DQN_TRAIN_SEEDS):  # 逐个独立随机种子执行完整 episodic training
            random.seed(training_seed)  # 固定当前随机种子的 Python 探索过程
            np.random.seed(training_seed)  # 固定当前随机种子的 NumPy 重置状态
            torch.manual_seed(training_seed)  # 固定当前随机种子的网络初始化和批次采样
            online = GraphQNetwork(6, 64, 2, self.benchmark.normalized_adjacency)  # 创建当前种子的在线 GCN-Q 网络
            target = GraphQNetwork(6, 64, 2, self.benchmark.normalized_adjacency)  # 创建当前种子的目标 GCN-Q 网络
            target.load_state_dict(online.state_dict())  # 同步初始在线网络和目标网络参数
            optimizer = torch.optim.Adam(online.parameters(), lr=1.0e-3)  # 使用较稳定学习率创建 Adam 优化器
            replay_items: list[tuple[np.ndarray, int, int, float, np.ndarray, np.ndarray, float]] = []  # 初始化包含终止标志的跨回合经验池
            episode_returns: list[float] = []  # 保存每个训练回合累计奖励
            terminal_objectives: list[float] = []  # 保存每个训练回合终止目标值
            losses: list[float] = []  # 保存全部参数更新损失
            gradient_updates = 0  # 初始化当前种子的梯度更新次数
            seed_cache_start = len(self.benchmark.cache)  # 记录当前种子开始前训练缓存规模
            for episode in range(DQN_TRAIN_EPISODES):  # 执行规定数量的完整训练回合
                if episode % 5 == 0:  # 周期性使用与最终评测一致的统一二分段起点
                    current_levels = tuple(1 for _ in range(self.benchmark.region_count))  # 构造固定训练起点
                else:  # 其余回合从随机可行网格状态开始以扩大覆盖范围
                    restart_rng = np.random.default_rng(training_seed + episode)  # 为当前回合创建可复现重置随机数发生器
                    restart = restart_rng.integers(0, 3, size=self.benchmark.region_count)  # 生成零到二级随机区域网格
                    current_levels = self.benchmark.repair_levels(restart, None)  # 将随机网格修复到三百三十六单元硬上限内
                current_solution = self.benchmark.solve(current_levels)  # 获取当前回合初始状态真实有限元响应
                current_objective, _, _, _, _ = self.benchmark.metrics(current_solution)  # 计算初始状态精度目标
                episode_return = 0.0  # 初始化当前回合累计奖励
                for step in range(DQN_EPISODE_STEPS):  # 在固定回合长度内执行确定性环境转移
                    state_features = self.benchmark.region_features(current_solution, current_levels)  # 构造当前图状态节点特征
                    priority = self.benchmark.hotspot_priority(current_solution, current_levels)  # 构造预算修复使用的区域优先级
                    valid_mask = self._valid_action_mask(current_levels, priority)  # 构造当前状态有效动作掩码
                    valid_pairs = np.argwhere(valid_mask)  # 提取全部有效区域动作对
                    if len(valid_pairs) == 0:  # 检查是否到达无可执行动作终止状态
                        break  # 提前结束当前训练回合
                    progress = (seed_index * DQN_TRAIN_EPISODES + episode) / float(len(DQN_TRAIN_SEEDS) * DQN_TRAIN_EPISODES - 1)  # 计算跨种子总体训练进度
                    epsilon = 0.05 + 0.90 * (1.0 - progress)  # 将探索率从零点九五线性衰减到零点零五
                    if random.random() < epsilon:  # 按当前探索率执行随机有效动作
                        selected_pair = valid_pairs[random.randrange(len(valid_pairs))]  # 随机选择一个有效区域动作
                        region = int(selected_pair[0])  # 读取动作区域编号
                        action_index = int(selected_pair[1])  # 读取粗化或细化动作编号
                    else:  # 使用当前在线网络执行贪心动作
                        with torch.no_grad():  # 禁用动作选择阶段梯度
                            q_values = online(torch.tensor(state_features[None, :, :], dtype=torch.float32))[0].numpy()  # 计算当前图全部动作 Q 值
                        q_values[~valid_mask] = -1.0e30  # 屏蔽全部无效动作
                        flat_index = int(np.argmax(q_values))  # 选择全图最大有效 Q 值动作
                        region, action_index = np.unravel_index(flat_index, q_values.shape)  # 还原动作区域和类型
                    candidate = np.asarray(current_levels, dtype=np.float64)  # 复制当前级别向量
                    candidate[region] += -1.0 if action_index == 0 else 1.0  # 应用当前粗化或细化动作
                    candidate_levels = self.benchmark.repair_levels(candidate, priority)  # 将候选修复到统一单元上限内
                    candidate_solution = self.benchmark.solve(candidate_levels)  # 执行或读取候选真实 CalculiX 响应
                    candidate_objective, _, _, _, _ = self.benchmark.metrics(candidate_solution)  # 计算候选精度目标
                    done = float(step + 1 >= DQN_EPISODE_STEPS)  # 在固定步数末端写入正式终止标志
                    reward = 8.0 * (current_objective - candidate_objective)  # 仅按精度目标改进定义即时奖励
                    next_features = self.benchmark.region_features(candidate_solution, candidate_levels)  # 构造实际下一状态图特征
                    next_priority = self.benchmark.hotspot_priority(candidate_solution, candidate_levels)  # 构造实际下一状态优先级
                    next_mask = self._valid_action_mask(candidate_levels, next_priority)  # 构造实际下一状态动作掩码
                    replay_items.append((state_features.copy(), int(region), int(action_index), float(reward), next_features.copy(), next_mask.copy(), done))  # 将真实且一致的环境转移加入经验池
                    if len(replay_items) > DQN_REPLAY_CAPACITY:  # 检查经验池是否超过固定容量
                        replay_items.pop(0)  # 删除最早经验以保持固定容量
                    current_levels = candidate_levels  # 无条件接受动作并更新实际环境状态
                    current_solution = candidate_solution  # 更新实际环境有限元响应
                    current_objective = candidate_objective  # 更新实际环境精度目标
                    episode_return += reward  # 累加当前回合即时奖励
                    if len(replay_items) >= DQN_REPLAY_WARMUP:  # 在经验池充分预热后执行一次梯度更新
                        batch = random.sample(replay_items, DQN_BATCH_SIZE)  # 从跨回合经验池均匀抽取训练批次
                        state_batch = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)  # 组成当前状态批次
                        region_batch = torch.tensor([item[1] for item in batch], dtype=torch.int64)  # 组成动作区域批次
                        action_batch = torch.tensor([item[2] for item in batch], dtype=torch.int64)  # 组成动作类型批次
                        reward_batch = torch.tensor([item[3] for item in batch], dtype=torch.float32)  # 组成即时奖励批次
                        next_batch = torch.tensor(np.stack([item[4] for item in batch]), dtype=torch.float32)  # 组成实际下一状态批次
                        mask_batch = torch.tensor(np.stack([item[5] for item in batch]), dtype=torch.bool)  # 组成下一状态动作掩码批次
                        done_batch = torch.tensor([item[6] for item in batch], dtype=torch.float32)  # 组成正式终止标志批次
                        q_batch = online(state_batch)  # 计算当前状态全部 Q 值
                        selected_q = q_batch[torch.arange(len(batch)), region_batch, action_batch]  # 提取实际执行动作 Q 值
                        with torch.no_grad():  # 禁用 Bellman 目标计算梯度
                            next_online = online(next_batch).masked_fill(~mask_batch, -1.0e30)  # 用在线网络选择下一状态动作
                            next_flat = next_online.view(len(batch), -1).argmax(dim=1)  # 获取 Double DQN 下一动作扁平编号
                            next_region = torch.div(next_flat, 2, rounding_mode='floor')  # 还原下一动作区域编号
                            next_action = next_flat % 2  # 还原下一动作类型编号
                            next_target = target(next_batch)[torch.arange(len(batch)), next_region, next_action]  # 用目标网络评价下一动作
                            target_q = reward_batch + 0.92 * (1.0 - done_batch) * next_target  # 使用终止标志形成正确的一步 Bellman 目标
                        loss = F.smooth_l1_loss(selected_q, target_q)  # 计算稳定的平滑 L1 损失
                        optimizer.zero_grad()  # 清空上一批次梯度
                        loss.backward()  # 反向传播当前批次损失
                        torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)  # 裁剪梯度以稳定有限元小样本训练
                        optimizer.step()  # 更新在线 GCN-Q 网络参数
                        gradient_updates += 1  # 累加当前种子梯度更新次数
                        losses.append(float(loss.detach().cpu().item()))  # 保存当前批次损失
                        if gradient_updates % DQN_TARGET_SYNC_UPDATES == 0:  # 按固定更新次数同步目标网络
                            target.load_state_dict(online.state_dict())  # 将在线网络参数复制到目标网络
                    if done > 0.5:  # 检查是否到达当前回合终止状态
                        break  # 结束当前训练回合
                episode_returns.append(float(episode_return))  # 保存当前回合累计奖励
                terminal_objectives.append(float(current_objective))  # 保存当前回合终止目标
            trained_states.append({name: tensor.detach().cpu().clone() for name, tensor in online.state_dict().items()})  # 冻结并保存当前独立种子网络参数
            training_summaries.append({'seed': int(training_seed), 'episodes': DQN_TRAIN_EPISODES, 'episode_steps': DQN_EPISODE_STEPS, 'transitions': len(replay_items), 'gradient_updates': int(gradient_updates), 'unique_training_solves_added': int(len(self.benchmark.cache) - seed_cache_start), 'mean_episode_return': float(np.mean(episode_returns)), 'final_32_episode_return_mean': float(np.mean(episode_returns[-32:])), 'mean_terminal_objective': float(np.mean(terminal_objectives)), 'final_32_terminal_objective_mean': float(np.mean(terminal_objectives[-32:])), 'mean_loss': float(np.mean(losses)) if losses else None})  # 记录当前种子的完整训练摘要
        training_unique_solves = len(self.benchmark.cache) - training_cache_start  # 统计三个种子共享缓存后的唯一训练求解总数
        if self.benchmark.run_root.exists():  # 检查训练阶段求解证据目录是否存在
            shutil.rmtree(self.benchmark.run_root)  # 删除数千个训练候选文件以避免污染正式评测证据和 Git 提交
        self.benchmark.run_root.mkdir(parents=True, exist_ok=True)  # 重新创建仅保存冻结策略评测的证据目录
        evaluation_results: list[MethodResult] = []  # 保存三个冻结策略的独立评测结果
        evaluation_summaries: list[dict[str, Any]] = []  # 保存三个种子的评测摘要
        for seed_index, training_seed in enumerate(DQN_TRAIN_SEEDS):  # 逐个冻结网络执行独立三十二求解评测
            random.seed(SEED + 10000 + seed_index)  # 固定当前评测的重置与平局过程
            np.random.seed(SEED + 10000 + seed_index)  # 固定当前评测 NumPy 随机状态
            evaluation_network = GraphQNetwork(6, 64, 2, self.benchmark.normalized_adjacency)  # 重建当前种子的评测网络
            evaluation_network.load_state_dict(trained_states[seed_index])  # 加载冻结训练参数
            evaluation_network.eval()  # 切换到冻结推理模式
            self.benchmark.cache = {}  # 清空训练与其他种子缓存以强制独立计算三十二个评测状态
            evaluation_start = len(self.benchmark.cache)  # 记录当前种子评测起点
            best_levels: tuple[int, ...] | None = None  # 初始化当前种子最优级别配置
            best_objective = float('inf')  # 初始化当前种子最优精度目标
            history: list[dict[str, float]] = []  # 初始化当前种子冻结策略评测轨迹
            visited: set[tuple[int, ...]] = set()  # 保存当前种子已评测配置以禁止循环
            evaluation_episode = 0  # 初始化冻结策略评测回合编号
            while len(self.benchmark.cache) - evaluation_start < DQN_EVALUATION_SOLVE_BUDGET and evaluation_episode < 32:  # 使用多个明确评测回合消耗统一三十二求解预算
                if evaluation_episode == 0:  # 第一回合使用统一二分段标准起点
                    current_levels = tuple(1 for _ in range(self.benchmark.region_count))  # 构造标准评测起点
                else:  # 后续回合使用确定性随机可行起点扩大冻结策略覆盖范围
                    restart_rng = np.random.default_rng(SEED + 20000 + seed_index * 100 + evaluation_episode)  # 创建当前评测回合随机数发生器
                    restart = restart_rng.integers(0, 3, size=self.benchmark.region_count)  # 生成零到二级随机网格
                    current_levels = self.benchmark.repair_levels(restart, None)  # 修复到统一单元硬上限内
                if current_levels in visited:  # 检查重置状态是否已经评测
                    evaluation_episode += 1  # 跳过重复重置并推进回合编号
                    continue  # 开始下一评测回合
                current_solution = self.benchmark.solve(current_levels)  # 执行当前评测回合初始真实求解
                current_objective, _, _, _, _ = self.benchmark.metrics(current_solution)  # 计算初始状态精度目标
                visited.add(current_levels)  # 标记初始状态已评测
                if current_objective < best_objective:  # 检查初始状态是否改进当前种子最优值
                    best_objective = current_objective  # 更新当前种子最优目标
                    best_levels = current_levels  # 更新当前种子最优级别配置
                history.append({'evaluation': float(len(self.benchmark.cache) - evaluation_start), 'best_objective': float(best_objective), 'candidate_objective': float(current_objective)})  # 记录当前初始评测点
                for _ in range(7):  # 每个评测回合最多执行七次冻结策略动作并连同起点形成八个状态
                    if len(self.benchmark.cache) - evaluation_start >= DQN_EVALUATION_SOLVE_BUDGET:  # 检查是否已用满统一评测预算
                        break  # 结束当前评测回合
                    state_features = self.benchmark.region_features(current_solution, current_levels)  # 构造当前评测图状态
                    priority = self.benchmark.hotspot_priority(current_solution, current_levels)  # 构造当前状态预算修复优先级
                    valid_mask = self._valid_action_mask(current_levels, priority)  # 构造当前状态有效动作掩码
                    with torch.no_grad():  # 禁用冻结策略推理梯度
                        q_values = evaluation_network(torch.tensor(state_features[None, :, :], dtype=torch.float32))[0].numpy()  # 计算当前状态全部动作 Q 值
                    ranked_actions = np.argsort(q_values.reshape(-1))[::-1]  # 按 Q 值从高到低排列全部区域动作
                    chosen_levels: tuple[int, ...] | None = None  # 初始化当前评测动作候选
                    for flat_index in ranked_actions:  # 依次检查冻结策略偏好的动作
                        region, action_index = np.unravel_index(int(flat_index), q_values.shape)  # 还原动作区域和类型
                        if not valid_mask[region, action_index]:  # 检查动作是否有效
                            continue  # 跳过无效动作
                        candidate = np.asarray(current_levels, dtype=np.float64)  # 复制当前级别向量
                        candidate[region] += -1.0 if action_index == 0 else 1.0  # 应用候选粗化或细化动作
                        candidate_levels = self.benchmark.repair_levels(candidate, priority)  # 修复候选到统一单元上限内
                        if candidate_levels not in visited:  # 检查候选是否为当前种子未评测状态
                            chosen_levels = candidate_levels  # 选择最高 Q 值未访问动作
                            break  # 停止继续检查次优动作
                    if chosen_levels is None:  # 检查当前状态是否不存在未访问有效动作
                        break  # 结束当前评测回合并通过新 reset 继续评测
                    candidate_solution = self.benchmark.solve(chosen_levels)  # 执行冻结策略选中状态真实 CalculiX 求解
                    candidate_objective, _, _, _, _ = self.benchmark.metrics(candidate_solution)  # 计算候选精度目标
                    current_levels = chosen_levels  # 无条件执行冻结策略动作并更新实际状态
                    current_solution = candidate_solution  # 更新当前真实有限元响应
                    current_objective = candidate_objective  # 更新当前精度目标
                    visited.add(current_levels)  # 标记当前配置已独立评测
                    if current_objective < best_objective:  # 检查当前候选是否改进当前种子最优值
                        best_objective = current_objective  # 更新当前种子最优目标
                        best_levels = current_levels  # 更新当前种子最优级别配置
                    history.append({'evaluation': float(len(self.benchmark.cache) - evaluation_start), 'best_objective': float(best_objective), 'candidate_objective': float(current_objective)})  # 记录当前冻结策略评测点
                evaluation_episode += 1  # 完成一个正式冻结策略评测回合
            if best_levels is None:  # 检查当前种子是否至少完成一个真实评测
                raise RuntimeError('frozen DQN policy did not evaluate a state')  # 无有效评测时停止实验
            seed_result = self.benchmark.method_result(f'dqn_gcn_seed_{seed_index + 1}', best_levels, len(self.benchmark.cache) - evaluation_start, history, f'随机种子 {training_seed}：训练 {DQN_TRAIN_EPISODES} 个完整 episode 后冻结网络，并使用独立 {DQN_EVALUATION_SOLVE_BUDGET} 次 CalculiX 求解评测。')  # 生成当前种子统一结果
            evaluation_results.append(seed_result)  # 保存当前种子评测结果
            evaluation_summaries.append({'seed': int(training_seed), 'objective': float(seed_result.objective), 'levels': list(seed_result.levels), 'elements': int(seed_result.solution.element_count), 'unique_evaluation_solves': int(seed_result.unique_solves), 'evaluation_episodes': int(evaluation_episode), 'torque_error': float(seed_result.torque_error), 'energy_error': float(seed_result.energy_error), 'probe_error': float(seed_result.probe_error), 'hotspot_recall': float(seed_result.hotspot_recall)})  # 保存当前种子完整评测摘要
        ordered_results = sorted(evaluation_results, key=lambda item: item.objective)  # 按冻结策略目标值排序三个独立种子
        median_result = ordered_results[len(ordered_results) // 2]  # 选择中位种子避免报告偶然最好结果
        median_result.name = 'dqn_gcn'  # 使用统一方法名称写入五方法对比表
        median_result.notes = f'三个独立随机种子各训练 {DQN_TRAIN_EPISODES} 个完整 episode、每回合最多 {DQN_EPISODE_STEPS} 步；训练后冻结网络，每个种子独立使用 {DQN_EVALUATION_SOLVE_BUDGET} 次真实求解评测，表中报告中位种子。训练阶段共有 {training_unique_solves} 个唯一 CalculiX 状态。'  # 写入训练和评测边界
        training_payload = {'schema': 'episodic-dqn-training-summary', 'training_seed_count': len(DQN_TRAIN_SEEDS), 'episodes_per_seed': DQN_TRAIN_EPISODES, 'steps_per_episode': DQN_EPISODE_STEPS, 'maximum_training_transitions': len(DQN_TRAIN_SEEDS) * DQN_TRAIN_EPISODES * DQN_EPISODE_STEPS, 'unique_training_solves': int(training_unique_solves), 'evaluation_solve_budget_per_seed': DQN_EVALUATION_SOLVE_BUDGET, 'reported_seed_rule': 'median objective across three independently trained frozen policies', 'training': training_summaries, 'evaluation': evaluation_summaries, 'reported_seed': int(DQN_TRAIN_SEEDS[evaluation_results.index(median_result)]) if median_result in evaluation_results else None}  # 构造完整 episodic DQN 训练审计记录
        (self.benchmark.output_root / 'dqn_training.json').write_text(json.dumps(training_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')  # 写出训练回合、求解数量和三种子评测结果
        return median_result  # 返回中位冻结策略作为五方法表中的 DQN 结果
'''

pattern = re.compile(r"    def run_dqn_gcn\(self\) -> MethodResult:.*?^    def run_all\(self\) -> list\[MethodResult\]:", re.MULTILINE | re.DOTALL)  # 定义旧 DQN 函数到下一方法边界的匹配规则
match = pattern.search(source)  # 搜索当前旧 DQN 实现
if match is None:  # 检查函数边界是否成功识别
    raise RuntimeError('run_dqn_gcn block not found')  # 在未知源码结构上拒绝替换
source = source[:match.start()] + new_function + "    def run_all(self) -> list[MethodResult]:" + source[match.end():]  # 用正式 episodic DQN 完整替换旧函数

old_report = "DQN+GCN 的结果同时包含其图表示能力和仅有三十二次真实求解在线训练造成的样本效率限制。"  # 定义旧错误报告语句
new_report = "DQN+GCN 以三个独立随机种子各训练一百二十八个完整 episode，训练后冻结网络，并为每个种子单独提供三十二次真实求解评测预算；表中报告三个冻结策略的中位目标结果。"  # 定义新训练边界说明
if old_report in source:  # 检查旧报告语句是否仍存在
    source = source.replace(old_report, new_report, 1)  # 更新自动生成报告中的 DQN 训练说明
elif new_report not in source:  # 检查报告是否处于未知状态
    raise RuntimeError('DQN report sentence not found')  # 拒绝静默遗漏报告修正

source_path.write_text(source, encoding='utf-8')  # 保存正式 episodic DQN 实现
