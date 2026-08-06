#!/usr/bin/env python3  # 使用当前 Python 解释器修正中位冻结策略的种子定位
from pathlib import Path  # 读取和覆盖已应用正式 DQN 的实验主程序

source_path = Path('experiments/cross_passage_torsion_benchmark/run_benchmark.py')  # 定位实验主程序
source = source_path.read_text(encoding='utf-8')  # 读取当前源码全文
old_line = "        training_payload = {'schema': 'episodic-dqn-training-summary', 'training_seed_count': len(DQN_TRAIN_SEEDS), 'episodes_per_seed': DQN_TRAIN_EPISODES, 'steps_per_episode': DQN_EPISODE_STEPS, 'maximum_training_transitions': len(DQN_TRAIN_SEEDS) * DQN_TRAIN_EPISODES * DQN_EPISODE_STEPS, 'unique_training_solves': int(training_unique_solves), 'evaluation_solve_budget_per_seed': DQN_EVALUATION_SOLVE_BUDGET, 'reported_seed_rule': 'median objective across three independently trained frozen policies', 'training': training_summaries, 'evaluation': evaluation_summaries, 'reported_seed': int(DQN_TRAIN_SEEDS[evaluation_results.index(median_result)]) if median_result in evaluation_results else None}  # 构造完整 episodic DQN 训练审计记录\n"  # 定义会触发 NumPy 数组相等比较的旧摘要行
new_lines = "        reported_seed_index = next(index for index, item in enumerate(evaluation_results) if item is median_result)  # 使用对象身份定位中位冻结策略以避免 NumPy 数组相等比较\n        training_payload = {'schema': 'episodic-dqn-training-summary', 'training_seed_count': len(DQN_TRAIN_SEEDS), 'episodes_per_seed': DQN_TRAIN_EPISODES, 'steps_per_episode': DQN_EPISODE_STEPS, 'maximum_training_transitions': len(DQN_TRAIN_SEEDS) * DQN_TRAIN_EPISODES * DQN_EPISODE_STEPS, 'unique_training_solves': int(training_unique_solves), 'evaluation_solve_budget_per_seed': DQN_EVALUATION_SOLVE_BUDGET, 'reported_seed_rule': 'median objective across three independently trained frozen policies', 'training': training_summaries, 'evaluation': evaluation_summaries, 'reported_seed': int(DQN_TRAIN_SEEDS[reported_seed_index])}  # 构造完整 episodic DQN 训练审计记录\n"  # 定义使用对象身份定位的新摘要代码
if old_line not in source:  # 检查旧摘要代码是否符合预期
    raise RuntimeError('DQN reported seed line not found')  # 在未知源码上拒绝静默修改
source = source.replace(old_line, new_lines, 1)  # 替换中位种子定位逻辑
source_path.write_text(source, encoding='utf-8')  # 保存中位种子定位修正
