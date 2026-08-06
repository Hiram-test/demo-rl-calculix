#!/usr/bin/env python3  # 使用当前 Python 解释器修正 DQN 训练证据、结果定位与随机种子公平性
from pathlib import Path  # 读取和覆盖已应用 episodic DQN 的实验主程序

source_path = Path('experiments/cross_passage_torsion_benchmark/run_benchmark.py')  # 定位实验主程序
source = source_path.read_text(encoding='utf-8')  # 读取当前源码全文
old_start = "        training_cache_start = len(self.benchmark.cache)  # 记录 DQN 训练开始前已有候选数量\n"  # 定义训练起点旧代码
new_start = "        training_cache_start = len(self.benchmark.cache)  # 记录 DQN 训练开始前已有候选数量\n        preserved_workdirs = {solution.workdir for solution in self.benchmark.cache.values()}  # 保存前四种方法已产生的求解证据目录\n"  # 定义同时保存原有证据目录的新代码
if old_start not in source:  # 检查训练起点代码是否符合预期
    raise RuntimeError('DQN training start anchor not found')  # 在未知源码上拒绝静默修改
source = source.replace(old_start, new_start, 1)  # 插入原有求解证据目录快照
old_cleanup = "        if self.benchmark.run_root.exists():  # 检查训练阶段求解证据目录是否存在\n            shutil.rmtree(self.benchmark.run_root)  # 删除数千个训练候选文件以避免污染正式评测证据和 Git 提交\n        self.benchmark.run_root.mkdir(parents=True, exist_ok=True)  # 重新创建仅保存冻结策略评测的证据目录\n"  # 定义会误删前四方法证据的旧清理逻辑
new_cleanup = "        training_only_workdirs = {solution.workdir for solution in self.benchmark.cache.values() if solution.workdir not in preserved_workdirs}  # 找出仅由 DQN 训练新增的候选目录\n        for training_workdir_text in training_only_workdirs:  # 逐个清理 DQN 训练临时候选证据\n            training_workdir = Path(training_workdir_text)  # 将目录文本转换为路径对象\n            if training_workdir.exists():  # 检查训练候选目录是否仍存在\n                shutil.rmtree(training_workdir)  # 仅删除训练新增目录并保留前四种方法证据\n        self.benchmark.run_root.mkdir(parents=True, exist_ok=True)  # 确保冻结策略评测证据根目录存在\n"  # 定义精确保留原有证据的新清理逻辑
if old_cleanup not in source:  # 检查旧清理代码是否符合预期
    raise RuntimeError('DQN cleanup block not found')  # 在未知源码上拒绝修改
source = source.replace(old_cleanup, new_cleanup, 1)  # 替换为仅删除训练新增目录的逻辑
old_progress = "                    progress = (seed_index * DQN_TRAIN_EPISODES + episode) / float(len(DQN_TRAIN_SEEDS) * DQN_TRAIN_EPISODES - 1)  # 计算跨种子总体训练进度\n"  # 定义会使三个种子使用不同探索区间的旧进度代码
new_progress = "                    progress = episode / float(DQN_TRAIN_EPISODES - 1)  # 为每个独立随机种子使用相同的零到一训练进度\n"  # 定义每个种子独立完整衰减的新进度代码
if old_progress not in source:  # 检查旧探索日程是否符合预期
    raise RuntimeError('DQN epsilon progress line not found')  # 在未知源码上拒绝修改
source = source.replace(old_progress, new_progress, 1)  # 使三个独立种子均从零点九五衰减到零点零五
old_payload = "        training_payload = {'schema': 'episodic-dqn-training-summary', 'training_seed_count': len(DQN_TRAIN_SEEDS), 'episodes_per_seed': DQN_TRAIN_EPISODES, 'steps_per_episode': DQN_EPISODE_STEPS, 'maximum_training_transitions': len(DQN_TRAIN_SEEDS) * DQN_TRAIN_EPISODES * DQN_EPISODE_STEPS, 'unique_training_solves': int(training_unique_solves), 'evaluation_solve_budget_per_seed': DQN_EVALUATION_SOLVE_BUDGET, 'reported_seed_rule': 'median objective across three independently trained frozen policies', 'training': training_summaries, 'evaluation': evaluation_summaries, 'reported_seed': int(DQN_TRAIN_SEEDS[evaluation_results.index(median_result)]) if median_result in evaluation_results else None}  # 构造完整 episodic DQN 训练审计记录\n"  # 定义会触发 NumPy 数组相等比较的旧摘要代码
new_payload = "        reported_seed_index = next(index for index, item in enumerate(evaluation_results) if item is median_result)  # 使用对象身份定位中位冻结策略以避免 NumPy 数组相等比较\n        training_payload = {'schema': 'episodic-dqn-training-summary', 'training_seed_count': len(DQN_TRAIN_SEEDS), 'episodes_per_seed': DQN_TRAIN_EPISODES, 'steps_per_episode': DQN_EPISODE_STEPS, 'maximum_training_transitions': len(DQN_TRAIN_SEEDS) * DQN_TRAIN_EPISODES * DQN_EPISODE_STEPS, 'unique_training_solves': int(training_unique_solves), 'evaluation_solve_budget_per_seed': DQN_EVALUATION_SOLVE_BUDGET, 'reported_seed_rule': 'median objective across three independently trained frozen policies', 'training': training_summaries, 'evaluation': evaluation_summaries, 'reported_seed': int(DQN_TRAIN_SEEDS[reported_seed_index])}  # 构造完整 episodic DQN 训练审计记录\n"  # 定义使用对象身份定位的新摘要代码
if old_payload not in source:  # 检查旧训练摘要代码是否符合预期
    raise RuntimeError('DQN training payload line not found')  # 在未知源码上拒绝修改
source = source.replace(old_payload, new_payload, 1)  # 修正中位冻结策略的种子定位
source_path.write_text(source, encoding='utf-8')  # 保存全部正式 DQN 训练修正
