#!/usr/bin/env python3  # 使用当前 Python 解释器修正 DQN 训练证据清理范围
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
source_path.write_text(source, encoding='utf-8')  # 保存证据清理修正
