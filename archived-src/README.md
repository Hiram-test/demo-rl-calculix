# DRL Adaptive Mesh Demo

本压缩包为“基于深度强化学习的有限元自适应网格划分”课程作业代码包。项目以 Abaqus/CAE 模型为计算环境，通过 DQN 智能体根据有限元反馈调整局部网格密度，并用 ALLSE、单元数量和最大位移等指标评价训练效果。

## 1. 运行环境

- Windows
- Abaqus/CAE，可通过命令行调用 `abaqus`
- Python 依赖主要由 Abaqus Python 环境和本目录脚本提供
- 如系统中 `abaqus` 命令不可直接调用，可在运行前通过环境变量 `ABAQUS_CMD` 指定 Abaqus 命令路径

## 2. 主要文件

- `DEMO.cae`：Abaqus 示例模型
- `DEMO.jnl`：Abaqus 建模/操作日志
- `rl_main.py`：DQN 训练主入口
- `rl_main_multiprocess.py`：多进程版本训练入口
- `dqn_agent.py`：DQN 智能体、Q 网络和经验回放
- `abaqus_env.py`：强化学习环境封装，负责 reset、step、奖励计算和状态组织
- `mesh_generation.py`：根据 cell 网格密度生成新的 Abaqus 网格
- `run_fea_analysis.py`：提交 Abaqus 有限元分析任务
- `extract_results.py`：从 ODB 中提取 ALLSE、单元数量等结果
- `get_max_displacement.py` / `run_max_displacement.py`：提取最大位移相关结果
- `rl_eval.py`：训练后策略或网格方案评估
- `test.py`：根据已有 `cell_mesh_density.json` 重新生成网格、运行 FEA 并输出评价结果

## 3. 训练运行方式

在解压后的目录中打开 PowerShell 或命令行，运行：

```powershell
abaqus cae noGUI=rl_main.py
```

如需指定参数，可参考脚本中的 argparse 参数，例如最大 episode 数、CPU 数、最大单元数、全局网格尺寸和 checkpoint 目录等。

示例：

```powershell
abaqus cae noGUI=rl_main.py -- --max-episodes 50 --cpus 4 --max-elements 60000
```

训练过程中脚本会自动完成以下流程：

1. 读取 `DEMO.cae` 作为初始模型；
2. 计算或读取高精度 baseline 指标；
3. DQN 智能体对局部 cell 选择 refine、coarsen 或 no-op；
4. 重新生成网格并提交 Abaqus 分析；
5. 从 ODB 提取 ALLSE、单元数等反馈；
6. 计算 reward，写入经验回放并更新 Q 网络；
7. 保存 checkpoint、reward history 和每轮训练结果。

## 4. 测试与验证说明

本项目聚焦训练阶段，因此测试没有拆成三个完全独立的大型工程算例，而是集成在训练与评估脚本中。对应课程要求中的“至少 3 个测试”，本代码包包含以下三类训练阶段检查：

| 测试内容 | 对应脚本/输出 | 目的 |
|---|---|---|
| 初始网格基准检查 | `rl_main.py`、`abaqus_env.py` | 计算初始模型和 baseline 的 ALLSE、单元数等指标，作为训练前对照 |
| 训练过程改善检查 | `rl_main.py`、`dqn_agent.py` | 记录每个 episode 的 reward、动作和有限元反馈，检查训练过程中 ALLSE 是否向 baseline 靠近 |
| 训练后结果评估 | `test.py`、`rl_eval.py`、`extract_results.py` | 对训练得到的网格密度方案重新生成网格并运行 FEA，输出单元数、ALLSE、资源占用和位移等评价指标 |

其中，失败或敏感性分析可从以下现象中提取：

- 单元数惩罚过强时，智能体可能倾向于少加密，导致 ALLSE 改善不足；
- 探索不足时，智能体可能长期选择 no-op 或固定少数 cell；
- Abaqus 求解失败、网格生成失败或超过最大单元数时，环境会给出异常惩罚或终止当前 episode；
- 奖励函数主要依赖 ALLSE，因此对局部应力峰值、疲劳或裂纹扩展等目标的代表性有限。

## 5. 训练后单个网格方案评估

如果已经有训练输出的 `cell_mesh_density.json`，可用 `test.py` 对该网格方案重新运行分析：

```powershell
python test.py --mesh_density_file simulations/run_021/cell_mesh_density.json --base-cae DEMO.cae --output-dir test_output --job-name test_job
```

该流程会依次执行：读取网格密度、生成 CAE 网格、运行 Abaqus FEA、提取 ODB 结果、保存评价 JSON。

## 6. 报告中可引用的代码对应关系

- 理论与算法：`dqn_agent.py`、`abaqus_env.py`
- 有限元反馈闭环：`mesh_generation.py`、`run_fea_analysis.py`、`extract_results.py`
- 训练过程：`rl_main.py`
- 结果验证：`test.py`、`rl_eval.py`
- 位移指标：`get_max_displacement.py`、`run_max_displacement.py`

## 7. 注意事项

- Abaqus 版本不同可能会触发 CAE 文件升级，建议保留原始 `DEMO.cae` 备份。
- 真实 Abaqus 训练耗时较长，完整 episode 运行可能产生大量 `.odb`、`.inp`、`.log` 文件。
- 若只需要检查代码流程，可先减少 `--max-episodes`、`--max-elements` 或使用较少 CPU 进行小规模试跑。
- 本代码包中的训练结果应与报告中的表格、曲线和日志配合使用，不建议只提交代码而不解释测试逻辑。
