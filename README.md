# Vision-Region AMR: 视觉分区驱动的少重网格自适应有限元

研究框架：比较**视觉分区多智能体短循环（VLA）**与四个对比算法（Dörfler 逐单元循环、局部预测一步多级、区域图 DQN 强化学习、监督尺寸场回归）在"误差 × 自由度 × 全局求解次数 × 预算合规 × 训练成本"五个轴上的表现。

核心约定：**所有重网格都经 Gmsh**。任何方法的决策统一表达为尺寸场（逐单元目标尺寸图或区域尺寸），由 Gmsh 重新生成网格，CalculiX 求解。仓库中不存在手写网格操作。

## 文档

- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) — 详细实验方案（问题集、六方法部署契约、比较协议、消融、验收门、步卡）
- [`docs/BASELINE_AUDIT.md`](docs/BASELINE_AUDIT.md) — 旧实现审计：四个对比算法历史部署的问题清单与修复对照
- [`docs/evidence/`](docs/evidence) — 试点运行的证据图（分区、网格、误差曲线）

## 安装

```bash
sudo apt-get install calculix-ccx libglu1-mesa
pip install -r requirements.txt
```

## 快速验证

```bash
# 门 G1：奇异问题上正确部署的 Dörfler 必须优于均匀加密
python3 scripts/run_smoke.py

# 单元测试（不需要 CalculiX）
python3 -m pytest tests/ -q

# 六方法基准（单实例演示规模，含 RL 训练与监督专家库生成）
python3 scripts/run_benchmark.py --problem lbracket --out results/bench_lbracket
python3 scripts/run_benchmark.py --problem plate_holes --out results/bench_plate

# 证据图
python3 scripts/make_figures.py
```

## 包结构

```
visionamr/
  geometry.py        基准问题族（L 形支架/带孔板，参数化采样器）
  mesher.py          Gmsh 网格器（唯一的网格来源）
  sizefield.py       尺寸场：区域尺寸 / 节点尺寸图 + Lipschitz 梯度限制
  calculix.py        CalculiX 卡片、运行器、FRD 解析
  fem_post.py        位移重构应力/应变能/von Mises
  indicators.py      ZZ 1987 恢复型误差指示子
  marking.py         Dörfler bulk 标记 / 极大值标记
  experiment.py      求解记账（每次 ccx 调用都有记录）、分级参考解
  baselines/
    uniform.py           均匀梯子
    dorfler.py           Dörfler 逐单元循环（ZZ + bulk 标记 + Gmsh 重网格）
    local_prediction.py  逐单元一步多级尺寸预测（ZZ 1987 / Li–Bettess 1995）
    supervised.py        专家网格自造 + 节点尺寸场 MLP + 预算标量部署
    rl_dqn.py            区域图 Double DQN + GCN（每步真实求解）
  vla/
    partition.py     视觉分区：LLM 头（多模态接口）+ 脚本头（热点聚类）
    regions.py       区域图（邻接、特征聚合）
    agents.py        子智能体一轮通信（邻域耦合 + 父级预算压力）+ 区域编辑
    pso.py           (s, κ) 两坐标 PSO 校准（幂律代理 + 闭式种子）
    pipeline.py      三次求解短循环：probe → regional → certified
```

## 归档

`archived-src/` 保留课程作业时期的 Abaqus/CalculiX RL 代码（V1/V2），仅作历史对照，不参与新实验。原始压缩包见 [`v1.0.0` release](../../releases/tag/v1.0.0)。
