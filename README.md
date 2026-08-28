# Vision-Region AMR: 视觉分区驱动的少重网格自适应有限元

研究框架：在**三维桥梁构件**（支座板局部承压、梁桥面板轮载）上，比较**视觉播种的类人分区多智能体短循环（VLA）**与四个对比算法（Dörfler 逐单元循环、局部预测一步多级、分区图 DQN 强化学习、监督尺寸场回归）。主目标：**少量求解次数（k=2..6）处击败 Dörfler**（渐近最优性让给它），且预算硬帽认证。

核心约定：
- **所有重网格都经 Gmsh**：任何方法的决策统一表达为尺寸场，由 Gmsh 重新生成网格，CalculiX 求解；仓库无手写网格操作。
- **区域绝不是盒子**：视觉头输出命名种子（结构锚点+场峰值+粗场点），区域是单元邻接图上的多源测地线 Voronoi 分区，形状贴几何生长，全域覆盖；粒度靠残差集中区自动分裂。
- 荷载/约束足迹压印进几何（OCC fragment），任意网格下合力精确。

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
# 门 G1（2D 基板）：正确部署的 Dörfler 必须优于均匀加密
python3 scripts/run_smoke.py

# 单元测试（不需要 CalculiX）
python3 -m pytest tests/ -q

# 三维桥梁构件基准（经典方法 + VLA）
python3 scripts/run_benchmark.py --problem bearing_block --n-eq-budget 8000
python3 scripts/run_benchmark.py --problem deck_panel --n-eq-budget 20000

# 2D 基板全方法基准（含 RL 训练与监督专家库，演示规模）
python3 scripts/run_benchmark.py --problem lbracket --with-learned

# 证据图（分区三视图、认证网格、误差曲线）
python3 scripts/make_figures.py bearing_block deck_panel
```

## 包结构

```
visionamr/
  geometry.py        问题族：3D 支座/桥面板（荷载足迹压印）+ 2D 基板，参数化采样器
  mesher.py          Gmsh 网格器（2D 三角/3D 四面体，唯一网格来源）
  sizefield.py       节点尺寸图 + Lipschitz 梯度限制 + 反距离插值
  calculix.py        CPS3/C3D4 卡片、运行器、FRD 解析
  fem_post.py        2D/3D B 矩阵重构应力/能量（含 3D 补丁测试）
  indicators.py      ZZ 1987 指示子（2D 三中点 / 3D 四点精确积分）
  marking.py         Dörfler bulk 标记 / 极大值标记
  experiment.py      求解记账、奇异线分级参考解
  baselines/
    uniform.py           均匀梯子（每档单元数翻倍，预算帽）
    dorfler.py           Dörfler 逐单元循环
    local_prediction.py  逐单元一步多级尺寸预测（理论指数 (d+2)/2）
    supervised.py        专家网格自造 + 节点尺寸场 MLP + 预算标量
    rl_dqn.py            分区图 Double DQN + GCN（每步真实求解）
  vla/
    partition.py     视觉头：LLM（正交三视图+JSON 种子）/ 脚本（峰值+结构锚点+粗场点）
    regions.py       类人分区：种子生长测地线 Voronoi、特征聚合、残差集中分裂
    agents.py        子智能体一轮通信（份额/邻域/父级预算压力）
    pso.py           决策超预算时的投影（`project_feasible`）；旧 `calibrate_measured` 留作单测
    pipeline.py      自适应短循环：读图 → first → 中间修订 → 最后实测 PSO → certified
```

## 归档

`archived-src/` 保留课程作业时期的 Abaqus/CalculiX RL 代码（V1/V2），仅作历史对照，不参与新实验。原始压缩包见 [`v1.0.0` release](../../releases/tag/v1.0.0)。
