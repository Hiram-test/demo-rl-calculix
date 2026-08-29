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

## 视觉模型（VLA 视觉头）配置

VLA 的视觉头 `visionamr/vla/partition.py::LLMVisionPartitioner` 把渲染出的正交三视图（PNG）发给一个**多模态大模型**，由它像工程师读图一样圈出命名区域并回严格 JSON。默认后端是**火山引擎方舟（Volcengine Ark）的 `doubao-seed-evolving`**，走 OpenAI 兼容的 `/chat/completions`，因此代码里没有任何厂商专用分支。

- 默认端点（`DEFAULT_VLM_API_BASE`）：`https://ark.cn-beijing.volces.com/api/v3`
- 默认模型（`DEFAULT_VLM_MODEL`）：`doubao-seed-evolving`（多模态，已验证可直接吃 `image_url` 的 base64 三视图并回 JSON）
- 端点解析统一在 `resolve_vlm_endpoint()`，按以下优先级取凭据（仓库**不含任何密钥**，公开仓库禁止硬编码 key）：

**方式 1：环境变量给 key（最简单）**

```bash
# Linux / macOS
export ARK_API_KEY="你的方舟 API Key"
# Windows PowerShell
$env:ARK_API_KEY = "你的方舟 API Key"
```

**方式 2：指向本地方舟配置 JSON（可复用控制台导出的配置，文件留在本机、不要提交）**

可直接复制模板 [`visionamr/vla/ark_config.example.json`](visionamr/vla/ark_config.example.json) 后填入 key：

```json
{
  "base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "api_key": "你的方舟 API Key",
  "model": "doubao-seed-evolving"
}
```

```bash
export VISIONAMR_ARK_CONFIG="/path/to/evolving_api.json"   # PowerShell 用 $env:
```

读取兼容 UTF-8 BOM；文件缺失或损坏时按"未配置"处理，不会报错中断。

**方式 3：换任意 OpenAI 兼容的多模态模型**

```bash
export VLM_API_BASE="https://你的端点/v1"
export VLM_API_KEY="你的 key"
export VLM_MODEL="你的视觉模型名"
```

历史的 `XAI_API_KEY`（xAI Grok）与 `OPENAI_API_KEY` 仍然兼容；也可以在代码里直接构造 `LLMVisionPartitioner(api_base=..., api_key=..., model=...)` 显式指定。

**无 key 时的行为**：`resolve_vlm_endpoint()` 返回 `api_key=None`，视觉头记录 `no_api_key` 并**自动回退到 `ScriptedVisionPartitioner`**（按荷载/支承/角点/孔等结构锚点画图），所以离线环境、CI 和单元测试无需任何密钥即可跑通；`results/**/llm_dump/` 里会留下真实视觉头的图纸、原始回复和解析出的种子，便于审计。方舟为北京节点，请求会自动绕过本机代理直连。

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
    pso.py           等级上的一次闭式微调（`calibrate_grades`，不搜索）
    pipeline.py      视觉（等级）→ 工具（微调/出网/求解）→ 再视觉
```

## 归档

`archived-src/` 保留课程作业时期的 Abaqus/CalculiX RL 代码（V1/V2），仅作历史对照，不参与新实验。原始压缩包见 [`v1.0.0` release](../../releases/tag/v1.0.0)。
