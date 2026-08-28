# Results (campaign execution of EXPERIMENT_PLAN.md)

本文只报告本仓库本机战役写入 `results/campaign/` 的 CalculiX 记录。
未跑完的格子留空。假设判定只允许：成立 / 不成立 / 证据不足。
不把试点登记数字写成新结论。

## 假设判定

- **H1**（少求解次数处 VLA 优于 Dörfler，加速比 ≥ 1.5×）：**不成立**
- **H2**（预算占用 ∈ [90%, 105%]）：**不成立**
- **H3**（免训练达到学习方法同量级）：**证据不足**
- **H4**（交叉点 k*）：`{'bearing_block': 5, 'deck_panel': 5}`

## A2′ 误差 @ k 次全局求解（canonical，试点预算档）

| family | k | Dörfler | VLA (scripted) | local_prediction |
|---|---:|---:|---:|---:|
| bearing_block | 1 | 0.4766 | 0.4766 | 0.4766 |
| bearing_block | 2 | 0.3762 | 0.2134 | 0.2006 |
| bearing_block | 3 | 0.2918 | 0.2019 | 0.1811 |
| bearing_block | 4 | 0.2121 | 0.2019 | — |
| bearing_block | 5 | 0.1459 | 0.2019 | — |
| bearing_block | 6 | 0.0761 | 0.2019 | — |
| deck_panel | 1 | 0.7085 | 0.7085 | 0.7085 |
| deck_panel | 2 | 0.6053 | 0.3610 | 0.3515 |
| deck_panel | 3 | 0.4986 | 0.3610 | 0.2830 |
| deck_panel | 4 | 0.3965 | 0.3549 | — |
| deck_panel | 5 | 0.2879 | 0.3549 | — |
| deck_panel | 6 | 0.1803 | 0.3549 | — |

## A2″ 加速比（到达 Dörfler 第 4/6 轮误差）

```json
{
 "bearing_block": {
  "dorfler_k4": {
   "target_e": 0.212075641307299,
   "vla_solves": 3,
   "speedup": 1.3333333333333333
  },
  "dorfler_k6": {
   "target_e": 0.07613859601027445,
   "vla_solves": null,
   "speedup": null
  }
 },
 "deck_panel": {
  "dorfler_k4": {
   "target_e": 0.39653825850383745,
   "vla_solves": 2,
   "speedup": 2.0
  },
  "dorfler_k6": {
   "target_e": 0.18031905430507825,
   "vla_solves": null,
   "speedup": null
  }
 }
}
```

## 测试集 Wilcoxon（H1）

```json
{
 "bearing_block": {
  "2": {
   "n": 8,
   "p": 0.00390625,
   "median_diff": -0.1142571737534526,
   "judgment": "\u6210\u7acb"
  },
  "3": {
   "n": 8,
   "p": 0.00390625,
   "median_diff": -0.06439218236917074,
   "judgment": "\u6210\u7acb"
  },
  "4": {
   "n": 8,
   "p": 0.6796875,
   "median_diff": 0.004243954358058785,
   "judgment": "\u4e0d\u6210\u7acb"
  }
 },
 "deck_panel": {
  "2": {
   "n": 8,
   "p": 0.00390625,
   "median_diff": -0.25774678926183725,
   "judgment": "\u6210\u7acb"
  },
  "3": {
   "n": 8,
   "p": 0.00390625,
   "median_diff": -0.15655207663817278,
   "judgment": "\u6210\u7acb"
  },
  "4": {
   "n": 8,
   "p": 0.00390625,
   "median_diff": -0.055071837310898236,
   "judgment": "\u6210\u7acb"
  }
 }
}
```

## LLM 视觉头回退率

```json
{
 "n": 18,
 "n_fallback": 0,
 "rate": 0.0,
 "details": [
  {
   "family": "bearing_block",
   "key": "canonical",
   "source": "llm_cache"
  },
  {
   "family": "bearing_block",
   "key": "test_9000",
   "source": "llm_cache"
  },
  {
   "family": "bearing_block",
   "key": "test_9001",
   "source": "llm_cache"
  },
  {
   "family": "bearing_block",
   "key": "test_9002",
   "source": "llm_cache"
  },
  {
   "family": "bearing_block",
   "key": "test_9003",
   "source": "llm_cache"
  },
  {
   "family": "bearing_block",
   "key": "test_9004",
   "source": "llm_cache"
  },
  {
   "family": "bearing_block",
   "key": "test_9005",
   "source": "llm_cache"
  },
  {
   "family": "bearing_block",
   "key": "test_9006",
   "source": "llm_cache"
  },
  {
   "family": "bearing_block",
   "key": "test_9007",
   "source": "llm_cache"
  },
  {
   "family": "deck_panel",
   "key": "canonical",
   "source": "llm_cache"
  },
  {
   "family": "deck_panel",
   "key": "test_9000",
   "source": "llm_cache"
  },
  {
   "family": "deck_panel",
   "key": "test_9001",
   "source": "llm_cache"
  },
  {
   "family": "deck_panel",
   "key": "test_9002",
   "source": "llm_cache"
  },
  {
   "family": "deck_panel",
   "key": "test_9003",
   "source": "llm_cache"
  },
  {
   "family": "deck_panel",
   "key": "test_9004",
   "source": "llm_cache"
  },
  {
   "family": "deck_panel",
   "key": "test_9005",
   "source": "llm_cache"
  },
  {
   "family": "deck_panel",
   "key": "test_9006",
   "source": "llm_cache"
  },
  {
   "family": "deck_panel",
   "key": "test_9007",
   "source": "llm_cache"
  }
 ]
}
```

## 消融（canonical × 试点档）

```json
{
 "bearing_block": [
  {
   "name": "full",
   "e_energy": 0.20192001625415698,
   "n_eq": 7425,
   "solves": 3
  },
  {
   "name": "AB1 random",
   "e_energy": 0.21371352021926265,
   "n_eq": 6585,
   "solves": 4
  },
  {
   "name": "AB2 box",
   "e_energy": 0.2049877497208842,
   "n_eq": 7458,
   "solves": 3
  },
  {
   "name": "AB3 no-anchor",
   "e_energy": 0.23884983129371573,
   "n_eq": 6048,
   "solves": 2
  },
  {
   "name": "AB4 no-split",
   "e_energy": 0.20192001625415698,
   "n_eq": 7425,
   "solves": 3
  },
  {
   "name": "AB5 no-comm",
   "e_energy": 0.20610937124732617,
   "n_eq": 7251,
   "solves": 3
  },
  {
   "name": "AB6 no-PSO",
   "e_energy": 0.19908872887406862,
   "n_eq": 7176,
   "solves": 2
  },
  {
   "name": "AB7 k=3",
   "e_energy": 0.21258234313602672,
   "n_eq": 6699,
   "solves": 3
  },
  {
   "name": "AB7 k=4",
   "e_energy": 0.20192001625415698,
   "n_eq": 7425,
   "solves": 3
  },
  {
   "name": "AB7 k=5",
   "e_energy": 0.20192001625415698,
   "n_eq": 7425,
   "solves": 3
  },
  {
   "name": "AB7 k=6",
   "e_energy": 0.20192001625415698,
   "n_eq": 7425,
   "solves": 3
  },
  {
   "name": "AB8 s-only",
   "e_energy": 0.20455317809302867,
   "n_eq": 6552,
   "solves": 3
  },
  {
   "name": "AB8 nelder",
   "e_energy": 0.2036369199782863,
   "n_eq": 6930,
   "solves": 2
  }
 ],
 "deck_panel": [
  {
   "name": "full",
   "e_energy": 0.35488001458683066,
   "n_eq": 16647,
   "solves": 4
  },
  {
   "name": "AB1 random",
   "e_energy": 0.34850544634884173,
   "n_eq": 17436,
   "solves": 4
  },
  {
   "name": "AB2 box",
   "e_energy": 0.35131807041365243,
   "n_eq": 16927,
   "solves": 4
  },
  {
   "name": "AB3 no-anchor",
   "e_energy": 0.35562473853582016,
   "n_eq": 19549,
   "solves": 2
  },
  {
   "name": "AB4 no-split",
   "e_energy": 0.3610466995920565,
   "n_eq": 16134,
   "solves": 2
  },
  {
   "name": "AB5 no-comm",
   "e_energy": 0.3463191131499309,
   "n_eq": 17081,
   "solves": 4
  },
  {
   "name": "AB6 no-PSO",
   "e_energy": 0.3514450167453172,
   "n_eq": 16051,
   "solves": 2
  },
  {
   "name": "AB7 k=3",
   "e_energy": 0.3610466995920565,
   "n_eq": 16134,
   "solves": 2
  },
  {
   "name": "AB7 k=4",
   "e_energy": 0.35488001458683066,
   "n_eq": 16647,
   "solves": 4
  },
  {
   "name": "AB7 k=5",
   "e_energy": 0.34092319341433414,
   "n_eq": 18572,
   "solves": 4
  },
  {
   "name": "AB7 k=6",
   "e_energy": 0.34092319341433414,
   "n_eq": 18572,
   "solves": 4
  },
  {
   "name": "AB8 s-only",
   "e_energy": 0.34645422353965877,
   "n_eq": 16837,
   "solves": 4
  },
  {
   "name": "AB8 nelder",
   "e_energy": 0.3513948363638845,
   "n_eq": 15801,
   "solves": 2
  }
 ]
}
```

## 学习方法部署账本（canonical；不升格为 H3）

实际训练规模（从产物推断；3D 对齐计划 24 专家 / 120×3）：

```json
{
 "bearing_block": {
  "supervised_experts": 24,
  "rl_episodes_s0": 120,
  "rl_episodes_s1": 120,
  "rl_episodes_s2": 120,
  "rl_seeds": 3
 },
 "deck_panel": {
  "supervised_experts": 24,
  "rl_episodes_s0": 120,
  "rl_episodes_s1": 120,
  "rl_episodes_s2": 120,
  "rl_seeds": 3
 },
 "lbracket": {
  "supervised_experts": 24,
  "rl_episodes_s0": 300,
  "rl_seeds": 1
 },
 "plate_holes": {
  "supervised_experts": 24
 }
}
```

| family | method | solves | n_eq | e_energy | budget frac | over cap |
|---|---|---:|---:|---:|---:|:---:|
| bearing_block | supervised | 2 | 4038 | 0.2237 | 0.5048 | no |
| bearing_block | rl_dqn_s0 | 2 | 8235 | 0.2141 | 1.0294 | yes |
| bearing_block | rl_dqn_s1 | 2 | 8235 | 0.2141 | 1.0294 | yes |
| bearing_block | rl_dqn_s2 | 2 | 8235 | 0.2141 | 1.0294 | yes |
| deck_panel | supervised | 2 | 8860 | 0.3793 | 0.4430 | no |
| deck_panel | rl_dqn_s0 | 2 | 19780 | 0.3626 | 0.9890 | no |
| deck_panel | rl_dqn_s1 | 2 | 18819 | 0.3877 | 0.9409 | no |
| deck_panel | rl_dqn_s2 | 2 | 18819 | 0.3877 | 0.9409 | no |
| lbracket | supervised | 2 | 8824 | 0.0524 | 1.1030 | yes |
| lbracket | rl_dqn_s0 | 5 | 6804 | 0.1255 | 0.8505 | no |
| plate_holes | supervised | 2 | 9671 | 0.0199 | 1.2089 | yes |

## 诚实边界

- Dörfler 的渐近最优性不在本文争夺范围；k* 交叉若出现必须画出。
- 局部预测是逐单元一步预测，不是分区方法；其预算偏差如实列入。
- LLM 头失败回退 Scripted 时计入回退率，不把 Scripted 数字标成 LLM。
- 训练期求解（监督专家库、RL episode）单列，不混进部署 k 轴。
- 论文主文只报 3D。S5 3D 监督 24 专家；S6 3D RL 120 回合 × 3 种子。H3 仍为证据不足。

