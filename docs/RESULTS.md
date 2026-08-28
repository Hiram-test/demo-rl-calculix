# Results (campaign execution of EXPERIMENT_PLAN.md)

本文只报告本仓库本机战役写入 `results/campaign/` 的 CalculiX 记录。
未跑完的格子留空。假设判定只允许：成立 / 不成立 / 证据不足。
不把试点登记数字写成新结论。

## 假设判定

- **H1**（少求解次数处 VLA 优于 Dörfler，加速比 ≥ 1.5×）：**不成立**
- **H2**（预算占用 ∈ [90%, 105%]）：**不成立**
- **H3**（免训练达到学习方法同量级）：**证据不足**
- **H4**（交叉点 k*，Dörfler 未封顶）：`{'bearing_block': 5, 'deck_panel': 5}`
- H4 预算内交叉（Dörfler N ≤ 1.05×试点档时）：`{'bearing_block': None, 'deck_panel': 5}`

## A2′ 误差 @ k 次全局求解（canonical，试点预算档）

带 `*` 的 Dörfler 轮次其 N 超过试点预算 105%（S2 经典循环封顶在最大档，
资源侧见 N/B 列与 A3）。监督/RL 第 2 次求解后交付并保持。

| family | k | Dörfler | Dörfler N/B | VLA (scripted) | local_prediction | supervised | RL (3种子中位) |
|---|---:|---:|---:|---:|---:|---:|---:|
| bearing_block | 1 | 0.4766 | 0.08 | 0.4766 | 0.4766 | 0.4766 | 0.4766 |
| bearing_block | 2 | 0.3762 | 0.13 | 0.2134 | 0.2006 | 0.2237 | 0.2141 |
| bearing_block | 3 | 0.2918 | 0.25 | 0.2019 | 0.1811 | 0.2237 | 0.2141 |
| bearing_block | 4 | 0.2121 | 0.53 | 0.2019 | — | 0.2237 | 0.2141 |
| bearing_block | 5 | 0.1459* | 1.17 | 0.2019 | — | 0.2237 | 0.2141 |
| bearing_block | 6 | 0.0761* | 2.76 | 0.2019 | — | 0.2237 | 0.2141 |
| deck_panel | 1 | 0.7085 | 0.10 | 0.7085 | 0.7085 | 0.7085 | 0.7085 |
| deck_panel | 2 | 0.6053 | 0.14 | 0.3610 | 0.3515 | 0.3793 | 0.3877 |
| deck_panel | 3 | 0.4986 | 0.23 | 0.3610 | 0.2830 | 0.3793 | 0.3877 |
| deck_panel | 4 | 0.3965 | 0.45 | 0.3549 | — | 0.3793 | 0.3877 |
| deck_panel | 5 | 0.2879 | 0.89 | 0.3549 | — | 0.3793 | 0.3877 |
| deck_panel | 6 | 0.1803* | 1.90 | 0.3549 | — | 0.3793 | 0.3877 |

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

## 测试集中位数 [IQR]（计划 §4）

```json
{
 "bearing_block": {
  "2": {
   "dorfler_median": 0.3709452798684745,
   "dorfler_iqr": [
    0.35947346841463174,
    0.38618709123868095
   ],
   "vla_median": 0.25561166751836906,
   "vla_iqr": [
    0.2297864800762033,
    0.2827571598802641
   ],
   "n_dorfler": 8,
   "n_vla": 8
  },
  "3": {
   "dorfler_median": 0.29167019447277975,
   "dorfler_iqr": [
    0.28032249908387247,
    0.3060743143158321
   ],
   "vla_median": 0.21318903371437367,
   "vla_iqr": [
    0.2112231895643017,
    0.22994920937233637
   ],
   "n_dorfler": 8,
   "n_vla": 8
  },
  "4": {
   "dorfler_median": 0.21830823110470726,
   "dorfler_iqr": [
    0.2041865871126107,
    0.22594024683573155
   ],
   "vla_median": 0.2129989948842165,
   "vla_iqr": [
    0.20909350494318737,
    0.22993963543847387
   ],
   "n_dorfler": 8,
   "n_vla": 8
  }
 },
 "deck_panel": {
  "2": {
   "dorfler_median": 0.608637436282876,
   "dorfler_iqr": [
    0.6018737752752925,
    0.6111261925106566
   ],
   "vla_median": 0.3506828689128814,
   "vla_iqr": [
    0.34110183538499733,
    0.3645411072670693
   ],
   "n_dorfler": 8,
   "n_vla": 8
  },
  "3": {
   "dorfler_median": 0.5058733994180606,
   "dorfler_iqr": [
    0.5021610376045645,
    0.5115018524861425
   ],
   "vla_median": 0.3504029443633892,
   "vla_iqr": [
    0.3411214229704252,
    0.35319965780301593
   ],
   "n_dorfler": 8,
   "n_vla": 8
  },
  "4": {
   "dorfler_median": 0.4033446157029582,
   "dorfler_iqr": [
    0.39650541204109213,
    0.409813598867791
   ],
   "vla_median": 0.3498372301205097,
   "vla_iqr": [
    0.3411214229704252,
    0.35306473507809927
   ],
   "n_dorfler": 8,
   "n_vla": 8
  }
 }
}
```

## A4 训练成本（离线求解，不进部署 k 轴）

监督行的 train_solves 为专家库制造（探针+Dörfler 到帽）实际发生的 CalculiX 求解数，
计自各 expert 运行目录；经典方法与 VLA 无离线成本。

```json
[
 {
  "family": "bearing_block",
  "kind": "supervised_experts",
  "n_experts": 24,
  "episodes": null,
  "train_solves": 149
 },
 {
  "family": "bearing_block",
  "kind": "rl_s0",
  "n_experts": null,
  "episodes": 120,
  "train_solves": 294
 },
 {
  "family": "bearing_block",
  "kind": "rl_s1",
  "n_experts": null,
  "episodes": 120,
  "train_solves": 310
 },
 {
  "family": "bearing_block",
  "kind": "rl_s2",
  "n_experts": null,
  "episodes": 120,
  "train_solves": 275
 },
 {
  "family": "deck_panel",
  "kind": "supervised_experts",
  "n_experts": 24,
  "episodes": null,
  "train_solves": 167
 },
 {
  "family": "deck_panel",
  "kind": "rl_s0",
  "n_experts": null,
  "episodes": 120,
  "train_solves": 231
 },
 {
  "family": "deck_panel",
  "kind": "rl_s1",
  "n_experts": null,
  "episodes": 120,
  "train_solves": 296
 },
 {
  "family": "deck_panel",
  "kind": "rl_s2",
  "n_experts": null,
  "episodes": 120,
  "train_solves": 250
 },
 {
  "family": "lbracket",
  "kind": "supervised_experts",
  "n_experts": 24,
  "episodes": null,
  "train_solves": null
 },
 {
  "family": "lbracket",
  "kind": "rl_s0",
  "n_experts": null,
  "episodes": 300,
  "train_solves": 1565
 },
 {
  "family": "lbracket",
  "kind": "rl_s1",
  "n_experts": null,
  "episodes": 80,
  "train_solves": 421
 },
 {
  "family": "lbracket",
  "kind": "rl_s2",
  "n_experts": null,
  "episodes": 80,
  "train_solves": 488
 },
 {
  "family": "plate_holes",
  "kind": "supervised_experts",
  "n_experts": 24,
  "episodes": null,
  "train_solves": null
 },
 {
  "family": "plate_holes",
  "kind": "rl_s0",
  "n_experts": null,
  "episodes": 80,
  "train_solves": 471
 },
 {
  "family": "plate_holes",
  "kind": "rl_s1",
  "n_experts": null,
  "episodes": 80,
  "train_solves": 456
 },
 {
  "family": "plate_holes",
  "kind": "rl_s2",
  "n_experts": null,
  "episodes": 80,
  "train_solves": 481
 }
]
```

## 局部预测轮数诊断（canonical，试点档；审核补充）

计划 §3.3 锁定每档 probe+2 轮=3 次求解；此诊断放开到 7 次以核查该限制是否压误差。

| family | solve | N | N/B | e_E |
|---|---:|---:|---:|---:|
| bearing_block | 1 | 639 | 0.08 | 0.4766 |
| bearing_block | 2 | 5478 | 0.68 | 0.2006 |
| bearing_block | 3 | 5961 | 0.75 | 0.1811 |
| bearing_block | 4 | 6186 | 0.77 | 0.1768 |
| bearing_block | 5 | 6243 | 0.78 | 0.1743 |
| bearing_block | 6 | 6276 | 0.78 | 0.1754 |
| bearing_block | 7 | 6294 | 0.79 | 0.1750 |
| deck_panel | 1 | 2073 | 0.10 | 0.7085 |
| deck_panel | 2 | 12411 | 0.62 | 0.3515 |
| deck_panel | 3 | 17552 | 0.88 | 0.2830 |
| deck_panel | 4 | 18092 | 0.90 | 0.2736 |
| deck_panel | 5 | 18645 | 0.93 | 0.2677 |
| deck_panel | 6 | 18778 | 0.94 | 0.2670 |
| deck_panel | 7 | 18846 | 0.94 | 0.2685 |

- bearing_block：3 次求解 e=0.1811，7 次内最优 e=0.1743（第 5 次），额外 4 次求解的相对改善 3.8%；最优后出现回弹：是。

- deck_panel：3 次求解 e=0.2830，7 次内最优 e=0.2670（第 6 次），额外 4 次求解的相对改善 5.6%；最优后出现回弹：是。

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
  },
  {
   "name": "AB9 fixed-q",
   "e_energy": 0.2008390112292959,
   "n_eq": 7620,
   "solves": 3
  },
  {
   "name": "AB10 no-drift",
   "e_energy": 0.20192001625415698,
   "n_eq": 7425,
   "solves": 3
  },
  {
   "name": "AB10 safety 0.92",
   "e_energy": 0.20192001625415698,
   "n_eq": 7425,
   "solves": 3
  },
  {
   "name": "AB10 safety 0.97",
   "e_energy": 0.20192001625415698,
   "n_eq": 7425,
   "solves": 3
  },
  {
   "name": "AB11 no-inplace",
   "e_energy": 0.19667413511883547,
   "n_eq": 7917,
   "solves": 4
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
  },
  {
   "name": "AB9 fixed-q",
   "e_energy": 0.3569896648731121,
   "n_eq": 16720,
   "solves": 4
  },
  {
   "name": "AB10 no-drift",
   "e_energy": 0.35488001458683066,
   "n_eq": 16647,
   "solves": 4
  },
  {
   "name": "AB10 safety 0.92",
   "e_energy": 0.3494593632292738,
   "n_eq": 17681,
   "solves": 4
  },
  {
   "name": "AB10 safety 0.97",
   "e_energy": 0.34092319341433414,
   "n_eq": 18572,
   "solves": 4
  },
  {
   "name": "AB11 no-inplace",
   "e_energy": 0.35488001458683066,
   "n_eq": 16647,
   "solves": 4
  }
 ]
}
```

## 学习方法测试集中位数（计划 §1.3；不升格为 H3）

```json
{
 "bearing_block": {
  "supervised": {
   "n": 8,
   "e_median": 0.22356099772970328,
   "e_iqr": [
    0.21966771404725438,
    0.22476420924589302
   ],
   "n_eq_median": 4135.5,
   "frac_median": 0.5169375,
   "n_over_cap": 0
  },
  "rl_dqn_s0": {
   "n": 8,
   "e_median": 0.20597351319692214,
   "e_iqr": [
    0.20057974676893914,
    0.22304010485527065
   ],
   "n_eq_median": 8032.5,
   "frac_median": 1.0040624999999999,
   "n_over_cap": 4
  },
  "rl_dqn_s1": {
   "n": 8,
   "e_median": 0.20597351319692214,
   "e_iqr": [
    0.20057974676893914,
    0.22304010485527065
   ],
   "n_eq_median": 8032.5,
   "frac_median": 1.0040624999999999,
   "n_over_cap": 4
  },
  "rl_dqn_s2": {
   "n": 8,
   "e_median": 0.20597351319692214,
   "e_iqr": [
    0.20057974676893914,
    0.22647383778861307
   ],
   "n_eq_median": 8032.5,
   "frac_median": 1.0040624999999999,
   "n_over_cap": 4
  }
 },
 "deck_panel": {
  "supervised": {
   "n": 8,
   "e_median": 0.3772728810045015,
   "e_iqr": [
    0.37002138683668395,
    0.3796621230469551
   ],
   "n_eq_median": 8767.0,
   "frac_median": 0.43835,
   "n_over_cap": 0
  },
  "rl_dqn_s0": {
   "n": 8,
   "e_median": 0.3671579460240108,
   "e_iqr": [
    0.36571317694154276,
    0.37613071636923945
   ],
   "n_eq_median": 20813.0,
   "frac_median": 1.04065,
   "n_over_cap": 6
  },
  "rl_dqn_s1": {
   "n": 8,
   "e_median": 0.39517048249010844,
   "e_iqr": [
    0.39186979254758025,
    0.39873695704714485
   ],
   "n_eq_median": 19210.5,
   "frac_median": 0.9605250000000001,
   "n_over_cap": 0
  },
  "rl_dqn_s2": {
   "n": 8,
   "e_median": 0.39517048249010844,
   "e_iqr": [
    0.39186979254758025,
    0.4004949473327017
   ],
   "n_eq_median": 19452.5,
   "frac_median": 0.9726250000000001,
   "n_over_cap": 2
  }
 }
}
```

## 学习方法部署账本（3D canonical；不升格为 H3）

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
  "rl_episodes_s1": 80,
  "rl_episodes_s2": 80,
  "rl_seeds": 3
 },
 "plate_holes": {
  "supervised_experts": 24,
  "rl_episodes_s0": 80,
  "rl_episodes_s1": 80,
  "rl_episodes_s2": 80,
  "rl_seeds": 3
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

## 审核补充：对照方法缺点实证（全部为真实 CalculiX 求解）

### E1 局部预测·朴素部署振荡（§3.3.1 失稳模式复现）

同一逐单元预测循环，改用文献朴素配方：指数 p=1、尺寸比界 [1/6, 3.0]
（v2 锁定值为 p=(d+2)/2、尺寸比界 [1/6, 1.8]）。

| family | solve | N | N/B | e_E |
|---|---:|---:|---:|---:|
| bearing_block | 1 | 639 | 0.08 | 0.4766 |
| bearing_block | 2 | 6945 | 0.87 | 0.1800 |
| bearing_block | 3 | 8640 | 1.08 | 0.1673 |
| bearing_block | 4 | 7971 | 1.00 | 0.1752 |
| bearing_block | 5 | 8382 | 1.05 | 0.1771 |
| bearing_block | 6 | 7449 | 0.93 | 0.1911 |
| bearing_block | 7 | 8259 | 1.03 | 0.1882 |
| deck_panel | 1 | 2073 | 0.10 | 0.7085 |
| deck_panel | 2 | 14622 | 0.73 | 0.3295 |
| deck_panel | 3 | 21035 | 1.05 | 0.2960 |
| deck_panel | 4 | 21577 | 1.08 | 0.2758 |
| deck_panel | 5 | 24549 | 1.23 | 0.2938 |
| deck_panel | 6 | 20343 | 1.02 | 0.3089 |
| deck_panel | 7 | 24237 | 1.21 | 0.3358 |

- bearing_block：最优 e=0.1673（第 3 次），末轮 e=0.1882，最优后恶化 12.5%；回弹：是。对照：v2 修正版（上节）最优后恶化 0.5%（亦有回弹）。

- deck_panel：最优 e=0.2758（第 4 次），末轮 e=0.3358，最优后恶化 21.8%；回弹：是。对照：v2 修正版（上节）最优后恶化 0.5%（亦有回弹）。

### E2 无认证语义的预算偏差（canonical+测试 8，按各自契约空间）

LP/监督承诺单元数（frac=单元数/单元预算）；VLA 承诺方程帽（frac=N/预算）。

| family | method | n | median | min | max | in [0.90,1.05] |
|---|---|---:|---:|---:|---:|---:|
| bearing_block | local_prediction | 27 | 0.80 | 0.76 | 0.89 | 0/27 |
| bearing_block | supervised | 9 | 0.52 | 0.50 | 0.54 | 0/9 |
| bearing_block | vla | 9 | 0.84 | 0.63 | 0.99 | 4/9 |
| deck_panel | local_prediction | 27 | 0.85 | 0.79 | 0.93 | 9/27 |
| deck_panel | supervised | 9 | 0.40 | 0.38 | 0.41 | 0/9 |
| deck_panel | vla | 9 | 0.91 | 0.81 | 0.99 | 5/9 |

### E3 监督·分布偏移（OOD 实例：全部参数在训练采样器支撑集外）

同场部署：监督（2 次求解）、LP 短跑（3 次）、VLA scripted。
免训练方法对新实例本来就是零样本；差距变化隔离偏移效应。

| family | key | supervised e (frac) | lp3 e (frac) | VLA e (frac) |
|---|---|---:|---:|---:|
| bearing_block | ood_9500 | 0.1970 (0.51) | 0.1539 (0.81) | 0.1647 (0.84) |
| bearing_block | ood_9501 | 0.1979 (0.49) | 0.1542 (0.81) | 0.1653 (0.83) |
| bearing_block | ood_9502 | 0.1990 (0.48) | 0.1498 (0.81) | 0.1667 (0.82) |
| bearing_block | ood_9503 | 0.1963 (0.49) | 0.1546 (0.80) | 0.1665 (0.83) |
| deck_panel | ood_9500 | 0.3446 (0.40) | 0.2414 (0.85) | 0.3150 (0.96) |
| deck_panel | ood_9501 | 0.3437 (0.39) | 0.2370 (0.86) | 0.3201 (0.99) |
| deck_panel | ood_9502 | 0.2961 (0.40) | 0.2000 (0.87) | 0.2681 (0.95) |
| deck_panel | ood_9503 | 0.2866 (0.39) | 0.1961 (0.86) | 0.2695 (0.91) |

- bearing_block：监督−VLA 中位差距，分布内 0.0058 → OOD 0.0323。

- deck_panel：监督−VLA 中位差距，分布内 0.0245 → OOD 0.0258。


## 诚实边界

- Dörfler 的渐近最优性不在本文争夺范围；k* 交叉若出现必须画出。
- A2′ 的 Dörfler 列不按试点档封顶（S2 给经典循环的帽是最大档）；超 105% 的轮次带 `*`，预算内交叉在假设判定单列。A2″ 的第 4/6 轮目标误差同样取自该未封顶序列。
- 局部预测是逐单元一步预测，不是分区方法；其预算偏差如实列入。
- VLA 第 2 次求解的初始尺寸复用同一逐单元等分布预测（按区几何平均，§3.6），故 k=2 处两者相近是结构性的；其后 VLA 走实测指数/漂移反馈/硬帽投影/就地认证（AB5/AB6/AB9–AB11 量化各自贡献），局部预测走再等分布（轮数诊断见上）。
- 监督专家库由训练实例上的 Dörfler-到帽循环蒸馏（离线求解已计入 A4）；其部署无预算帽语义，canonical 交付停在 44–52% 档位。
- LLM 头失败回退 Scripted 时计入回退率，不把 Scripted 数字标成 LLM。
- 训练期求解（监督专家库、RL episode）单列，不混进部署 k 轴。
- 论文主文只报 3D。S5 3D 监督 24 专家；S6 3D RL 120 回合 × 3 种子。H3 仍为证据不足。
- 图政策（§8）：局部预测按预算档分列，不跨档拼线；e_E–N 图标注加密原子，不是单一 Pareto；k* 画在误差@k 主图上。
- 学习方法按 §1.3 在测试集 8 实例上汇总；该表不升格为 H3。

## S3 验收门

| gate | pass |
|---|---|
| G1 | PASS |
| G2 | PASS |
| G3 | PASS |
| G4 | PASS |
| G5 | PASS |
| G7 | PASS |

