# Bridge Component Mesh Evidence Suite

本套件直接面向普通桥梁工程人员提出的网格问题，不要求用户预先掌握奇异性、J积分、塑性本构或网格收敛理论。

输入问题可以只是：

- “支座反力放在一个节点，网格越细最大应力越高，怎么办？”
- “腹板圆孔边应该全局加密还是局部加密？”
- “裂纹尖端应力一直增大，还要继续加密吗？”
- “横隔板矩形开孔尖角应力不收敛，是否需要圆角？”

## 完整闭环

每个构件场景都真实执行：

1. 参数化建模；
2. 至少三档网格；
3. 二维四节点平面应力有限元求解；
4. 局部峰值、固定物理位置结果、整体位移和应变能提取；
5. `energy_consistency` Skill；
6. 可用时的理论解或守恒关系交叉验证；
7. 物理载荷或几何变体；
8. CalculiX 独立求解器交叉核验；
9. DeepSeek 面向普通用户的证据复核；
10. Markdown、PDF、DOCX 和全部原始数组报告。

## 当前四种桥梁构件场景

| 场景 | 数值问题 | 已落实处理 |
|---|---|---|
| 支座/横梁载荷引入 | 单节点力附近峰值随加密增长 | 同合力、同作用线的有限承压宽度；远场位移、截面应力和能量校核 |
| 腹板圆孔 | 孔边应力对网格敏感 | 孔边 O 形环向-径向分级网格；Kirsch 解校核 |
| 钢板中心裂纹 | 裂纹尖端最大应力不收敛 | 放弃逐点最大应力；由裂纹微增模型计算线弹性 `J=G` 并换算 `K` |
| 横隔板矩形开孔 | 零圆角尖角峰值持续增长 | 固定距离结构应力 + 16 mm 圆角实体化对照 |

## 能量 Skill

旧的 slotted-bracket 在线运行记录中没有 `energy` 或 `strain_energy` Skill 调用。本套件将能量检查变成强制证据：每个场景执行一次，检查 `0.5 u^T K u` 与 `0.5 f^T u` 的一致性，并把末两档应变能变化用于判断整体响应是否稳定。

## 本地开发运行

```bash
python -m pip install -r bridge_mesh_suite/requirements.txt
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python scripts/run_bridge_mesh_suite.py --no-deepseek
```

完整 GitHub Actions 运行要求 `ds` Environment 中存在 DeepSeek 密钥，并安装 CalculiX：

```bash
PYTHONPATH=. python scripts/run_bridge_mesh_suite.py --require-calculix
```
