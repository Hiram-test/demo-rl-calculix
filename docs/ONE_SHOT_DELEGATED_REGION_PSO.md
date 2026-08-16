# 单次委派区域智能体 + 母粒子微型 PSO

## 1. 固定决策链

本实现严格采用以下顺序，不允许回退为多轮多智能体搜索，也不允许 PSO 重新开放全部区域尺寸变量：

1. 全局智能体只委派一次完整区域尺寸向量
   \[
   \mathbf h^{(0)}=(h_1^{(0)},\ldots,h_K^{(0)}).
   \]
2. 每个区域智能体只在自己的委派尺寸上做一次误差—资源估计。
3. 每个区域智能体只发送一次消息；全部消息经过一次同步通信。
4. 每个区域智能体读取通信结果后，只调整一次自己的区域尺寸，形成
   \[
   \mathbf h^{\mathrm{MA}}=(h_1^{(1)},\ldots,h_K^{(1)}).
   \]
5. 总智能体只评估一次 \(\mathbf h^{\mathrm{MA}}\)，并将它定义为唯一母粒子。
6. PSO 只围绕该母粒子开放两个残余校准坐标，严格更新一代或两代。
7. 所有区域估计和 PSO 粒子评价都使用粗解锚定代理；只有最终选中的一个粒子允许调用一次真实网格编译与高保真求解器认证。

因此，完整求解预算保持为“一次粗解 + 一次终局认证”。PSO 内部不会调用 CalculiX。

## 2. 区域智能体的一次误差—资源估计

当前可运行原型使用粗解锚定的局部幂律响应：

\[
\widehat e_i(h_i)=e_i^{\rm ref}\left(\frac{h_i}{h_i^{\rm ref}}\right)^{p_i},
\qquad
\widehat c_i(h_i)=c_i^{\rm ref}\left(\frac{h_i^{\rm ref}}{h_i}\right)^{d_i}.
\]

其中 \(p_i>0\) 是局部误差阶次，\(d_i>0\) 是资源维数。真实接入时，这个接口可以直接替换成粗解上的 patch 反事实估计、ZZ 外推、局部残差响应或已有代理模型；状态机仍强制每个区域只调用一次。

令

\[
r_{E,i}=\log\frac{\widehat e_i}{e_i^\star},
\qquad
r_{C,i}=\log\frac{\widehat c_i}{c_i^\star},
\qquad x_i=\log h_i.
\]

区域智能体先独立求一个一维线性化修正：

\[
\Delta x_i^{\rm loc}
=
\operatorname{clip}
\frac{w_Cd_i r_{C,i}-w_Ep_i r_{E,i}}
{w_Ep_i^2+w_Cd_i^2+\lambda}.
\]

误差高于区域目标时，公式推动 \(h_i\) 变小；资源超过区域份额时，公式推动 \(h_i\) 变大。

## 3. 唯一一次通信与区域调整

每条区域消息只包含：本区误差、资源、对数残差、局部灵敏度、边际效率、通信前候选尺寸和置信度。

通信枢纽只汇总一次全局压力：

\[
r_E^g=\log\frac{\widehat E}{\varepsilon},
\qquad
r_C^g=\log\frac{\widehat C}{B},
\]

并以同样的一维线性化形式得到共同尺度压力 \(\Delta x^g\)。每个区域随后只执行一次最终调整：

\[
\Delta x_i
=
\Delta x_i^{\rm loc}
+\alpha_g\Delta x^g
-\alpha_\rho\left(\log\rho_i-\overline{\log\rho}\right)
+\alpha_n\left(\overline{x}_{\mathcal N_i}^{\rm cand}-x_i^{\rm cand}\right).
\]

这里 \(\rho_i\) 是区域边际误差—资源效率。高收益区域获得负的尺寸修正，即更细；低收益区域相反。最后一项只使用这一轮邻居消息协调相邻区域尺寸。完成后，区域智能体不再运行第二轮。

## 4. 总智能体形成唯一母粒子

全部区域调整结果直接拼接为

\[
\mathbf h^{\mathrm{MA}}.
\]

总智能体只对它做一次完整代理评估，得到预测误差、资源、相邻尺寸质量和约束残差适应度。此后不重新分区、不重新委派，也不重新调用区域智能体。

## 5. PSO 只校准两个残余自由度

PSO 不直接搜索 \((h_1,\ldots,h_K)\)。完整候选始终由母粒子通过两个坐标解码：

\[
\log h_i(s,\kappa)
=
\log h_i^{\mathrm{MA}}+s+\kappa v_i.
\]

- \(s\)：整体粗细尺度，用于消除总误差与总资源的共同残差。
- \(\kappa\)：资源转移强度，用于把少量资源从低边际收益区域移向高边际收益区域。

方向 \(\mathbf v\) 由区域通信中的边际效率编译，并满足一阶资源中性条件：

\[
\sum_i d_i\widehat c_i v_i=0.
\]

初始微型粒子群固定为五个点：母粒子、\(\pm s\) 两个轴向扰动和 \(\pm\kappa\) 两个轴向扰动。随后只允许一代或两代标准 PSO 更新。母粒子评估复用总智能体结果，因此：

- 一代：总代理评价 10 次，其中 PSO 新增 9 次；
- 两代：总代理评价 15 次，其中 PSO 新增 14 次；
- PSO 高保真求解次数始终为 0。

## 6. 当前可运行结果

五区域示例中，区域智能体通信后形成的母粒子为：

- 预测误差：`0.1069991626`，目标上限：`0.105`；
- 预测资源：`188.8924866`，预算：`194`。

母粒子表现为“资源尚有余量，但误差略超限”。一代微型 PSO 选择二维坐标 \((-0.02,0)\)，得到：

- 预测误差：`0.1045144933`；
- 预测资源：`193.7445810`；
- 误差、资源和相邻尺寸过渡均满足代理约束。

这只是对决策机制和调用预算的可重复验证，不冒充真实 CalculiX 终局认证。当前运行环境没有 `ccx`；脚本已提供 `--certifier-command` 接口，可把选中尺寸向量交给真实网格编译器与 CalculiX，并且只执行一次。

## 7. 运行

```bash
PYTHONPATH=src python3 scripts/run_one_shot_region_pso.py \
  --input examples/one_shot_region_pso_case.json \
  --pso-generations 1 \
  --output artifacts/one_generation_report.json
```

接入终局求解器时，认证命令必须读取 `{sizes_json}` 并生成 `{result_json}`：

```bash
PYTHONPATH=src python3 scripts/run_one_shot_region_pso.py \
  --input coarse_case.json \
  --pso-generations 2 \
  --certifier-command "python3 scripts/final_calculix_adapter.py --sizes {sizes_json} --output {result_json}" \
  --output artifacts/final_report.json
```

## 8. 自动验证的硬约束

测试直接验证以下不变量：

- 全局委派次数严格等于 1；
- 每个区域的局部估计次数严格等于 1；
- 通信轮次严格等于 1；
- 每个区域的尺寸调整次数严格等于 1；
- PSO 坐标维数严格等于 2；
- PSO 代数只能为 1 或 2；
- 一代和两代的代理评价预算分别固定为 10 和 15；
- 高保真认证器只接收最终选中的完整尺寸粒子，并且最多调用一次；
- 任何试图重复运行同一编排器的操作都会失败。
