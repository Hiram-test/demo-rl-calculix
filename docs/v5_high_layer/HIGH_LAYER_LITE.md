# V5 提高层轻量诊断档案

本切片把 FEN-003、FEN-014 和 CCX-015 记录为机器可读的未执行诊断设计。三个 JSON 的 status 均为 draft_not_executed；它们没有生成网格、没有调用模型、没有运行 CalculiX，也不包含工程结论。

## 共同原则

- 所有几何、材料、载荷、边界、单位、QoI、容差、质量阈值、预算和优化配置必须来自当前任务证据。
- 不允许从 V4、历史 benchmark、旧 fixture 或其他案例静默迁移数值。
- 缺失事实必须写入 missing_facts，不能用 null、TBD、unknown 或模糊占位替代。
- fixed_qoi_contract_fields 和 stop_condition_fields 只声明后续必须冻结的字段名，不提供任何默认值。
- 求解器收敛、工作流完成或 JSON 格式正确都不能单独证明工程结果可信。

## 顶层字段

profile_version 表示本组静态档案的合同版本。

case_id 是研究计划冻结的案例标识。

status 固定为 draft_not_executed，防止档案被误认为真实运行证据。

objective 描述该案例要验证的工程判断能力。

decision_question 描述系统最终需要回答的决策问题。

applicable_skill_ids 只能引用当前 skills/engineering 目录已经存在的通用 Skill。

skill_coverage_gaps 记录当前 Skill 库尚未覆盖的通用能力；它不是可执行 Skill ID。

required_evidence 列出每类证据的名称、用途和可接受来源。

minimal_experiment 记录最小区分性实验的设计、步骤和未执行状态。

fixed_qoi_contract_fields 列出未来必须固定的 QoI 字段名。

stop_condition_fields 列出未来必须从当前任务建立的停止、回退或策略切换字段名。

missing_facts 沿用 ProblemManifest 的 path、reason、question 和 acceptable_sources 结构。

interpretation_limits 记录即使未来获得部分结果也不能越过的解释边界。

## 固定 QoI 合同字段

qoi_id 用于在 trace、结果和报告中稳定引用同一目标量。

engineering_purpose 说明该目标量服务的具体工程决策。

physical_quantity 与 component_or_invariant 共同冻结物理量及其分量或不变量。

unit、spatial_selection、coordinate_frame 和 analysis_step_or_time 冻结单位、空间选择、坐标系与分析步。

aggregation_or_reduction、extraction_method 和 extraction_implementation_ref 冻结聚合方式、提取方法与确定性实现。

comparison_rule 冻结不同网格或候选之间如何比较。

singularity_exposure 记录该量是否暴露于点载荷、尖角、裂尖或其他奇异性。

provenance_refs 连接当前输入、代码、工作流、原始结果和用户确认。

## 案例定位

### FEN-003

该案例回答是否需要继续加密。现有 problem-definition-source-audit 和 mesh-convergence-and-singularity 已覆盖主要推理框架，因此本档案不声明额外 Skill 缺口。

最小实验先冻结当前模型和 QoI，再审计已有受控网格序列。只有现有证据无法区分离散误差、提取问题或奇异性迹象时，才补充一个具有区分力的网格层级。

### FEN-014

该案例回答当前外部网格是否可信。现有来源审计 Skill 可以冻结输入与事实，但缺少外部网格完整性、方向、单位尺度、标签集合和求解器映射的通用诊断 Skill。

最小实验保持物理模型与求解设置不变，对可运行规则网格和失败导入网格做分层差分检查。静态预检未通过时不启动求解器。

### CCX-015

该案例回答如何安全地自动细化。现有来源审计、网格收敛和优化准备 Skill 只能部分覆盖，需要补充质量约束细化、过渡区、求解前硬门、回退和策略切换合同。

最小实验只从最后一个已接受状态生成一个受约束候选。候选先通过拓扑与质量预检，才允许求解；QoI 改善不能覆盖质量硬门或求解稳定性失败。

## 解释边界

FEN-003 的单个新增网格层级只能帮助区分当前竞争解释，不能证明普遍收敛。

FEN-014 的文件解析成功不能证明单位、方向、标签和边界映射正确。

CCX-015 的单个候选不能证明优化充分性，局部尺寸变小也不能保证误差下降。

本切片不修改 V4 论文、旧结果、旧 trace、旧工作流或任何 PR18 及以后内容。
