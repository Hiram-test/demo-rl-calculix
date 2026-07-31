from __future__ import annotations  # 启用现代类型注解并保持测试环境兼容。

import json  # 检查公开反馈没有泄露隐藏Skill名称。
import tempfile  # 为冻结计划和执行审计创建独立临时目录。
import unittest  # 使用标准库测试框架避免新增测试依赖。
from pathlib import Path  # 管理临时轮次目录。

from experiments.hidden_executor.contracts import freeze_proposal  # 测试第一API提案先冻结。
from experiments.skill_planner.contracts import freeze_skill_plan  # 测试第二API计划在Skill执行前冻结。
from experiments.skill_planner.contracts import validate_skill_plan  # 测试完整覆盖和禁止部分执行合同。
from experiments.skill_planner.executor import execute_frozen_plan  # 测试确定性Skill执行和公开反馈隔离。
from experiments.skill_planner.executor import validate_runtime_plan  # 测试未知Skill、参数来源和数值幻觉拒绝。
from experiments.skill_planner.planner_api import PLANNER_SYSTEM_PROMPT  # 检查第二API承担Skill规划职责。
from experiments.skill_planner.registry import SkillContext  # 构造纯后处理Skill的合成真实历史。
from experiments.skill_planner.skills import build_registry  # 读取实际隐藏Skill目录和处理函数。
from scripts.run_deepseek_crack_hidden_skill_planner import DECISION_SYSTEM_PROMPT  # 检查第一API看不到Skill标识。
from scripts.run_deepseek_crack_hidden_skill_planner import DECISION_USER_INSTRUCTION  # 检查每轮第一API指令不泄露目录。


class HiddenSkillPlannerTests(unittest.TestCase):  # 汇总双API隔离、计划冻结和Skill执行合同测试。
    def setUp(self) -> None:  # 为每个测试构造独立Skill注册表。
        self.registry = build_registry()  # 加载真实隐藏Skill目录。
        self.proposal = {"competing_hypotheses": ["材料数据足够", "材料数据不足"], "evidence_refs": ["initial_evidence"], "uncertainties": ["屈服强度未知"], "proposal_type": "request_information", "information_request": {"question": "请提供材料屈服强度"}}  # 构造满足第一API合同的材料请求提案。
        self.evidence = {"initial_evidence": {"user_question": "判断裂纹模型并给出下一步", "model_facts": {"material": {"yield_strength_mpa": None}}, "initial_mesh_history": [{"nx": 40, "h_local_mm": 5.0}, {"nx": 60, "h_local_mm": 3.3333333333333335}, {"nx": 80, "h_local_mm": 2.5}]}, "previous_rounds": []}  # 构造不含Skill目录的真实证据包。

    def _material_plan(self) -> dict[str, object]:  # 构造完整覆盖外部材料需求的合法Skill计划。
        return {"experiment_spec": {"objective": "获取线弹性适用性判断所需的材料屈服事实", "scope": "information_request", "interventions": [], "invariants": ["现有有限元模型和结果不变"], "observables": [], "derivations": [], "external_dependencies": [{"id": "material_yield", "description": "材料单轴屈服强度"}], "acceptance_rule": "取得真实屈服强度后继续评估塑性区", "completion_scope": "current_step"}, "plan_type": "execute", "calls": [{"call_id": "call_1", "skill_id": "material.request", "arguments": {"fields": ["屈服强度"]}, "argument_sources": {"fields": "proposal"}, "covers": ["material_yield"], "depends_on": []}], "uncovered_requirements": [], "proposal_fully_preserved": True, "unsupported_reason": None}  # 返回第二API应生成的结构。

    def test_decision_api_cannot_see_skill_catalog(self) -> None:  # 检查第一API提示不包含具体Skill标识或目录内容。
        visible = DECISION_SYSTEM_PROMPT + DECISION_USER_INSTRUCTION  # 组合第一API全部固定提示。
        for skill_id in self.registry.ids():  # 遍历实际隐藏Skill标识。
            self.assertNotIn(skill_id, visible)  # 断言第一API看不到任何Skill名称。
        self.assertNotIn("available_skills", visible)  # 断言第一API提示没有目录字段。
        self.assertIn("不知道执行环境拥有哪些软件、Skill或能力", visible)  # 断言明确禁止猜测隐藏能力。

    def test_planner_api_is_explicitly_skill_aware(self) -> None:  # 检查第二API提示承担目录可见的编译职责。
        self.assertIn("隐藏Skill执行编译器", PLANNER_SYSTEM_PROMPT)  # 断言第二API角色明确。
        self.assertIn("Skill目录", PLANNER_SYSTEM_PROMPT)  # 断言第二API可以读取目录。
        self.assertIn("不能只执行容易的一部分", PLANNER_SYSTEM_PROMPT)  # 断言禁止部分执行。

    def test_partial_plan_is_rejected(self) -> None:  # 检查遗漏外部依赖时不能声称完整执行。
        plan = self._material_plan()  # 构造合法基础计划。
        plan["calls"] = []  # 删除唯一覆盖调用制造部分执行。
        plan["proposal_fully_preserved"] = True  # 模拟第二API错误声称完整保真。
        errors = validate_skill_plan(plan)  # 执行结构合同校验。
        self.assertTrue(any("execute plan must contain" in error or "coverage declarations" in error for error in errors))  # 断言空或遗漏计划被拒绝。

    def test_unknown_skill_is_rejected_deterministically(self) -> None:  # 检查第二API不能调用目录外能力。
        plan = self._material_plan()  # 构造合法基础计划。
        plan["calls"][0]["skill_id"] = "imaginary.skill"  # 替换为不存在的Skill标识。
        errors = validate_runtime_plan(plan, self.proposal, self.evidence, self.registry)  # 执行目录和参数来源校验。
        self.assertTrue(any("unknown skill" in error for error in errors))  # 断言未知Skill被确定性拒绝。

    def test_ungrounded_numeric_argument_is_rejected(self) -> None:  # 检查第二API不能发明冻结提案和证据中没有的网格尺寸。
        plan = {"experiment_spec": {"objective": "加密网格", "scope": "experiment", "interventions": ["减小网格尺寸"], "invariants": ["几何和载荷不变"], "observables": [{"id": "mesh_result", "description": "新网格有限元响应"}], "derivations": [], "external_dependencies": [], "acceptance_rule": "比较结果变化", "completion_scope": "current_step"}, "plan_type": "execute", "calls": [{"call_id": "call_1", "skill_id": "mesh.refine", "arguments": {"target_h_mm": 1.234567}, "argument_sources": {"target_h_mm": "proposal"}, "covers": ["mesh_result"], "depends_on": []}], "uncovered_requirements": [], "proposal_fully_preserved": True, "unsupported_reason": None}  # 构造带无来源数值的计划。
        errors = validate_runtime_plan(plan, self.proposal, self.evidence, self.registry)  # 执行数值来源校验。
        self.assertTrue(any("ungrounded numeric value" in error for error in errors))  # 断言无来源目标尺寸被拒绝。

    def test_plan_is_frozen_before_skill_execution(self) -> None:  # 检查第二API计划先写盘封存再调用Skill。
        plan = self._material_plan()  # 构造合法材料请求计划。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时实验目录。
            root = Path(directory)  # 转换为路径对象。
            frozen = freeze_proposal(self.proposal, root, 1, "0" * 64)  # 先冻结第一API提案。
            round_dir = Path(frozen["round_dir"])  # 定位当前轮目录。
            proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取提案摘要。
            seal = freeze_skill_plan(round_dir, plan, proposal_hash, self.registry.catalog_hash())  # 冻结第二API计划和目录版本。
            self.assertTrue((round_dir / "skill_plan.json").is_file())  # 断言原始Skill计划已经写盘。
            self.assertTrue((round_dir / "skill_plan_seal.json").is_file())  # 断言计划封条已经写盘。
            feedback, audit = execute_frozen_plan(round_dir, proposal_hash, str(seal["skill_plan_sha256"]), self.proposal, self.evidence, self.registry)  # 执行冻结后的真实材料请求Skill。
            self.assertEqual(feedback["status"], "information_required")  # 断言外部事实被正确标记为阻塞。
            self.assertEqual(audit["skill_calls"][0]["skill_id"], "material.request")  # 断言内部审计保留真实Skill标识。
            visible = json.dumps(feedback, ensure_ascii=False)  # 序列化第一API可见反馈。
            self.assertNotIn("material.request", visible)  # 断言公开反馈不泄露Skill名称。
            self.assertTrue((round_dir / "skill_plan_validation.json").is_file())  # 断言确定性计划校验收据已经保存。
            self.assertTrue((round_dir / "skill_execution_audit.json").is_file())  # 断言完整内部Skill审计已经保存。

    def test_richardson_skill_consumes_existing_public_evidence(self) -> None:  # 检查数学后处理Skill不会重新运行有限元或依赖关键词适配器。
        history = [{"round": 1, "execution_feedback": {"status": "completed", "observations": {"rows": [{"h_local_mm": 5.0, "stress_intensity_mpa_sqrt_mm": 869.688}, {"h_local_mm": 3.3333333333333335, "stress_intensity_mpa_sqrt_mm": 858.561}, {"h_local_mm": 2.5, "stress_intensity_mpa_sqrt_mm": 853.030}]}}}]  # 构造真实三档公开K序列。
        context = SkillContext(initial_evidence=self.evidence["initial_evidence"], public_history=history, prior_skill_outputs={})  # 构造纯后处理Skill上下文。
        skill = self.registry.get("postprocess.richardson")  # 读取注册表中的外推Skill。
        result = skill.handler({"source_field": "stress_intensity_mpa_sqrt_mm", "source_round": 1}, context)  # 执行广义Richardson外推。
        self.assertEqual(result["status"], "completed")  # 断言纯后处理成功完成。
        self.assertIn("extrapolated_value", result["observations"])  # 断言返回连续极限估计。
        self.assertIn("observed_order", result["observations"])  # 断言返回观测收敛阶次。
        self.assertIn("不运行新的有限元", result["executed_change"])  # 断言Skill没有把数学后处理误配成新求解。

    def test_catalog_hash_is_stable(self) -> None:  # 检查Skill目录版本可被封条稳定绑定。
        first = self.registry.catalog_hash()  # 计算第一次目录摘要。
        second = build_registry().catalog_hash()  # 重新构造注册表并计算摘要。
        self.assertEqual(first, second)  # 断言相同能力合同产生相同目录哈希。
        self.assertGreaterEqual(len(self.registry.ids()), 6)  # 断言初始目录覆盖多项可组合能力。


if __name__ == "__main__":  # 仅在测试文件直接执行时启动测试。
    unittest.main()  # 执行全部双API和Skill合同测试。
