from __future__ import annotations  # 启用现代类型注解并保持测试环境兼容。

import unittest  # 使用标准库测试框架避免新增依赖。

import scripts.run_deepseek_crack_hidden_skill_planner_v2 as runner_v2  # 导入扩展双API运行配置而不启动真实调用。
from experiments.skill_planner.executor_v2 import _collect_requested_information  # 测试嵌套材料请求标准化。
from experiments.skill_planner.executor_v2 import _numbers_with_text  # 测试冻结自然语言中的数值溯源。
from experiments.skill_planner.registry import SkillContext  # 构造确定性Skill测试上下文。
from experiments.skill_planner.skills_v2 import build_registry  # 测试扩展隐藏Skill目录。


class HiddenSkillPlannerV2Tests(unittest.TestCase):  # 汇总新增模型适用性和任务闭环合同测试。
    def test_irwin_skill_is_registered_and_computes_condition_rows(self) -> None:  # 检查第二API可以选择真实Irwin条件评估Skill。
        registry = build_registry()  # 构造扩展Skill注册表。
        self.assertIn("fracture.irwin_plastic_zone", registry.ids())  # 断言新增Skill存在于隐藏目录。
        skill = registry.get("fracture.irwin_plastic_zone")  # 读取确定性处理函数。
        context = SkillContext(initial_evidence={}, public_history=[], prior_skill_outputs={})  # 构造不含额外事实的最小上下文。
        result = skill.handler({"stress_intensity_mpa_sqrt_mm": 853.03, "yield_strengths_mpa": [250.0, 400.0, 600.0], "half_crack_mm": 20.0, "ligament_mm": 80.0, "plane_condition": "plane_stress"}, context)  # 执行冻结提案中出现过的条件敏感性计算。
        rows = result["observations"]["rows"]  # 读取三个屈服强度条件结果。
        self.assertEqual(len(rows), 3)  # 断言每个假设强度均产生一行。
        self.assertGreater(rows[0]["plastic_zone_radius_mm"], rows[1]["plastic_zone_radius_mm"])  # 断言屈服强度升高时塑性区减小。
        self.assertGreater(rows[1]["plastic_zone_radius_mm"], rows[2]["plastic_zone_radius_mm"])  # 断言单调物理关系继续成立。
        self.assertEqual(result["status"], "completed")  # 断言该Skill是纯后处理可执行能力。

    def test_nested_material_requests_are_flattened_without_skill_names(self) -> None:  # 检查任务控制器可以识别组合结果中的真实外部阻塞。
        value = {"result_groups": [{"observations": {"requested_information": ["yield_strength", "stress_strain_curve"]}}, {"observations": {"requested_information": ["yield_strength"]}}]}  # 构造真实组合Skill反馈形状。
        self.assertEqual(_collect_requested_information(value), ["yield_strength", "stress_strain_curve"])  # 断言请求被稳定去重并提升。

    def test_numeric_literals_in_frozen_text_are_grounded(self) -> None:  # 检查明确写在工程语言中的目标参数不会被误判为幻觉。
        value = {"change": "将nx从80增加到160，使裂尖尺寸从2.5 mm降至1.25 mm", "measure": ["K_I变化小于1%"]}  # 构造真实第一API提案中的自然语言参数。
        numbers = _numbers_with_text(value)  # 提取结构化文本中的全部真实字面量。
        for expected in (80.0, 160.0, 2.5, 1.25, 1.0):  # 遍历应当被来源门识别的参数。
            self.assertIn(expected, numbers)  # 断言每个显式数值均可作为Skill参数来源。

    def test_mesh_sequence_closes_mesh_strategy_evidence_category(self) -> None:  # 检查跨网格K序列被视为网格策略证据。
        self.assertIn("fracture_parameter_sequence", runner_v2.runner.task_controller._MESH_EVIDENCE_OPERATIONS)  # 断言任务控制器不会忽略已有多网格比较。

    def test_decision_prompt_avoids_repeated_external_requests(self) -> None:  # 检查第一API被明确要求处理已记录阻塞而非消耗轮次重复询问。
        visible = runner_v2.runner.DECISION_SYSTEM_PROMPT + runner_v2.runner.DECISION_USER_INSTRUCTION  # 组合第一API实际可见固定提示。
        self.assertIn("不得反复请求", visible)  # 断言系统提示包含重复请求禁止语义。
        self.assertIn("条件性最终答复", visible)  # 断言外部阻塞后允许提交条件性任务完成声明。
        self.assertNotIn("fracture.irwin_plastic_zone", visible)  # 断言新增Skill名称仍不泄露给第一API。


if __name__ == "__main__":  # 允许直接运行本测试文件。
    unittest.main()  # 启动标准库测试运行器。
