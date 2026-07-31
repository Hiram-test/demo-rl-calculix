from __future__ import annotations  # 启用现代类型注解并保持测试环境兼容。

import copy  # 验证隐藏映射不会改写模型原始提案。
import json  # 读取冻结文件和公开反馈。
import tempfile  # 为每个测试创建独立临时目录。
import unittest  # 使用标准库测试框架避免新增测试依赖。
from pathlib import Path  # 管理临时实验目录和仓库文件路径。

import scripts.run_deepseek_crack_hidden_executor_adapter as task_runner  # 导入任务级提示词和运行编排而不启动真实模型。
from experiments.hidden_executor import task_controller  # 测试总体任务决策门和完成裁决。
from experiments.hidden_executor.contracts import canonical_json  # 生成稳定提案文本用于完整性比较。
from experiments.hidden_executor.contracts import freeze_proposal  # 测试先冻结后映射合同。
from experiments.hidden_executor.executor_adapter_v5 import execute_mapping  # 测试路线停止和任务完成候选不会触发全局结束。
from experiments.hidden_executor.executor_adapter_v5 import map_frozen_proposal  # 测试最终隐藏映射和任务作用域。

SYSTEM_PROMPT = task_runner.runner.SYSTEM_PROMPT  # 读取适配后实际发送给DeepSeek的系统提示词。
USER_INSTRUCTION = task_runner.runner.USER_INSTRUCTION  # 读取适配后实际发送给DeepSeek的每轮指令。
FORBIDDEN_VISIBLE_NAMES = ("crosscheck_with_closed_form", "compare_nearby_geometry_energy", "probe_fixed_physical_locations", "compare_region_average", "refine_current_model", "request_material_curve", "request_load_or_boundary_detail", "fracture_parameter_sequence", "refine_to_explicit_target_size")  # 定义绝不能出现在模型提示中的内部名称。


def experiment_proposal(change: str, measure: list[str]) -> dict[str, object]:  # 构造满足无工具合同的测试实验提案。
    return {  # 返回包含竞争假设和控制实验的完整对象。
        "competing_hypotheses": ["局部峰值反映不可收敛的理想化场", "当前离散仍不足以支持工程判断"],  # 提供两个可竞争解释。
        "evidence_refs": ["initial_evidence.initial_mesh_history"],  # 引用输入证据路径。
        "uncertainties": ["材料非线性参数尚未提供"],  # 保留模型适用性未知。
        "proposal_type": "experiment",  # 标记该对象为控制实验提案。
        "experiment": {"purpose": "区分局部奇异行为和整体离散不足", "change": change, "hold_fixed": ["几何外形", "材料弹性参数", "载荷和边界"], "measure": measure, "decision_rule": "若目标量稳定则支持第一条假设，否则支持第二条假设", "stop_condition": "完成当前受控实验后结束本路线步骤"},  # 定义不含工具名称的实验设计。
        "information_request": {},  # 保留统一输出字段但不请求信息。
        "provisional_answer": "暂时不能根据裂尖峰值直接判断网格是否充分。",  # 提供当前工程答复。
    }  # 完成测试提案。


def resolution_proposal() -> dict[str, object]:  # 构造覆盖原始三项工程问题的任务完成提案。
    return {  # 返回结构化总体任务完成声明。
        "competing_hypotheses": ["现有证据足以形成有条件工程结论", "现有证据仍不足以关闭原始问题"],  # 保留完成前的竞争判断。
        "evidence_refs": ["previous_rounds"],  # 引用已执行公开证据。
        "uncertainties": ["真实材料数据仍可能改变模型适用范围"],  # 保留明确不确定性。
        "proposal_type": "resolve_task",  # 声明总体任务完成候选。
        "final_answer": {"continue_refinement": "不再沿局部峰值继续加密。", "model_usability": "当前模型只在已列明假设和适用范围内用于判断。", "next_action": "使用已验证评价量并补充缺失材料事实。", "remaining_uncertainties": ["材料非线性数据尚未完全给出"]},  # 回答三个工程问题并保留不确定性。
    }  # 完成任务级提案。


def synthetic_initial() -> dict[str, object]:  # 构造不包含路线答案的最小初始任务事实。
    return {"user_question": "是否继续加密、当前模型能否判断裂纹、下一步怎么处理？", "model_facts": {"material": {"plastic_curve": None}}}  # 返回材料数据不完整的原始问题。


def completed_audit(operation: str) -> dict[str, object]:  # 构造一个已经产生真实公开证据的审计摘要。
    return {"internal_mapping": {"operation": operation}, "public_feedback": {"status": "completed"}}  # 返回任务控制器可识别的最小审计对象。


class HiddenExecutorContractTests(unittest.TestCase):  # 汇总隔离提示、冻结、执行器和任务控制合同测试。
    def test_model_prompt_contains_no_internal_tool_catalog(self) -> None:  # 检查模型提案阶段看不到任何内部工具名称。
        visible_prompt = SYSTEM_PROMPT + USER_INSTRUCTION  # 组合全部固定模型可见提示词。
        for forbidden in FORBIDDEN_VISIBLE_NAMES:  # 遍历全部内部名称。
            self.assertNotIn(forbidden, visible_prompt)  # 断言提示词没有泄露内部执行能力。
        self.assertIn("switch_route", visible_prompt)  # 断言提示明确提供路线级结束语义。
        self.assertIn("resolve_task", visible_prompt)  # 断言提示明确提供任务级完成语义。
        self.assertNotIn("proposal_type只能是experiment、request_information或stop", visible_prompt)  # 断言旧无作用域stop合同已经移除。

    def test_proposal_is_frozen_before_hidden_mapping(self) -> None:  # 检查映射只读取已经写盘并封存的提案。
        proposal = experiment_proposal("把两侧裂纹各延长0.5 mm", ["比较延长前后的总应变能变化"])  # 构造自主提出的裂纹微扰实验。
        original = copy.deepcopy(proposal)  # 保存映射前提案副本。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时实验目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 先写入提案和哈希封条。
            round_dir = Path(frozen["round_dir"])  # 获取当前轮冻结目录。
            mapping = map_frozen_proposal(round_dir, str(frozen["seal"]["proposal_sha256"]))  # 在冻结完成后调用隐藏映射。
            self.assertEqual(mapping["operation"], "geometry_energy")  # 断言执行器忠实识别裂纹长度与能量比较意图。
            self.assertEqual(canonical_json(proposal), canonical_json(original))  # 断言映射没有改写内存中的提案。
            saved = json.loads((round_dir / "proposal.json").read_text(encoding="utf-8"))  # 重新读取磁盘冻结内容。
            self.assertEqual(canonical_json(saved), canonical_json(original))  # 断言磁盘提案也保持原样。

    def test_refinement_with_energy_based_k_maps_as_one_experiment(self) -> None:  # 检查执行器不会把模型要求的联合实验拆成普通加密。
        proposal = experiment_proposal("Refine the crack-tip local element size from 2.5 mm to 1.25 mm.", ["Total strain energy before and after a one-element crack extension", "Energy release rate G", "Stress intensity factor K = sqrt(EG)"])  # 构造真实轨迹中出现的英文联合实验。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结联合实验提案。
            mapping = map_frozen_proposal(Path(frozen["round_dir"]), str(frozen["seal"]["proposal_sha256"]))  # 调用模型不可见联合映射。
            self.assertEqual(mapping["operation"], "refine_and_fracture_parameter")  # 断言加密和派生量复算保持为同一个实验目的。

    def test_refinement_with_unavailable_contour_j_is_not_partially_executed(self) -> None:  # 检查执行器不会只完成联合实验中的网格加密部分。
        proposal = experiment_proposal("Refine the crack-tip mesh from 2.5 mm to 1.25 mm.", ["J-integral values on multiple contours", "Consistency of J across contours"])  # 构造当前后端缺少多围道能力的联合实验。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 在映射前冻结完整联合实验目的。
            mapping = map_frozen_proposal(Path(frozen["round_dir"]), str(frozen["seal"]["proposal_sha256"]))  # 调用最终忠实性映射门。
            self.assertEqual(mapping["operation"], "unsupported")  # 断言执行器整项拒绝且不会降格为普通网格加密。

    def test_explicit_target_size_is_not_replaced_by_default_grid(self) -> None:  # 检查普通加密严格保留模型给出的目标尺寸。
        proposal = experiment_proposal("将裂尖附近网格目标尺寸从2.5 mm减小到1.25 mm。", ["远程开口位移", "总应变能", "裂尖最大应力"])  # 构造真实轨迹中的显式目标尺寸实验。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结目标尺寸提案。
            mapping = map_frozen_proposal(Path(frozen["round_dir"]), str(frozen["seal"]["proposal_sha256"]))  # 调用最终隐藏映射。
            self.assertEqual(mapping["operation"], "refine_to_explicit_target_size")  # 断言执行器不会回退到默认网格级别。
            self.assertAlmostEqual(float(mapping["target_size_mm"]), 1.25)  # 断言目标尺寸按提案原值保存。

    def test_legacy_stop_becomes_route_transition(self) -> None:  # 检查旧stop不会再结束总体任务。
        proposal = {"competing_hypotheses": ["当前路线已完成", "总体问题仍未完成"], "evidence_refs": ["initial_evidence"], "uncertainties": ["还没有适合工程决策的评价量证据"], "proposal_type": "stop", "provisional_answer": "停止继续追逐局部峰值，下一步改用新的评价路线。"}  # 构造遗留无作用域stop提案。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结遗留stop提案。
            round_dir = Path(frozen["round_dir"])  # 读取当前轮目录。
            mapping = map_frozen_proposal(round_dir, str(frozen["seal"]["proposal_sha256"]))  # 生成模型不可见路线切换映射。
            feedback = execute_mapping(round_dir, str(frozen["seal"]["proposal_sha256"]), mapping)  # 生成公开路线切换反馈。
            self.assertEqual(mapping["operation"], "route_transition")  # 断言stop只映射到当前路线结束。
            self.assertEqual(feedback["status"], "route_transition")  # 断言公开状态要求总体任务继续。
            self.assertTrue(feedback["observations"]["task_continues"])  # 断言反馈明确保留下一轮。

    def test_switch_route_requires_another_stage(self) -> None:  # 检查显式路线切换也不会触发全局完成。
        proposal = {"competing_hypotheses": ["当前评价量不适合", "当前评价量仍可能有效"], "evidence_refs": ["previous_rounds"], "uncertainties": ["替代评价量尚未验证"], "proposal_type": "switch_route", "route_transition": {"route_conclusion": "停止追逐局部峰值。", "next_route": "提出能够验证工程评价量的新实验。", "reason": "当前路线无法回答原始问题。"}}  # 构造结构化路线切换提案。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结路线切换提案。
            round_dir = Path(frozen["round_dir"])  # 读取冻结目录。
            mapping = map_frozen_proposal(round_dir, str(frozen["seal"]["proposal_sha256"]))  # 映射路线控制操作。
            feedback = execute_mapping(round_dir, str(frozen["seal"]["proposal_sha256"]), mapping)  # 生成下一轮公开反馈。
            self.assertEqual(feedback["status"], "route_transition")  # 断言当前路线结束后继续总体任务。
            self.assertEqual(feedback["observations"]["next_route"], "提出能够验证工程评价量的新实验。")  # 断言下一路线按冻结提案原样保留。

    def test_resolution_is_rejected_before_decision_quantity_evidence(self) -> None:  # 检查只完成网格实验时不能关闭总体任务。
        proposal = resolution_proposal()  # 构造过早的任务完成声明。
        history: list[dict[str, object]] = []  # 初始化无外部阻塞公开历史。
        audits = [completed_audit("refine_to_explicit_target_size")]  # 只提供新增网格证据。
        adjudication = task_controller.adjudicate_resolution(synthetic_initial(), history, audits, proposal)  # 调用独立任务门裁决。
        self.assertEqual(adjudication["status"], "resolution_rejected")  # 断言局部网格结论不能结束原始问题。
        self.assertIn("decision_quantity_evidence", adjudication["task_state"]["unresolved_decision_gates"])  # 断言适合工程决策的评价量仍需隐藏工具阶段验证。

    def test_resolution_passes_after_mesh_and_decision_quantity_evidence(self) -> None:  # 检查隐藏工具证据和完整答复齐备时才允许任务结束。
        proposal = resolution_proposal()  # 构造完整任务答复。
        audits = [completed_audit("refine_to_explicit_target_size"), completed_audit("fracture_parameter_sequence")]  # 提供网格策略和替代评价量的真实执行证据。
        adjudication = task_controller.adjudicate_resolution(synthetic_initial(), [], audits, proposal)  # 调用独立完成门。
        self.assertEqual(adjudication["status"], "task_resolved")  # 断言全部通用决策门关闭后允许完成。
        self.assertEqual(adjudication["task_state"]["unresolved_decision_gates"], [])  # 断言最终不存在未决门。

    def test_public_task_packet_contains_no_hidden_operation_names(self) -> None:  # 检查任务控制信息不会反向泄露隐藏能力。
        state = task_controller.assess_task_state(synthetic_initial(), [], [])  # 构造初始未决任务状态。
        packet = task_controller.public_task_packet(state)  # 生成下一轮模型可见任务合同。
        visible_text = json.dumps(packet, ensure_ascii=False)  # 序列化公开任务包。
        for forbidden in FORBIDDEN_VISIBLE_NAMES:  # 遍历内部名称。
            self.assertNotIn(forbidden, visible_text)  # 断言任务门只描述工程目标而不列出隐藏工具。
        self.assertIn("unresolved_decision_gates", packet)  # 断言模型明确看到尚未解决的总体目标。

    def test_ambiguous_proposal_is_not_replaced_with_a_known_route(self) -> None:  # 检查执行器不会把未知实验偷偷换成预设方法。
        proposal = experiment_proposal("改变观测时间窗", ["比较频域响应中的相位差"])  # 构造当前裂纹后端无法执行的实验。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结未知实验提案。
            mapping = map_frozen_proposal(Path(frozen["round_dir"]), str(frozen["seal"]["proposal_sha256"]))  # 调用隐藏映射。
            self.assertEqual(mapping["operation"], "unsupported")  # 断言执行器诚实拒绝而不替换实验目的。

    def test_closed_form_feedback_removes_internal_tool_field(self) -> None:  # 检查理论参照反馈不向模型泄露内部函数标记。
        proposal = experiment_proposal("不改变模型，计算当前理想化中心裂纹的解析参照。", ["理论应力强度因子", "理论能量释放率"])  # 构造理论校核实验。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结理论参照提案。
            round_dir = Path(frozen["round_dir"])  # 读取冻结目录。
            mapping = map_frozen_proposal(round_dir, str(frozen["seal"]["proposal_sha256"]))  # 生成隐藏理论参照映射。
            feedback = execute_mapping(round_dir, str(frozen["seal"]["proposal_sha256"]), mapping)  # 执行理论参照并净化公开反馈。
            self.assertNotIn("tool", json.dumps(feedback, ensure_ascii=False))  # 断言公开反馈没有内部工具字段。

    def test_new_runner_only_writes_to_explicit_output_directory(self) -> None:  # 检查隔离运行器没有生产目录写入常量。
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_deepseek_crack_hidden_executor_adapter.py").read_text(encoding="utf-8")  # 读取任务级运行器源码。
        self.assertIn("experiment_result.json", source)  # 断言结果写入显式输出目录中的统一JSON。
        self.assertNotIn("experiments/results/latest_deepseek_crack_open_discovery", source)  # 断言不会覆盖旧开放发现结果。
        self.assertIn("inconclusive_budget_exhausted", source)  # 断言预算耗尽不会伪装成任务完成。


if __name__ == "__main__":  # 仅在测试文件直接运行时启动测试。
    unittest.main()  # 执行全部隔离合同测试。
