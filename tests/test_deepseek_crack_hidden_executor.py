from __future__ import annotations  # 启用现代类型注解并保持测试环境兼容。

import copy  # 验证隐藏映射不会改写模型原始提案。
import json  # 读取冻结文件和公开反馈。
import tempfile  # 为每个测试创建独立临时目录。
import unittest  # 使用标准库测试框架避免新增测试依赖。
from pathlib import Path  # 管理临时实验目录和仓库文件路径。

from experiments.hidden_executor.contracts import canonical_json  # 生成稳定提案文本用于完整性比较。
from experiments.hidden_executor.contracts import freeze_proposal  # 测试先冻结后映射合同。
from experiments.hidden_executor.executor_adapter_v3 import execute_mapping  # 测试目标尺寸忠实执行和统一公开反馈净化。
from experiments.hidden_executor.executor_adapter_v3 import map_frozen_proposal  # 测试最终多层隐藏映射。
from scripts.run_deepseek_crack_hidden_executor import SYSTEM_PROMPT  # 检查系统提示没有工具目录。
from scripts.run_deepseek_crack_hidden_executor import USER_INSTRUCTION  # 检查每轮指令没有工具暗示。

FORBIDDEN_VISIBLE_NAMES = ("crosscheck_with_closed_form", "compare_nearby_geometry_energy", "probe_fixed_physical_locations", "compare_region_average", "refine_current_model", "request_material_curve", "request_load_or_boundary_detail")  # 定义绝不能出现在模型提示中的内部名称。


def experiment_proposal(change: str, measure: list[str]) -> dict[str, object]:  # 构造满足无工具合同的测试实验提案。
    return {  # 返回包含竞争假设和控制实验的完整对象。
        "competing_hypotheses": ["局部峰值反映不可收敛的理想化场", "当前离散仍不足以支持工程判断"],  # 提供两个可竞争解释。
        "evidence_refs": ["initial_evidence.initial_mesh_history"],  # 引用输入证据路径。
        "uncertainties": ["材料非线性参数尚未提供"],  # 保留模型适用性未知。
        "proposal_type": "experiment",  # 标记该对象为控制实验提案。
        "experiment": {"purpose": "区分局部奇异行为和整体离散不足", "change": change, "hold_fixed": ["几何外形", "材料弹性参数", "载荷和边界"], "measure": measure, "decision_rule": "若目标量稳定则支持第一条假设，否则支持第二条假设", "stop_condition": "相邻两次目标量变化低于预设容差"},  # 定义不含工具名称的实验设计。
        "information_request": {},  # 保留统一输出字段但不请求信息。
        "provisional_answer": "暂时不能根据裂尖峰值直接判断网格是否充分。",  # 提供当前工程答复。
    }  # 完成测试提案。


class HiddenExecutorContractTests(unittest.TestCase):  # 汇总隔离提示、冻结和执行器合同测试。
    def test_model_prompt_contains_no_internal_tool_catalog(self) -> None:  # 检查模型提案阶段看不到任何内部工具名称。
        visible_prompt = SYSTEM_PROMPT + USER_INSTRUCTION  # 组合全部固定模型可见提示词。
        for forbidden in FORBIDDEN_VISIBLE_NAMES:  # 遍历全部内部名称。
            self.assertNotIn(forbidden, visible_prompt)  # 断言提示词没有泄露内部执行能力。
        self.assertNotIn("工具目录", visible_prompt)  # 断言提示词没有列出候选动作菜单。
        self.assertIn("不知道执行环境拥有哪些软件或能力", visible_prompt)  # 断言提示明确要求无能力假设。

    def test_proposal_is_frozen_before_hidden_mapping(self) -> None:  # 检查映射只读取已经写盘并封存的提案。
        proposal = experiment_proposal("把两侧裂纹各延长 0.5 mm", ["比较延长前后的总应变能变化"])  # 构造自主提出的裂纹微扰实验。
        original = copy.deepcopy(proposal)  # 保存映射前提案副本。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时实验目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 先写入提案和哈希封条。
            round_dir = Path(frozen["round_dir"])  # 获取当前轮冻结目录。
            self.assertTrue((round_dir / "proposal.json").is_file())  # 断言模型原始提案已经落盘。
            self.assertTrue((round_dir / "proposal_seal.json").is_file())  # 断言哈希封条已经落盘。
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
            self.assertEqual(mapping["operation"], "refine_and_fracture_parameter")  # 断言加密和K复算保持为同一个实验目的。

    def test_explicit_target_size_is_not_replaced_by_default_grid(self) -> None:  # 检查普通加密严格保留模型给出的目标尺寸。
        proposal = experiment_proposal("将裂尖附近网格目标尺寸从2.5 mm减小到1.25 mm。", ["远程开口位移", "总应变能", "裂尖最大应力"])  # 构造真实轨迹中的显式目标尺寸实验。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结目标尺寸提案。
            mapping = map_frozen_proposal(Path(frozen["round_dir"]), str(frozen["seal"]["proposal_sha256"]))  # 调用最终隐藏映射。
            self.assertEqual(mapping["operation"], "refine_to_explicit_target_size")  # 断言执行器不会回退到默认网格级别。
            self.assertAlmostEqual(float(mapping["target_size_mm"]), 1.25)  # 断言目标尺寸按提案原值保存。

    def test_ambiguous_proposal_is_not_replaced_with_a_known_route(self) -> None:  # 检查执行器不会把未知实验偷偷换成预设方法。
        proposal = experiment_proposal("改变观测时间窗", ["比较频域响应中的相位差"])  # 构造当前裂纹后端无法执行的实验。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结未知实验提案。
            mapping = map_frozen_proposal(Path(frozen["round_dir"]), str(frozen["seal"]["proposal_sha256"]))  # 调用隐藏映射。
            self.assertEqual(mapping["operation"], "unsupported")  # 断言执行器诚实拒绝而不替换实验目的。

    def test_public_feedback_hides_internal_operation_name(self) -> None:  # 检查下一轮模型只看到物理反馈而看不到工具标签。
        proposal = {"competing_hypotheses": ["现有证据足够", "现有证据不足"], "evidence_refs": ["initial_evidence"], "uncertainties": ["真实材料参数仍未知"], "proposal_type": "stop", "experiment": {}, "information_request": {}, "provisional_answer": "现有证据只支持暂定结论，停止自动计算。"}  # 构造模型自主停止提案。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结停止提案。
            round_dir = Path(frozen["round_dir"])  # 读取当前轮目录。
            mapping = map_frozen_proposal(round_dir, str(frozen["seal"]["proposal_sha256"]))  # 生成模型不可见映射收据。
            feedback = execute_mapping(round_dir, str(frozen["seal"]["proposal_sha256"]), mapping)  # 执行停止操作并生成公开反馈。
            visible_text = json.dumps(feedback, ensure_ascii=False)  # 序列化下一轮模型可见反馈。
            self.assertEqual(feedback["status"], "analysis_complete")  # 断言自主停止被转换为中性公开状态。
            self.assertNotIn("internal_operation", visible_text)  # 断言反馈没有内部操作字段。
            self.assertNotIn("finish", visible_text)  # 断言反馈没有内部英文操作名称。
            self.assertTrue((round_dir / "execution_audit.json").is_file())  # 断言完整内部审计仍单独保存。
            self.assertTrue((round_dir / "public_feedback.json").is_file())  # 断言脱敏反馈单独保存。

    def test_closed_form_feedback_removes_internal_tool_field(self) -> None:  # 检查理论参照反馈不向模型泄露内部函数标记。
        proposal = experiment_proposal("不改变模型，计算当前理想化中心裂纹的解析参照。", ["理论应力强度因子", "理论能量释放率"])  # 构造理论校核实验。
        with tempfile.TemporaryDirectory() as directory:  # 创建独立临时目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结理论参照提案。
            round_dir = Path(frozen["round_dir"])  # 读取冻结目录。
            mapping = map_frozen_proposal(round_dir, str(frozen["seal"]["proposal_sha256"]))  # 生成隐藏理论参照映射。
            feedback = execute_mapping(round_dir, str(frozen["seal"]["proposal_sha256"]), mapping)  # 执行理论参照并净化公开反馈。
            self.assertNotIn("tool", json.dumps(feedback, ensure_ascii=False))  # 断言公开反馈没有内部工具字段。

    def test_new_runner_only_writes_to_explicit_output_directory(self) -> None:  # 检查隔离运行器没有生产目录写入常量。
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_deepseek_crack_hidden_executor.py").read_text(encoding="utf-8")  # 读取新运行器源码。
        self.assertIn("artifacts/deepseek_crack_hidden_executor", source)  # 断言默认输出落在独立 Artifact 目录。
        self.assertNotIn("docs/", source)  # 断言运行时不会写入论文文档目录。
        self.assertNotIn("experiments/results/latest_deepseek_crack_open_discovery", source)  # 断言不会覆盖旧开放发现结果。


if __name__ == "__main__":  # 仅在测试文件直接运行时启动测试。
    unittest.main()  # 执行全部隔离合同测试。
