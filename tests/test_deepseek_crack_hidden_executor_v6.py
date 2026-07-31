from __future__ import annotations  # 启用现代类型注解并保持测试环境兼容。

import json  # 构造前轮公开证据并检查最终反馈字段。
import tempfile  # 为每个映射和执行测试创建独立目录。
import unittest  # 使用标准库测试框架避免新增测试依赖。
from pathlib import Path  # 管理冻结提案和前轮证据文件。

from experiments.hidden_executor.contracts import freeze_proposal  # 在隐藏映射前冻结测试提案。
from experiments.hidden_executor.executor_adapter_v6 import DISPLACEMENT_K_OPERATION  # 检查位移法K联合操作映射。
from experiments.hidden_executor.executor_adapter_v6 import DISPLACEMENT_MATERIAL_OPERATION  # 检查位移证据与材料请求联合映射。
from experiments.hidden_executor.executor_adapter_v6 import DISPLACEMENT_PROBE_OPERATION  # 检查原始位移数据映射。
from experiments.hidden_executor.executor_adapter_v6 import STRESS_POSTPROCESS_OPERATION  # 检查纯既有数据后处理映射。
from experiments.hidden_executor.executor_adapter_v6 import execute_mapping  # 执行v6专用隐藏路径。
from experiments.hidden_executor.executor_adapter_v6 import map_frozen_proposal  # 测试v6映射优先级和忠实性。


def experiment(change: str, measure: list[str], purpose: str = "区分竞争假设") -> dict[str, object]:  # 构造满足冻结合同的通用实验提案。
    return {"competing_hypotheses": ["当前评价量可以形成稳定工程证据", "当前评价量仍受离散误差影响"], "evidence_refs": ["previous_rounds"], "uncertainties": ["评价量适用范围仍需验证"], "proposal_type": "experiment", "experiment": {"purpose": purpose, "change": change, "hold_fixed": ["几何", "材料", "载荷", "边界"], "measure": measure, "decision_rule": "比较三档网格的相对变化", "stop_condition": "完成本次数据提取或代数计算"}}  # 返回完整受控实验对象。


def information_request(question: str) -> dict[str, object]:  # 构造模型向现有数值环境请求数据的提案。
    return {"competing_hypotheses": ["现有数值解包含所需数据", "所需数据必须由外部补充"], "evidence_refs": ["initial_evidence"], "uncertainties": ["执行环境能力对模型不可见"], "proposal_type": "request_information", "information_request": {"question": question}}  # 返回合法信息请求对象。


def write_prior_stress_feedback(root: Path) -> None:  # 创建模型已经看到的固定距离应力公开证据。
    round_dir = root / "round_01"  # 定义前轮冻结目录。
    round_dir.mkdir(parents=True, exist_ok=True)  # 创建前轮目录。
    rows = [  # 定义三档网格固定距离应力证据。
        {"nx": 40, "h_local_mm": 5.0, "samples": {"distance_2.5_mm_mean_sigma_y_mpa": 210.0}},  # 保存粗网格样本。
        {"nx": 60, "h_local_mm": 3.3333333333, "samples": {"distance_2.5_mm_mean_sigma_y_mpa": 220.0}},  # 保存中网格样本。
        {"nx": 80, "h_local_mm": 2.5, "samples": {"distance_2.5_mm_mean_sigma_y_mpa": 225.0}},  # 保存细网格样本。
    ]  # 完成前轮结果数组。
    feedback = {"status": "completed", "executed_change": "在固定物理距离提取法向应力", "actual_parameters": {"distances_mm": [2.5]}, "observations": {"rows": rows}, "limitations": []}  # 组织模型可见前轮反馈。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存前轮公开证据文件。


class HiddenExecutorV6Tests(unittest.TestCase):  # 汇总位移提取、纯后处理和严格加密识别合同。
    def test_displacement_k_experiment_is_not_mapped_to_stress_probe(self) -> None:  # 检查裂纹面位移法K实验不会被固定应力关键词抢占。
        proposal = experiment("不改变模型，在裂纹面上距裂尖r=2.5 mm处提取上下表面竖向位移差。", ["三个网格的裂纹面张开位移", "按裂尖位移渐近式计算应力强度因子KI"])  # 复现真实运行中的位移法提案。
        with tempfile.TemporaryDirectory() as directory:  # 创建隔离冻结目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 在映射前冻结提案。
            round_dir = Path(frozen["round_dir"])  # 读取当前轮目录。
            proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取冻结提案摘要。
            mapping = map_frozen_proposal(round_dir, proposal_hash)  # 调用v6隐藏映射器。
            self.assertEqual(mapping["operation"], DISPLACEMENT_K_OPERATION)  # 断言映射为真实位移提取与K换算。
            feedback = execute_mapping(round_dir, proposal_hash, mapping)  # 执行真实三档网格位移提取。
            self.assertEqual(feedback["status"], "completed")  # 断言联合实验完整完成。
            self.assertIn("mean_half_opening_mm", feedback["observations"]["rows"][0])  # 断言反馈包含真实裂纹面半张开位移。
            self.assertIn("stress_intensity_from_opening_mpa_sqrt_mm", feedback["observations"]["rows"][0])  # 断言反馈包含位移法K数值。
            self.assertNotIn("samples", feedback["observations"]["rows"][0])  # 断言位移实验没有被替换为固定应力采样。

    def test_plain_displacement_request_returns_node_coordinates_and_displacements(self) -> None:  # 检查数据请求能够从现有解中返回原始裂纹面节点位移。
        proposal = information_request("请提供三个已有网格在裂纹面距裂尖2.5 mm最近节点的坐标、上下面节点u_y和张开位移。")  # 构造不要求派生K的位移请求。
        with tempfile.TemporaryDirectory() as directory:  # 创建隔离冻结目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结信息请求。
            round_dir = Path(frozen["round_dir"])  # 读取当前轮目录。
            proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取提案摘要。
            mapping = map_frozen_proposal(round_dir, proposal_hash)  # 执行隐藏映射。
            self.assertEqual(mapping["operation"], DISPLACEMENT_PROBE_OPERATION)  # 断言映射到原始位移提取。
            feedback = execute_mapping(round_dir, proposal_hash, mapping)  # 执行真实位移提取。
            sample = feedback["observations"]["rows"][0]["tip_samples"][0]  # 读取第一个裂尖节点样本。
            self.assertIn("upper_uy_mm", sample)  # 断言返回上裂纹面竖向位移。
            self.assertIn("lower_uy_mm", sample)  # 断言返回下裂纹面竖向位移。
            self.assertIn("actual_x_mm", sample)  # 断言返回实际节点坐标以审计距离修复。

    def test_material_and_displacement_request_preserves_both_purposes(self) -> None:  # 检查联合请求不会丢失可计算位移或外部材料部分。
        proposal = information_request("请提取裂纹面节点位移，同时提供材料屈服强度和真实应力-塑性应变曲线。")  # 构造可计算证据与外部事实联合请求。
        with tempfile.TemporaryDirectory() as directory:  # 创建隔离冻结目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结联合请求。
            round_dir = Path(frozen["round_dir"])  # 读取当前轮目录。
            proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取提案摘要。
            mapping = map_frozen_proposal(round_dir, proposal_hash)  # 映射完整联合目的。
            self.assertEqual(mapping["operation"], DISPLACEMENT_MATERIAL_OPERATION)  # 断言映射保留位移和材料两部分。
            feedback = execute_mapping(round_dir, proposal_hash, mapping)  # 提取位移并记录外部阻塞。
            self.assertEqual(feedback["status"], "information_required")  # 断言材料事实仍被诚实标记为外部请求。
            self.assertTrue(feedback["observations"]["rows"])  # 断言可计算位移证据同时返回。
            self.assertIn("requested_information", feedback["observations"])  # 断言外部材料请求没有被丢弃。

    def test_existing_stress_formula_is_pure_postprocess_not_refinement(self) -> None:  # 检查基于已有应力的K公式不会触发新的有限元加密。
        proposal = experiment("对nx=40、60、80已有结果，仅基于已有数据按KI=σ_yy·√(2πr)计算，r=2.5 mm，不进行新的有限元计算。", ["三个KI估计值", "相对变化百分比"])  # 复现真实运行中的纯代数后处理提案。
        with tempfile.TemporaryDirectory() as directory:  # 创建包含前轮公开证据的实验目录。
            root = Path(directory)  # 读取临时根目录。
            write_prior_stress_feedback(root)  # 写入模型已经看到的固定距离应力结果。
            frozen = freeze_proposal(proposal, root, 2, "1" * 64)  # 把后处理提案冻结为第二轮。
            round_dir = Path(frozen["round_dir"])  # 读取当前轮目录。
            proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取提案摘要。
            mapping = map_frozen_proposal(round_dir, proposal_hash)  # 调用v6映射器。
            self.assertEqual(mapping["operation"], STRESS_POSTPROCESS_OPERATION)  # 断言映射为纯后处理且不映射为refine。
            feedback = execute_mapping(round_dir, proposal_hash, mapping)  # 使用前轮公开数据执行代数计算。
            self.assertIn("未运行新模型", feedback["executed_change"])  # 断言公开反馈明确没有新求解。
            self.assertEqual(len(feedback["observations"]["rows"]), 3)  # 断言三档网格均获得K估计值。
            self.assertEqual(feedback["actual_parameters"]["source"], "prior public_feedback only")  # 断言数据来源仅为前轮公开证据。

    def test_nx_references_without_resolution_change_cannot_map_to_refine(self) -> None:  # 检查仅引用已有网格编号时普通加密映射被禁止。
        proposal = experiment("对nx=40、60、80三个已有结果计算一个无量纲比值，不进行新的仿真。", ["三个比值", "相对离散度"])  # 构造不改变网格的未知后处理实验。
        with tempfile.TemporaryDirectory() as directory:  # 创建隔离冻结目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结只引用nx的提案。
            mapping = map_frozen_proposal(Path(frozen["round_dir"]), str(frozen["seal"]["proposal_sha256"]))  # 调用最终映射器。
            self.assertEqual(mapping["operation"], "unsupported")  # 断言执行器拒绝未知后处理且不会错误重跑粗网格。


if __name__ == "__main__":  # 仅在测试文件直接运行时启动测试。
    unittest.main()  # 执行v6忠实映射合同测试。
