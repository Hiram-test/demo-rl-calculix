from __future__ import annotations  # 启用现代类型注解并保持测试环境兼容。

import json  # 构造冻结提案和读取映射收据。
import tempfile  # 为每个合同测试创建独立临时目录。
import unittest  # 使用标准库执行无额外依赖的回归测试。
from pathlib import Path  # 管理临时冻结目录和状态文件。

from experiments.hidden_executor.contracts import freeze_proposal  # 复用现有提案冻结合同。
from experiments.shaft_grid.backend import MeshConfig  # 导入网格配置对象。
from experiments.shaft_grid.backend import SolveResult  # 导入可构造的求解结果对象。
from experiments.shaft_grid.backend import VISIBLE_FORCE_N  # 导入可见轴向拉力。
from experiments.shaft_grid.backend import VISIBLE_TORQUE_NMM  # 导入可见扭矩。
from experiments.shaft_grid.backend import _boundary_displacement  # 导入连续体位移场用于无求解器测试。
from experiments.shaft_grid.backend import _generate_nodes  # 导入结构化圆柱节点生成器。
from experiments.shaft_grid.backend import analytical_optimum  # 导入隐藏解析最优方向。
from experiments.shaft_grid.backend import angle_sweep  # 导入方向扫描后处理。
from experiments.shaft_grid.backend import solve  # 导入无求解器依赖的烟测后端。
from experiments.shaft_grid.experiment import USER_QUESTION  # 导入模型可见工程师问题。
from experiments.shaft_grid.experiment import map_frozen_proposal  # 导入隐藏确定性映射器。


def _synthetic_result(config: MeshConfig) -> SolveResult:  # 用连续体节点位移构造无需求解器的单元测试结果。
    nodes = _generate_nodes(config)  # 生成当前方向和密度的圆柱节点。
    displacements = {node_id: _boundary_displacement(point, VISIBLE_FORCE_N, VISIBLE_TORQUE_NMM) for node_id, point in nodes.items()}  # 在全部节点赋予连续体拉扭位移场。
    return SolveResult(config=config, nodes=nodes, displacements=displacements, solver="synthetic patch field", cache_dir="")  # 返回可供后处理测试的结果对象。


def _proposal(change: str, measure: list[str]) -> dict[str, object]:  # 构造满足PR十八合同的最小测试提案。
    return {"competing_hypotheses": ["当前差异来自网格方向", "当前差异来自标线结果读取方式"], "evidence_refs": ["initial_calculations"], "uncertainties": ["推荐角度是否随网格改变"], "proposal_type": "experiment", "experiment": {"purpose": "区分两个解释", "change": change, "hold_fixed": ["几何", "材料", "载荷"], "measure": measure, "decision_rule": "比较推荐角度变化", "stop_condition": "两种解释可以区分"}, "information_request": {}, "provisional_answer": "暂不下结论"}  # 返回完整冻结提案对象。


class ShaftGridContractsTest(unittest.TestCase):  # 汇总圆轴网格发现烟测的静态与数值合同。
    def test_user_question_hides_competition_and_answer(self) -> None:  # 检查工程师问题没有泄露竞赛来源或固定答案。
        lowered = USER_QUESTION.lower()  # 统一转换为小写便于检查英文标记。
        self.assertNotIn("周培源", USER_QUESTION)  # 禁止泄露竞赛名称。
        self.assertNotIn("22.5", USER_QUESTION)  # 禁止泄露原题特殊最优角。
        self.assertNotIn("pso", lowered)  # 禁止在用户问题中指定优化算法。
        self.assertNotIn("粒子群", USER_QUESTION)  # 禁止在用户问题中指定粒子群路线。
        self.assertIn("可以直接采用的角度", USER_QUESTION)  # 确认问题仍然要求实际工程结果。

    def test_smoke_backend_solves_without_external_solver(self) -> None:  # 检查当前阶段能够直接生成离散场并保存清单。
        with tempfile.TemporaryDirectory() as temporary:  # 创建独立临时目录。
            result = solve(MeshConfig(16, 3, 8, 0.0), Path(temporary))  # 运行轻量烟测后端。
            self.assertEqual(result.solver, "analytical surface-grid smoke backend")  # 确认没有调用外部有限元求解器。
            self.assertGreater(len(result.nodes), 0)  # 确认离散网格节点已经生成。
            self.assertEqual(len(result.nodes), len(result.displacements))  # 确认每个节点都有位移采样。
            self.assertTrue((Path(result.cache_dir) / "manifest.json").exists())  # 确认轻量审计清单已经写入。

    def test_interpolation_recovers_hidden_continuum_optimum(self) -> None:  # 检查精确端点插值能够恢复连续体最优方向。
        result = _synthetic_result(MeshConfig(64, 8, 32, 15.0))  # 构造一组已细化的斜交表面网格连续体位移场。
        scan = angle_sweep(result, "surface_interpolation", 0.25)  # 使用精确端点插值扫描方向。
        truth = analytical_optimum(VISIBLE_FORCE_N, VISIBLE_TORQUE_NMM)  # 计算当前非特殊载荷比隐藏真值。
        self.assertLess(abs(float(scan["best_beta_deg"]) - truth["beta_deg"]), 0.6)  # 要求插值结果在零点六度内命中解析最优值。

    def test_orientation_comparison_runs_for_both_meshes(self) -> None:  # 检查正交和斜交网格都能完成方向扫描而不预设差异必须存在。
        orthogonal = angle_sweep(_synthetic_result(MeshConfig(32, 5, 16, 0.0)), "nearest_node", 0.5)  # 扫描正交表面网格。
        skewed = angle_sweep(_synthetic_result(MeshConfig(32, 5, 16, 20.0)), "nearest_node", 0.5)  # 扫描同密度斜交表面网格。
        self.assertGreaterEqual(float(orthogonal["best_beta_deg"]), 0.0)  # 确认正交网格返回合法方向下限。
        self.assertLessEqual(float(orthogonal["best_beta_deg"]), 60.0)  # 确认正交网格返回合法方向上限。
        self.assertGreaterEqual(float(skewed["best_beta_deg"]), 0.0)  # 确认斜交网格返回合法方向下限。
        self.assertLessEqual(float(skewed["best_beta_deg"]), 60.0)  # 确认斜交网格返回合法方向上限。
        self.assertGreater(float(orthogonal["best_delta_beta_deg"]), 0.0)  # 确认正交网格产生非零方向响应。
        self.assertGreater(float(skewed["best_delta_beta_deg"]), 0.0)  # 确认斜交网格产生非零方向响应。

    def test_hidden_mapping_recognizes_orientation_without_mutation(self) -> None:  # 检查网格方向提案被忠实映射且冻结文件不被改写。
        with tempfile.TemporaryDirectory() as temporary:  # 创建独立临时冻结目录。
            output_dir = Path(temporary)  # 转换为路径对象。
            proposal = _proposal("保持单元数量不变，把整套网格分别转成正交和斜交排列", ["推荐角度", "最大角度变化"])  # 构造方向对照提案。
            frozen = freeze_proposal(proposal, output_dir, 1, "0" * 64)  # 在映射前冻结提案并生成哈希链。
            original = (Path(frozen["round_dir"]) / "proposal.json").read_text(encoding="utf-8")  # 保存映射前原始提案文本。
            mapping = map_frozen_proposal(Path(frozen["round_dir"]), str(frozen["seal"]["proposal_sha256"]))  # 执行隐藏确定性映射。
            after = (Path(frozen["round_dir"]) / "proposal.json").read_text(encoding="utf-8")  # 读取映射后提案文本。
            self.assertEqual(mapping["operation"], "orientation_compare")  # 确认方向提案映射到等规模网格方向对照。
            self.assertEqual(original, after)  # 确认隐藏映射未改写冻结提案。
            receipt = json.loads((Path(frozen["round_dir"]) / "mapping_receipt.json").read_text(encoding="utf-8"))  # 读取映射审计收据。
            self.assertTrue(receipt["proposal_unchanged"])  # 确认收据明确记录提案未变。

    def test_hidden_mapping_prioritizes_extraction_when_requested(self) -> None:  # 检查模型提出端点插值时不会被替换成预设加密路线。
        with tempfile.TemporaryDirectory() as temporary:  # 创建独立临时冻结目录。
            output_dir = Path(temporary)  # 转换为路径对象。
            proposal = _proposal("网格和载荷不变，比较最近节点与形函数插值读取标线两端位移", ["两种方法的推荐角度"])  # 构造结果提取对照提案。
            frozen = freeze_proposal(proposal, output_dir, 1, "0" * 64)  # 在映射前冻结提案。
            mapping = map_frozen_proposal(Path(frozen["round_dir"]), str(frozen["seal"]["proposal_sha256"]))  # 执行隐藏确定性映射。
            self.assertEqual(mapping["operation"], "extraction_compare")  # 确认提取提案得到忠实支持。


if __name__ == "__main__":  # 仅在直接运行测试文件时启动unittest。
    unittest.main()  # 执行全部合同和数值回归测试。
