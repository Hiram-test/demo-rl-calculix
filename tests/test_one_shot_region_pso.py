from __future__ import annotations  # 启用延迟类型注解。
import copy  # 提供测试载荷的隔离复制。
import json  # 读取示例输入载荷。
import math  # 验证低维解码与资源中性条件。
import sys  # 配置测试源码导入路径。
import unittest  # 提供标准库测试框架。
from pathlib import Path  # 定位实现根目录与示例文件。
ROOT = Path(__file__).resolve().parents[1]  # 定位测试项目根目录。
SRC = ROOT / "src"  # 定位Python源码目录。
if str(SRC) not in sys.path:  # 检查源码目录是否已在模块搜索路径中。
    sys.path.insert(0, str(SRC))  # 将源码目录插入最高优先级。
from engineering_agent.one_shot_region_pso import algorithm_from_mapping  # 导入严格一次性算法构造函数。
class OneShotRegionPSOTests(unittest.TestCase):  # 定义完整状态机与数值行为测试集。
    def setUp(self) -> None:  # 为每个测试加载独立输入载荷。
        case_path = ROOT / "examples" / "one_shot_region_pso_case.json"  # 定位五区域示例输入。
        self.payload = json.loads(case_path.read_text(encoding="utf-8"))  # 读取示例载荷。
    def _algorithm(self, generations: int = 1):  # 构造指定一代或两代PSO的独立算法实例。
        payload = copy.deepcopy(self.payload)  # 复制输入以避免测试间状态污染。
        payload.setdefault("config", {})["pso_generations"] = generations  # 设置严格PSO代数。
        return algorithm_from_mapping(payload)  # 返回未运行的算法实例。
    def test_exact_one_generation_state_machine_counts(self) -> None:  # 验证一代校准下所有一次性调用计数。
        result = self._algorithm(1).run()  # 执行一代微型PSO流程。
        self.assertEqual(result.trace.global_delegations, 1)  # 验证全局委派仅一次。
        self.assertEqual(result.trace.regional_estimates, 5)  # 验证五个区域各估计一次。
        self.assertEqual(result.trace.communication_rounds, 1)  # 验证区域通信仅一轮。
        self.assertEqual(result.trace.regional_adjustments, 5)  # 验证五个区域各调整一次。
        self.assertEqual(result.trace.global_base_reviews, 1)  # 验证PSO前总智能体仅评估一次母粒子。
        self.assertEqual(result.trace.pso_coordinate_dimension, 2)  # 验证PSO未开放五个区域尺寸自由度。
        self.assertEqual(result.trace.pso_generations, 1)  # 验证只执行一代PSO更新。
        self.assertEqual(result.trace.pso_surrogate_evaluations, 9)  # 验证母粒子缓存后PSO仅新增九次代数评价。
        self.assertEqual(result.trace.total_surrogate_evaluations, 10)  # 验证母粒子加PSO总评价次数为十次。
        self.assertEqual(result.trace.high_fidelity_certifications, 0)  # 验证无认证器时不调用高保真求解。
    def test_two_generations_keep_fixed_micro_pso_budget(self) -> None:  # 验证两代校准仍保持固定五粒子二维预算。
        result = self._algorithm(2).run()  # 执行两代微型PSO流程。
        self.assertEqual(result.trace.pso_generations, 2)  # 验证两代配置生效。
        self.assertEqual(result.trace.pso_surrogate_evaluations, 14)  # 验证PSO只新增十四次代数评价。
        self.assertEqual(result.trace.total_surrogate_evaluations, 15)  # 验证总评价次数固定为十五次。
        self.assertEqual(len(result.pso_history), 15)  # 验证轨迹包含一个缓存母粒子与十四次新增评价。
    def test_every_region_estimates_delegated_size_and_adjusts_once(self) -> None:  # 验证区域智能体没有二次探测或二次修正。
        result = self._algorithm(1).run()  # 执行一次完整流程。
        message_ids = [message.region_id for message in result.regional_messages]  # 收集消息区域标识。
        decision_ids = [decision.region_id for decision in result.regional_decisions]  # 收集决策区域标识。
        self.assertEqual(len(message_ids), len(set(message_ids)))  # 验证每个区域只发送一条消息。
        self.assertEqual(len(decision_ids), len(set(decision_ids)))  # 验证每个区域只形成一个决策。
        for message in result.regional_messages:  # 遍历全部区域消息。
            self.assertAlmostEqual(message.estimate.size, result.delegated_sizes[message.region_id])  # 验证唯一估计发生在委派尺寸上。
        for decision in result.regional_decisions:  # 遍历全部区域决策。
            self.assertGreater(decision.adjusted_size, 0.0)  # 验证调整后尺寸保持物理可行。
    def test_selected_particle_improves_base_constraint_balance(self) -> None:  # 验证微型PSO只做误差资源残差校准。
        result = self._algorithm(2).run()  # 执行两代校准流程。
        base = result.base_particle.evaluation  # 读取总智能体评估后的母粒子。
        selected = result.selected_particle.evaluation  # 读取微型PSO选中的终局粒子。
        self.assertLessEqual(selected.fitness, base.fitness)  # 验证选中粒子不劣于母粒子。
        self.assertTrue(selected.error_feasible)  # 验证选中粒子满足误差上限。
        self.assertTrue(selected.resource_feasible)  # 验证选中粒子满足资源预算。
        self.assertTrue(selected.quality_feasible)  # 验证选中粒子满足相邻尺寸过渡。
    def test_selected_sizes_are_decoded_only_from_two_coordinates(self) -> None:  # 验证PSO没有逐区域自由搜索。
        algorithm = self._algorithm(2)  # 构造两代校准算法。
        result = algorithm.run()  # 执行完整流程。
        scale = result.selected_particle.scale_coordinate  # 读取整体尺度坐标。
        transfer = result.selected_particle.transfer_coordinate  # 读取资源转移坐标。
        base_sizes = result.base_particle.evaluation.sizes  # 读取唯一母粒子尺寸向量。
        for region in algorithm.regions:  # 遍历全部区域。
            expected = base_sizes[region.region_id] * math.exp(scale + transfer * result.transfer_direction[region.region_id])  # 用二维坐标重建该区域尺寸。
            expected = min(max(expected, region.minimum_size), region.maximum_size)  # 应用与实现相同的边界投影。
            actual = result.selected_particle.evaluation.sizes[region.region_id]  # 读取PSO终局区域尺寸。
            self.assertAlmostEqual(actual, expected, places=12)  # 验证终局尺寸完全由两个坐标解码。
    def test_transfer_direction_is_first_order_resource_neutral(self) -> None:  # 验证资源转移方向不混入整体尺度自由度。
        algorithm = self._algorithm(1)  # 构造一代校准算法。
        result = algorithm.run()  # 执行完整流程。
        weighted_sum = 0.0  # 初始化资源一阶变化加权和。
        for region in algorithm.regions:  # 遍历全部区域。
            estimate = algorithm.surrogate.predict_region(region.region_id, result.base_particle.evaluation.sizes[region.region_id])  # 计算母粒子处资源灵敏度。
            weight = estimate.resource * estimate.resource_log_sensitivity * region.confidence  # 构造实现采用的资源中性权重。
            weighted_sum += weight * result.transfer_direction[region.region_id]  # 累加资源方向一阶变化。
        self.assertAlmostEqual(weighted_sum, 0.0, places=10)  # 验证资源转移方向与整体尺度方向正交。
    def test_high_fidelity_certifier_is_called_exactly_once(self) -> None:  # 验证真实求解器只认证PSO选中的一个终局粒子。
        calls: list[dict[str, float]] = []  # 初始化认证调用记录。
        def certifier(sizes):  # 定义计数型高保真认证器。
            calls.append(dict(sizes))  # 记录被认证的完整尺寸粒子。
            return {"status": "certified", "solver_calls": 1}  # 返回模拟认证证书。
        result = self._algorithm(1).run(certifier=certifier)  # 执行带终局认证的流程。
        self.assertEqual(len(calls), 1)  # 验证认证器只被调用一次。
        self.assertEqual(result.trace.high_fidelity_certifications, 1)  # 验证轨迹记录一次高保真调用。
        self.assertEqual(calls[0], result.selected_particle.evaluation.sizes)  # 验证被认证对象正是PSO终局粒子。
    def test_invalid_generation_count_is_rejected(self) -> None:  # 验证长程PSO配置被硬性拒绝。
        payload = copy.deepcopy(self.payload)  # 复制输入载荷。
        payload.setdefault("config", {})["pso_generations"] = 3  # 尝试开放第三代PSO。
        with self.assertRaises(ValueError):  # 断言构造阶段必须失败。
            algorithm_from_mapping(payload)  # 触发一到两代限制验证。
    def test_same_orchestrator_cannot_run_twice(self) -> None:  # 验证一次性编排器不能被重复使用。
        algorithm = self._algorithm(1)  # 构造一次性算法实例。
        algorithm.run()  # 执行第一次合法运行。
        with self.assertRaises(RuntimeError):  # 断言第二次运行必须失败。
            algorithm.run()  # 尝试破坏一次性状态机。
if __name__ == "__main__":  # 检查测试文件是否直接执行。
    unittest.main()  # 启动标准库测试运行器。
