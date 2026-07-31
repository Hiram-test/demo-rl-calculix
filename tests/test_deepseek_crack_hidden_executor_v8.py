from __future__ import annotations  # 启用现代类型注解并保持测试环境兼容。

import tempfile  # 为每个材料请求映射测试创建独立冻结目录。
import unittest  # 使用标准库测试框架避免新增测试依赖。
from pathlib import Path  # 管理冻结轮次目录。

from experiments.hidden_executor.contracts import freeze_proposal  # 在隐藏映射前冻结模型请求。
from experiments.hidden_executor.executor_adapter_v8 import FRACTURE_MATERIAL_OPERATION  # 检查断裂参量和材料联合请求映射。
from experiments.hidden_executor.executor_adapter_v8 import execute_mapping  # 执行纯材料请求并检查真实阻塞反馈。
from experiments.hidden_executor.executor_adapter_v8 import map_frozen_proposal  # 测试材料请求优先级。


def request(question: str) -> dict[str, object]:  # 构造满足合同的request_information提案。
    return {"competing_hypotheses": ["当前线弹性模型适用", "材料非线性可能改变工程结论"], "evidence_refs": ["initial_evidence", "previous_rounds"], "uncertainties": ["材料数据未提供"], "proposal_type": "request_information", "information_request": {"question": question}}  # 返回完整信息请求对象。


class HiddenExecutorV8Tests(unittest.TestCase):  # 汇总材料阻塞和断裂数据联合请求合同。
    def test_fracture_and_material_question_preserves_both_purposes(self) -> None:  # 检查同一句话中的可计算K和外部屈服强度不会相互覆盖。
        proposal = request("请提供三档网格的应力强度因子K_I，并提供材料屈服强度和真实应力-塑性应变曲线。")  # 复现真实运行中的联合信息请求。
        with tempfile.TemporaryDirectory() as directory:  # 创建隔离冻结目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结联合请求。
            mapping = map_frozen_proposal(Path(frozen["round_dir"]), str(frozen["seal"]["proposal_sha256"]))  # 调用v8优先映射。
            self.assertEqual(mapping["operation"], FRACTURE_MATERIAL_OPERATION)  # 断言同时保留数值断裂参量和外部材料事实两部分。

    def test_material_only_question_does_not_repeat_fracture_calculation(self) -> None:  # 检查纯屈服强度请求不会因历史证据中的K再次触发数值计算。
        proposal = request("请提供该钢板材料的单轴屈服强度、材料牌号或保守下限值，以评估小范围屈服条件。")  # 构造纯外部材料请求。
        with tempfile.TemporaryDirectory() as directory:  # 创建隔离冻结目录。
            frozen = freeze_proposal(proposal, Path(directory), 1, "0" * 64)  # 冻结材料请求。
            round_dir = Path(frozen["round_dir"])  # 读取当前轮目录。
            proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取冻结提案摘要。
            mapping = map_frozen_proposal(round_dir, proposal_hash)  # 调用v8映射器。
            self.assertEqual(mapping["operation"], "request_material")  # 断言直接映射为真实外部材料阻塞。
            feedback = execute_mapping(round_dir, proposal_hash, mapping)  # 生成模型可见材料请求反馈。
            self.assertEqual(feedback["status"], "information_required")  # 断言缺失材料不会被伪装成计算完成。
            self.assertIn("屈服强度", feedback["observations"]["requested_information"])  # 断言反馈明确列出所需屈服强度。
            self.assertNotIn("rows", feedback["observations"])  # 断言纯材料请求没有重复执行K计算。


if __name__ == "__main__":  # 仅在测试文件直接运行时启动测试。
    unittest.main()  # 执行v8材料请求合同测试。
