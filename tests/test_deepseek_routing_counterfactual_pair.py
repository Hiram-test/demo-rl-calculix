from __future__ import annotations  # 允许测试使用现代类型注解并保持 Python 3.11 兼容。

import ast  # 在不触发有限元求解或网络调用的情况下检查脚本结构。
import importlib.util  # 从明确文件路径加载反事实实验模块以测试纯函数。
import unittest  # 使用标准库测试框架执行离线合同检查。
from pathlib import Path  # 定位脚本、工作流与实验文档。
from types import ModuleType  # 标注动态加载后的脚本模块类型。

ROOT = Path(__file__).resolve().parents[1]  # 定位当前仓库根目录。
SCRIPT_PATH = ROOT / "scripts" / "run_deepseek_routing_counterfactual_pair.py"  # 定位独立反事实运行脚本。
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "deepseek-routing-counterfactual-pair.yml"  # 定位只监听一次性标记的付费工作流。
DOC_PATH = ROOT / "docs" / "DEEPSEEK_ROUTING_COUNTERFACTUAL_PAIR.md"  # 定位实验协议文档。
TRIGGER_PATH = ROOT / "experiments" / "triggers" / "DEEPSEEK_ROUTING_COUNTERFACTUAL_PAIR_ONCE.md"  # 定位首次 push 使用且不再修改的一次性标记。


def _load_module() -> ModuleType:  # 加载脚本模块但不执行其 main 函数或真实 API 调用。
    spec = importlib.util.spec_from_file_location("counterfactual_pair_module", SCRIPT_PATH)  # 为目标脚本创建独立导入规范。
    if spec is None or spec.loader is None:  # 检查 Python 是否成功创建脚本加载器。
        raise AssertionError("unable to create module spec")  # 在加载器缺失时给出明确测试失败。
    module = importlib.util.module_from_spec(spec)  # 根据规范创建未执行的模块对象。
    spec.loader.exec_module(module)  # 执行常量与函数定义但不会进入受保护的 main 入口。
    return module  # 返回可测试其纯函数和常量的模块。


class DeepSeekRoutingCounterfactualPairTests(unittest.TestCase):  # 定义独立双样本反事实实验的离线合同测试。
    @classmethod  # 让模块只加载一次以减少测试开销。
    def setUpClass(cls) -> None:  # 为本测试类准备共享脚本模块。
        cls.module = _load_module()  # 加载模块定义但不执行真实有限元场景或 API 请求。

    def test_script_compiles(self) -> None:  # 验证独立运行脚本具有合法 Python 语法。
        ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))  # 解析脚本并在语法错误时直接失败。

    def test_request_budget_is_exactly_two_without_sdk_retries(self) -> None:  # 验证费用控制硬约束不会退化为多轮调用。
        source = SCRIPT_PATH.read_text(encoding="utf-8")  # 读取完整脚本源代码以检查客户端与调用结构。
        self.assertEqual(2, self.module.MAX_HTTP_REQUESTS)  # 确认实验总请求硬上限固定为两个。
        self.assertIn("max_retries=0", source)  # 确认 OpenAI SDK 的隐式重试被明确关闭。
        self.assertNotIn("for attempt in", source)  # 确认没有模型结构修复重试循环。
        self.assertEqual(1, source.count("client.chat.completions.create("))  # 确认所有请求只经过一个受硬上限保护的调用点。

    def test_common_prefix_is_identical_and_has_no_oracle(self) -> None:  # 验证两例共享同一协议且隐藏答案未混入提示词。
        first = self.module._common_messages()  # 构造第一份公共消息前缀。
        second = self.module._common_messages()  # 再次独立构造公共消息前缀。
        self.assertEqual(first, second)  # 确认两次构造逐字段一致以支持服务端前缀缓存。
        serialized = self.module._canonical_json(first).lower()  # 序列化真正发送的公共前缀。
        self.assertNotIn("expected_route", serialized)  # 确认公共前缀不包含隐藏期望路线字段。
        self.assertNotIn("oracle", serialized)  # 确认公共前缀不包含评分答案标记。
        self.assertNotIn(self.module.MISLEADING_SAMPLE_ID, serialized)  # 确认公共前缀不包含第一个匿名样本编号。
        self.assertNotIn(self.module.HIDDEN_SAMPLE_ID, serialized)  # 确认公共前缀不包含第二个匿名样本编号。

    def test_lexical_inversion_contract_is_frozen(self) -> None:  # 验证一例含目标词而另一例的用户文本完全隐藏目标词。
        self.assertIn("裂纹", self.module.MISLEADING_USER_TEXT)  # 确认连续拓扑样本保留故意误导的目标词。
        lowered = self.module.HIDDEN_USER_TEXT.lower()  # 统一隐藏样本文本大小写以执行禁词检查。
        leaked = [term for term in self.module.HIDDEN_FORBIDDEN_TERMS if term.lower() in lowered]  # 搜索任何目标类别或专用量泄漏。
        self.assertEqual([], leaked)  # 确认内部不连续样本的用户文本不点名答案。

    def test_posthoc_scoring_is_not_part_of_common_protocol(self) -> None:  # 验证评分映射只在回答冻结后由本地纯函数使用。
        protocol = self.module._canonical_json(self.module.COMMON_PROTOCOL)  # 序列化实际发送的共用协议。
        for sample_id in self.module.POSTHOC_EXPECTED_ROUTES:  # 遍历仅供事后评分的两个匿名编号。
            self.assertNotIn(sample_id, protocol)  # 确认共用协议未包含任何评分映射键。
        fake_result = {"sample_id": self.module.MISLEADING_SAMPLE_ID, "decision": {"route_id": "R1", "selected_tool_id": "T3", "evidence_refs": ["model_facts.local_topology.connectivity_discontinuity", "mesh_history[2].fixed_15mm_mpa"]}, "validation_errors": []}  # 构造不需要网络的最小正确回答。
        score = self.module._score_result(fake_result)  # 执行只在冻结后使用的确定性评分函数。
        self.assertTrue(score["route_match"])  # 确认隐藏路线映射能识别预期选择。
        self.assertTrue(score["cites_topology"])  # 确认证据路径覆盖检查能识别拓扑引用。
        self.assertTrue(score["cites_mesh_history"])  # 确认证据路径覆盖检查能识别网格历史引用。

    def test_workflow_is_one_shot_ds_and_artifact_only(self) -> None:  # 验证付费实验只由一次性标记 push 触发且不会因结果回写形成循环。
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")  # 读取独立工作流全文。
        self.assertIn("\n  push:", workflow)  # 确认实验分支首次加入标记时能够直接触发。
        self.assertNotIn("workflow_dispatch", workflow)  # 确认界面不能手动重复启动付费实验。
        self.assertIn("experiments/triggers/DEEPSEEK_ROUTING_COUNTERFACTUAL_PAIR_ONCE.md", workflow)  # 确认唯一监听路径是一次性标记。
        self.assertTrue(TRIGGER_PATH.is_file())  # 确认本次提交实际包含所监听的一次性标记。
        self.assertIn("environment: ds", workflow)  # 确认 DeepSeek 凭据来自仓库长期化 ds Environment。
        self.assertIn("contents: read", workflow)  # 确认工作流没有把结果写回仓库的权限。
        self.assertIn("actions/upload-artifact@v4", workflow)  # 确认原始回答和评分通过 artifact 冻结。
        self.assertNotIn("git push", workflow)  # 确认工作流不会因结果提交再次触发自己。

    def test_protocol_document_records_independence(self) -> None:  # 验证文档明确说明该测试与既有论文流程相互独立。
        text = DOC_PATH.read_text(encoding="utf-8")  # 读取反事实实验协议文档。
        self.assertIn("独立", text)  # 确认文档声明实验独立性。
        self.assertIn("两次", text)  # 确认文档声明固定请求次数。
        self.assertIn("不回写", text)  # 确认文档声明 artifact-only 的无循环策略。

    def test_every_nonblank_python_line_has_chinese_comment_marker(self) -> None:  # 验证新写 Python 代码逐行满足仓库全局注释规则。
        for path in (SCRIPT_PATH, Path(__file__).resolve()):  # 依次检查运行脚本和测试脚本自身。
            lines = path.read_text(encoding="utf-8").splitlines()  # 读取当前 Python 文件的全部文本行。
            missing = [index for index, line in enumerate(lines, start=1) if line.strip() and "#" not in line]  # 找出没有注释标记的非空代码行。
            self.assertEqual([], missing, f"{path.name} uncommented lines: {missing}")  # 报告所有违反逐行注释约束的行号。


if __name__ == "__main__":  # 允许从命令行直接运行本合同测试文件。
    unittest.main()  # 启动标准库测试运行器且不调用 DeepSeek。
