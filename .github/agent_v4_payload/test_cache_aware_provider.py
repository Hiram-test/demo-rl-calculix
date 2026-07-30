# 启用延迟类型注解，避免测试辅助类的前向引用问题。
from __future__ import annotations

# 导入 JSON 模块，用于构造模型响应并读取缓存收据。
import json
# 导入系统模块，用于临时替换 openai SDK 模块。
import sys
# 导入临时目录模块，保证测试不会污染仓库工件。
import tempfile
# 导入简单命名空间，用于构造最小 SDK 响应对象。
import types
# 导入单元测试框架。
import unittest
# 导入路径类型，用于定位 case 文件和测试输出。
from pathlib import Path
# 导入字典补丁工具，用于注入不联网的假 OpenAI 模块。
from unittest.mock import patch

# 导入需要验证的缓存感知提供器。
from bridge_agent.cache_aware_provider import DeepSeekProvider
# 导入 case 合同，用于构造区域多样性测试上下文。
from bridge_agent.contracts import CaseDefinition
# 导入动态区域 Skill，用于验证历史失败门已前移到执行合同。
from bridge_agent.region_skills import propose_dynamic_regions
# 导入运行上下文，用于添加最小 partition evidence。
from bridge_agent.runtime import RunContext


# 定位展开 payload 的根目录。
ROOT = Path(__file__).resolve().parents[1]


# 定义假 completions 端点，记录消息并顺序返回预置响应。
class _FakeCompletions:
    """模拟 client.chat.completions，仅执行内存记录。"""

    # 初始化预置响应队列。
    def __init__(self, responses: list[object]) -> None:
        # 保存待返回响应的可变副本。
        self.responses = list(responses)
        # 初始化收到的请求关键字参数列表。
        self.requests: list[dict[str, object]] = []

    # 模拟 create 方法。
    def create(self, **kwargs: object) -> object:
        # 保存请求快照，用于验证消息前缀增长。
        self.requests.append(dict(kwargs))
        # 没有预置响应时立即失败，防止测试误发额外请求。
        if not self.responses:
            # 抛出明确错误，指出提供器超过预期请求数。
            raise AssertionError("fake DeepSeek response queue is empty")
        # 返回并移除队首响应。
        return self.responses.pop(0)


# 定义假 OpenAI 客户端，提供与 SDK 相同的 chat.completions 属性链。
class _FakeClient:
    """承载假 completions 端点的最小客户端。"""

    # 初始化客户端及其预置响应。
    def __init__(self, responses: list[object]) -> None:
        # 创建可记录请求的 completions 端点。
        completions = _FakeCompletions(responses)
        # 用命名空间构造 chat.completions 属性链。
        self.chat = types.SimpleNamespace(completions=completions)


# 定义假 DeepSeek 成功响应构造器。
def _response(*, hit_tokens: int, miss_tokens: int, content: str | None = None) -> object:
    """构造含官方缓存 usage 字段的 OpenAI 兼容响应。"""
    # 构造符合 Agent 决策合同的默认 JSON。
    decision = {"engineer_facing_summary": "继续获取最小必要证据。", "current_judgment": "先检查构件事实。", "main_uncertainty": "当前构件证据是否完整。", "decision": {"type": "call_skill", "skill": "inspect_component", "arguments": {}, "why": "建立后续判断所需的事实基线。", "expected_result": "生成模型摘要。"}}
    # 优先使用显式正文，否则编码默认决策。
    response_text = content if content is not None else json.dumps(decision, ensure_ascii=False)
    # 构造 completion token 细分。
    completion_details = types.SimpleNamespace(reasoning_tokens=7)
    # 构造包含缓存命中和未命中 token 的 usage。
    usage = types.SimpleNamespace(prompt_tokens=hit_tokens + miss_tokens, completion_tokens=20, total_tokens=hit_tokens + miss_tokens + 20, prompt_cache_hit_tokens=hit_tokens, prompt_cache_miss_tokens=miss_tokens, completion_tokens_details=completion_details)
    # 构造消息对象。
    message = types.SimpleNamespace(content=response_text)
    # 构造首个 choice。
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    # 返回完整响应对象。
    return types.SimpleNamespace(model="deepseek-v4-pro", usage=usage, choices=[choice])


# 定义缓存提供器与区域合同测试集合。
class CacheAwareProviderTests(unittest.TestCase):
    """验证跨进程增长历史、重试关闭、usage 审计和区域多样性门。"""

    # 验证两个隔离 case 进程仍使用同一条增长消息历史。
    def test_cross_process_history_reuses_exact_prefix_and_cache(self) -> None:
        # 创建自动清理的临时根目录。
        with tempfile.TemporaryDirectory() as temporary_directory:
            # 将临时目录转换为 Path。
            root = Path(temporary_directory)
            # 定义跨进程共享会话状态文件。
            state_path = root / "deepseek_conversation_state.json"
            # 定义缓存审计收据文件。
            receipt_path = root / "deepseek_cache_receipt.json"
            # 为第一个 case 准备零命中的首轮响应。
            first_client = _FakeClient([_response(hit_tokens=0, miss_tokens=120)])
            # 为第二个 case 准备复用首轮前缀的命中响应。
            second_client = _FakeClient([_response(hit_tokens=120, miss_tokens=30)])
            # 保存假客户端队列。
            fake_clients = [first_client, second_client]
            # 初始化 SDK 构造参数记录。
            constructor_calls: list[dict[str, object]] = []

            # 定义假 OpenAI 构造器。
            def fake_openai(**kwargs: object) -> _FakeClient:
                # 记录 timeout 与 max_retries 等构造参数。
                constructor_calls.append(dict(kwargs))
                # 按 case 进程顺序返回一个客户端。
                return fake_clients[len(constructor_calls) - 1]

            # 构造包含假 OpenAI 类的模块。
            fake_openai_module = types.SimpleNamespace(OpenAI=fake_openai)
            # 在两个提供器调用期间替换真实 SDK，保证测试绝不联网。
            with patch.dict(sys.modules, {"openai": fake_openai_module}):
                # 创建第一个 case 的提供器。
                first_provider = DeepSeekProvider(api_key="test-only-key", state_path=state_path, receipt_path=receipt_path)
                # 执行第一个 case 的一轮决策。
                first_provider.decide({"case": {"case_id": "case-one"}, "iteration": 1}, io_dir=root / "runs" / "case-one" / "model_io", iteration=1)
                # 创建模拟第二个隔离进程的新提供器，并从磁盘恢复同一会话。
                second_provider = DeepSeekProvider(api_key="test-only-key", state_path=state_path, receipt_path=receipt_path)
                # 执行第二个 case 的一轮决策。
                second_provider.decide({"case": {"case_id": "case-two"}, "iteration": 1}, io_dir=root / "runs" / "case-two" / "model_io", iteration=1)
            # 验证每个隔离 case 只创建一次客户端。
            self.assertEqual(len(constructor_calls), 2)
            # 验证 SDK 内部重试被显式关闭。
            self.assertTrue(all(call.get("max_retries") == 0 for call in constructor_calls))
            # 验证每个 SDK 请求都有显式一百八十秒超时。
            self.assertTrue(all(call.get("timeout") == 180.0 for call in constructor_calls))
            # 读取第一个 case 实际发送的消息。
            first_messages = first_client.chat.completions.requests[0]["messages"]
            # 读取第二个 case 实际发送的消息。
            second_messages = second_client.chat.completions.requests[0]["messages"]
            # 验证首轮只有 system 与 user 两条消息。
            self.assertEqual(len(first_messages), 2)
            # 验证第二轮前两条消息逐字复用首轮完整请求前缀。
            self.assertEqual(second_messages[:2], first_messages)
            # 验证第二轮包含首轮 assistant 回答。
            self.assertEqual(second_messages[2]["role"], "assistant")
            # 验证第二轮最后才追加新 case 的 user 消息。
            self.assertEqual(second_messages[-1]["role"], "user")
            # 读取最终缓存收据。
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            # 验证两个 case 保持同一个逻辑会话。
            self.assertEqual(receipt["conversation_mode"], "single-growing-history-across-cases")
            # 验证相邻成功决策历史哈希链连续。
            self.assertTrue(receipt["history_chain_valid"])
            # 验证第二轮一百二十个输入 token 实际命中缓存。
            self.assertEqual(receipt["prompt_cache_hit_tokens"], 120)
            # 验证总体命中率为一百二十除以二百七十。
            self.assertAlmostEqual(receipt["cache_hit_ratio"], 120 / 270)

    # 验证无效 JSON 的修复请求仍接在同一消息链上。
    def test_contract_retry_appends_assistant_and_correction(self) -> None:
        # 创建自动清理的临时目录。
        with tempfile.TemporaryDirectory() as temporary_directory:
            # 将临时目录转换为 Path。
            root = Path(temporary_directory)
            # 准备先无效、后有效的两次响应。
            fake_client = _FakeClient([_response(hit_tokens=0, miss_tokens=80, content="not-json"), _response(hit_tokens=80, miss_tokens=10)])
            # 创建允许一次显式重试的提供器。
            provider = DeepSeekProvider(api_key="test-only-key", retries=1, state_path=root / "state.json", receipt_path=root / "receipt.json")
            # 直接注入假客户端，避免导入真实 SDK。
            provider._client = fake_client
            # 手动记录一次客户端创建，使跨进程审计计数符合真实路径。
            provider._client_creations = 1
            # 执行会先合同失败再成功的决策。
            provider.decide({"case": {"case_id": "retry-case"}, "iteration": 1}, io_dir=root / "runs" / "retry-case" / "model_io", iteration=1)
            # 验证真实 HTTP 尝试恰好为两次。
            self.assertEqual(len(fake_client.chat.completions.requests), 2)
            # 读取第二次修复请求消息。
            retry_messages = fake_client.chat.completions.requests[1]["messages"]
            # 验证修复请求包含 system、原 user、无效 assistant 和纠错 user。
            self.assertEqual([message["role"] for message in retry_messages], ["system", "user", "assistant", "user"])
            # 读取缓存收据。
            receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
            # 验证无效响应也被计入真实费用事件。
            self.assertEqual(receipt["response_count"], 2)
            # 验证只有最终响应被接受为 Agent 决策。
            self.assertEqual(receipt["accepted_decisions"], 1)
            # 验证 SDK 内层重试仍为零。
            self.assertEqual(receipt["sdk_max_retries"], 0)

    # 验证区域 Skill 在昂贵 PSO 前拒绝单一 shape 分区。
    def test_dynamic_region_contract_rejects_single_shape_partition(self) -> None:
        # 创建自动清理的临时目录。
        with tempfile.TemporaryDirectory() as temporary_directory:
            # 加载圆孔 case 合同。
            case = CaseDefinition.load(ROOT / "cases" / "circular_opening.json")
            # 创建最小运行上下文。
            context = RunContext(case, Path(temporary_directory))
            # 添加无候选点的最小分区证据，使测试只覆盖 shape 合同。
            context.add_json_artifact("partition_evidence", "test", "minimal partition evidence", {"field_candidates": []})
            # 构造两个合法但都为 box 的区域，其中一个明确可稀疏。
            arguments = {"regions": [{"region_id": "hot", "name": "hot", "shape": "box", "geometry": {"xmin": -40, "xmax": 40, "ymin": -40, "ymax": 40}, "role": "high_importance", "rationale": "cover the opening"}, {"region_id": "far", "name": "far", "shape": "box", "geometry": {"xmin": 60, "xmax": 110, "ymin": 60, "ymax": 110}, "role": "coarsenable", "rationale": "release far field"}]}
            # 断言单一 shape 在生成 region_partition artifact 前被拒绝。
            with self.assertRaisesRegex(ValueError, "at least two distinct shapes"):
                # 调用经过补丁的动态区域 Skill。
                propose_dynamic_regions(context, arguments)


# 仅在直接执行测试文件时运行 unittest。
if __name__ == "__main__":
    # 启动标准 unittest 运行器。
    unittest.main()
