# 启用延迟解析类型注解，避免运行时因前向引用产生额外依赖。
from __future__ import annotations

# 导入哈希模块，用于证明相邻请求复用了完全一致的消息前缀。
import hashlib
# 导入 JSON 模块，用于序列化请求、会话状态和缓存审计记录。
import json
# 导入操作系统模块，用于读取 GitHub Actions 与 DeepSeek 配置环境变量。
import os
# 导入时间模块，用于在受控外层重试之间执行短暂退避。
import time
# 导入 UUID 生成器，用于本地运行时创建不含敏感信息的会话标识。
import uuid
# 导入路径类型，用于以跨平台方式管理模型 I/O 和会话状态文件。
from pathlib import Path
# 导入通用类型，用于描述 DeepSeek 返回值和审计载荷。
from typing import Any

# 导入模型决策合同及其归一化函数，确保缓存改造不改变原有动作语义。
from .contracts import ContractError, normalize_model_decision
# 复用经原始 payload 固定的系统提示词，避免复制时改变首 token 前缀。
from .provider import SYSTEM_PROMPT


# 固定会话状态格式版本，便于阻止不同实现错误复用同一个缓存前缀。
STATE_SCHEMA_VERSION = "deepseek-growing-conversation/1.0"
# 固定缓存审计格式版本，便于工作流执行机器可读的验收门。
CACHE_SCHEMA_VERSION = "deepseek-context-cache-receipt/1.0"
# 使用 DeepSeek 官方 OpenAI 兼容入口作为默认服务地址。
DEFAULT_BASE_URL = "https://api.deepseek.com"
# 使用仓库当前验证过的推理模型作为默认模型。
DEFAULT_MODEL = "deepseek-v4-pro"
# 每轮默认最多允许五千个输出 token，以保持原论文流程的决策能力。
DEFAULT_MAX_TOKENS = 5000
# 每个决策默认最多发出两次 HTTP 请求，即一次首调和一次显式重试。
DEFAULT_MAX_ATTEMPTS_PER_DECISION = 2
# 每次 HTTP 请求默认最多等待一百八十秒，避免 SDK 默认十分钟等待失控。
DEFAULT_TIMEOUT_SECONDS = 180
# 全套四场景默认最多允许九十六次 HTTP 请求，对应八十八轮上限外加八次重试余量。
DEFAULT_MAX_HTTP_REQUESTS = 96


# 定义受范围保护的整数环境变量读取函数，避免异常配置扩大付费调用。
def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """读取整数环境变量，并强制限制在明确的安全区间内。"""
    # 读取原始字符串；未配置时使用经过审计的默认值。
    raw_value = os.environ.get(name, str(default)).strip()
    # 尝试把字符串转换为整数，非法输入将产生明确配置错误。
    try:
        # 完成十进制整数解析，不接受浮点近似。
        parsed_value = int(raw_value)
    # 捕获类型或数值格式错误，向调用方报告具体配置键。
    except (TypeError, ValueError) as exc:
        # 抛出包含变量名的错误，防止静默回退掩盖成本配置问题。
        raise RuntimeError(f"{name} must be an integer, got {raw_value!r}") from exc
    # 检查解析值是否落在预先定义的闭区间内。
    if not minimum <= parsed_value <= maximum:
        # 超界时立即失败，避免错误配置制造无界调用或无效超时。
        raise RuntimeError(f"{name} must be in [{minimum}, {maximum}], got {parsed_value}")
    # 返回已经通过类型与范围校验的整数。
    return parsed_value


# 定义稳定 JSON 编码函数，确保同一消息历史在不同进程中得到同一哈希。
def _canonical_json(value: Any) -> str:
    """将任意 JSON 兼容值编码为稳定、无多余空白的 UTF-8 文本。"""
    # 禁止 ASCII 转义以保留中文 token，并对字典键排序保证跨进程确定性。
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# 定义 JSON 载荷哈希函数，用于记录缓存前缀链而不重复保存每轮全部历史。
def _json_sha256(value: Any) -> str:
    """返回稳定 JSON 文本的 SHA-256 十六进制摘要。"""
    # 将稳定 JSON 文本编码为 UTF-8 字节后计算不可逆摘要。
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


# 定义原子 JSON 写入函数，避免进程中断留下半个会话状态文件。
def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """先写同目录临时文件，再原子替换目标 JSON。"""
    # 创建父目录，使首次运行也能直接持久化会话状态。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用固定临时后缀；四个场景顺序运行，因此不存在并发写冲突。
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    # 使用两空格缩进保存可审计文本，并在文件末尾保留标准换行。
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 用同卷原子替换提交完整文件，防止下一个场景读取截断状态。
    temporary_path.replace(path)


# 定义缓存感知 DeepSeek 提供器，保持原 AgentRuntime 所需的 decide 接口不变。
class DeepSeekProvider:
    """在四场景之间持久化一条增长消息链，并审计 DeepSeek 缓存命中。"""

    # 初始化模型、请求预算、会话状态和审计计数器。
    def __init__(
        # 当前提供器实例本身。
        self,
        # DeepSeek 私密 API 密钥；只驻留内存且绝不写入工件。
        api_key: str,
        # 当前请求使用的 DeepSeek 模型名称。
        model: str = DEFAULT_MODEL,
        # DeepSeek OpenAI 兼容 API 根地址。
        base_url: str = DEFAULT_BASE_URL,
        # 单次决策允许的最大输出 token 数。
        max_tokens: int = DEFAULT_MAX_TOKENS,
        # 单个决策允许的外层重试次数；一表示最多两次请求。
        retries: int = DEFAULT_MAX_ATTEMPTS_PER_DECISION - 1,
        # 单次 HTTP 请求的总超时秒数。
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        # 整个论文运行允许的 HTTP 请求总数。
        max_http_requests: int = DEFAULT_MAX_HTTP_REQUESTS,
        # 跨场景共享的完整消息历史状态文件。
        state_path: Path | None = None,
        # 不含消息正文的缓存命中审计收据文件。
        receipt_path: Path | None = None,
    ) -> None:
        # 保存 API 密钥，仅供 SDK 客户端构造使用。
        self.api_key = api_key
        # 保存模型名称，并作为会话状态兼容性校验的一部分。
        self.model = model
        # 保存服务根地址，允许受控测试注入兼容端点。
        self.base_url = base_url
        # 保存每轮最大输出 token 上限。
        self.max_tokens = max_tokens
        # 保存唯一的外层重试层数量，SDK 内层重试会被关闭。
        self.retries = retries
        # 保存显式请求超时秒数。
        self.timeout_seconds = timeout_seconds
        # 保存整套运行的 HTTP 请求硬上限。
        self.max_http_requests = max_http_requests
        # 保存可选的跨进程会话状态路径。
        self.state_path = state_path
        # 保存可选的机器可读缓存收据路径。
        self.receipt_path = receipt_path
        # 以原始系统提示词创建会话首消息，保证第零 token 前缀稳定。
        self._messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 初始化所有收到 HTTP 响应的 token 使用事件。
        self._response_events: list[dict[str, Any]] = []
        # 初始化传输、限流或合同解析失败记录。
        self._attempt_errors: list[dict[str, Any]] = []
        # 初始化已实际发出的 HTTP 请求总数。
        self._http_requests = 0
        # 初始化已成功归一化并交给 AgentRuntime 的决策数。
        self._accepted_decisions = 0
        # 初始化 OpenAI SDK 客户端引用，首次请求时才创建。
        self._client: Any = None
        # 初始化跨进程累计客户端创建数；每个隔离场景最多创建一个。
        self._client_creations = 0
        # 优先使用 GitHub run 标识构造可追溯但不含秘密的会话 ID。
        github_run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
        # 读取 GitHub rerun 尝试号，区分同一个 run 的重新执行。
        github_run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1").strip()
        # 有 GitHub run 时使用稳定组合，否则为本地测试生成随机 ID。
        self._conversation_id = f"github-{github_run_id}-attempt-{github_run_attempt}" if github_run_id else f"local-{uuid.uuid4().hex}"
        # 如果前一场景已经写入状态，则恢复同一条完整增长消息链。
        self._load_state()

    # 从受控环境变量构造提供器，集中执行凭据和成本配置校验。
    @classmethod
    def from_env(cls, model: str | None = None) -> "DeepSeekProvider":
        """从环境变量创建带硬预算和跨场景状态的提供器。"""
        # 读取并去除密钥两端空白，避免不可见字符导致鉴权失败。
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        # 缺少密钥时立即停止，不允许回放结果伪装成 live 论文证据。
        if not api_key:
            # 抛出明确错误供 GitHub Actions 凭据门定位。
            raise RuntimeError("DEEPSEEK_API_KEY is missing")
        # 优先使用命令行指定模型，否则读取环境变量并最终回退到固定模型。
        requested_model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        # 读取可选服务根地址，默认使用 DeepSeek 官方端点。
        base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        # 将最大输出 token 限制在一百至八千之间，避免意外超大输出。
        max_tokens = _bounded_env_int("DEEPSEEK_MAX_TOKENS", DEFAULT_MAX_TOKENS, 100, 8000)
        # 将每个决策的 HTTP 尝试数限制在一至两次。
        max_attempts = _bounded_env_int("DEEPSEEK_MAX_ATTEMPTS_PER_DECISION", DEFAULT_MAX_ATTEMPTS_PER_DECISION, 1, 2)
        # 将单次请求超时限制在三十至六百秒。
        timeout_seconds = _bounded_env_int("DEEPSEEK_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, 30, 600)
        # 将整套运行请求总数限制在一至九十六次，阻断隐式重试爆炸。
        max_http_requests = _bounded_env_int("DEEPSEEK_MAX_HTTP_REQUESTS", DEFAULT_MAX_HTTP_REQUESTS, 1, DEFAULT_MAX_HTTP_REQUESTS)
        # 读取跨场景会话状态路径；未配置时仅在当前进程内保持历史。
        state_text = os.environ.get("DEEPSEEK_CONVERSATION_STATE_PATH", "").strip()
        # 将非空状态路径转换为 Path，否则保留为空。
        state_path = Path(state_text) if state_text else None
        # 读取缓存审计收据路径；工作流会把它放在最终 artifact 根目录。
        receipt_text = os.environ.get("DEEPSEEK_CACHE_RECEIPT_PATH", "").strip()
        # 将非空收据路径转换为 Path，否则尝试从状态文件旁派生。
        receipt_path = Path(receipt_text) if receipt_text else (state_path.with_name("deepseek_cache_receipt.json") if state_path else None)
        # 返回完成全部安全配置的提供器实例。
        return cls(
            # 传入只驻留内存的 API 密钥。
            api_key=api_key,
            # 传入已经解析的模型名称。
            model=requested_model,
            # 传入官方或显式覆盖的服务根地址。
            base_url=base_url,
            # 传入受范围保护的输出 token 上限。
            max_tokens=max_tokens,
            # 将尝试次数转换为“额外重试次数”。
            retries=max_attempts - 1,
            # 传入显式请求超时。
            timeout_seconds=timeout_seconds,
            # 传入全套运行 HTTP 请求硬上限。
            max_http_requests=max_http_requests,
            # 传入跨场景状态路径。
            state_path=state_path,
            # 传入缓存审计收据路径。
            receipt_path=receipt_path,
        )

    # 从前一场景留下的状态文件恢复消息历史和累计用量。
    def _load_state(self) -> None:
        """校验并恢复跨进程会话状态。"""
        # 没有配置状态路径时保持当前进程内会话，不执行磁盘恢复。
        if self.state_path is None:
            # 提前返回，避免无意义的文件系统访问。
            return
        # 首个场景尚未创建状态文件时使用初始 system 消息。
        if not self.state_path.exists():
            # 提前返回，后续首个成功响应会创建状态。
            return
        # 解析 UTF-8 JSON 状态；损坏文件会直接阻断而不是开启新会话。
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        # 验证状态格式版本，防止不兼容字段被静默解释。
        if payload.get("schema_version") != STATE_SCHEMA_VERSION:
            # 报告实际版本，便于审计人员定位旧工件。
            raise RuntimeError(f"incompatible conversation state schema: {payload.get('schema_version')!r}")
        # 验证模型一致；切换模型会改变 token 化与缓存前缀。
        if payload.get("requested_model") != self.model:
            # 阻止把其他模型历史混入当前论文运行。
            raise RuntimeError("conversation state model does not match requested DeepSeek model")
        # 计算当前固定系统提示词摘要。
        expected_prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        # 验证系统提示词没有发生任何字符级变化。
        if payload.get("system_prompt_sha256") != expected_prompt_hash:
            # 阻止在前缀变化后假称同一会话。
            raise RuntimeError("conversation state system prompt hash does not match current payload")
        # 读取持久化消息列表。
        messages = payload.get("messages")
        # 验证消息列表至少包含固定 system 消息。
        if not isinstance(messages, list) or not messages:
            # 拒绝空历史，避免无声退化成新会话。
            raise RuntimeError("conversation state messages are missing")
        # 验证第一条消息角色和正文都与当前系统提示词完全一致。
        if messages[0] != {"role": "system", "content": SYSTEM_PROMPT}:
            # 拒绝第零 token 不一致的历史。
            raise RuntimeError("conversation state does not start with the fixed system prompt")
        # 恢复完整消息链，使新请求把旧请求与响应原样放在前缀中。
        self._messages = [dict(message) for message in messages]
        # 恢复已记录的 HTTP 响应事件。
        self._response_events = list(payload.get("response_events") or [])
        # 恢复传输和合同错误事件。
        self._attempt_errors = list(payload.get("attempt_errors") or [])
        # 恢复已发 HTTP 请求总数，确保预算跨四个子进程累计。
        self._http_requests = int(payload.get("http_requests") or 0)
        # 恢复成功决策总数。
        self._accepted_decisions = int(payload.get("accepted_decisions") or 0)
        # 恢复累计客户端创建数，用于证明每个隔离场景只创建一次客户端。
        self._client_creations = int(payload.get("client_creations") or 0)
        # 恢复同一会话 ID，证明四个场景没有重新开逻辑会话。
        self._conversation_id = str(payload.get("conversation_id") or self._conversation_id)

    # 延迟创建并复用 OpenAI SDK 客户端，同时关闭 SDK 隐式重试。
    def _client_instance(self) -> Any:
        """返回当前场景进程唯一的 DeepSeek SDK 客户端。"""
        # 已存在客户端时直接复用，避免每轮重新建立连接池。
        if self._client is not None:
            # 返回同一对象，保证当前场景内不会产生“每轮新客户端”。
            return self._client
        # 仅在首次真实请求前导入 OpenAI SDK，保持本地合同测试轻量。
        from openai import OpenAI
        # 构造客户端并显式关闭 SDK 的两次默认重试，只保留可计数的外层重试。
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0, timeout=float(self.timeout_seconds))
        # 累加跨进程客户端创建数；四个隔离 case 理论上最多为四。
        self._client_creations += 1
        # 返回刚创建的唯一客户端。
        return self._client

    # 将 SDK usage 对象转换为包含缓存字段的普通字典。
    def _usage_payload(self, response: Any) -> dict[str, Any]:
        """提取总 token、推理 token及官方缓存命中/未命中 token。"""
        # 读取可选 usage 对象；异常响应可能没有该字段。
        usage = getattr(response, "usage", None)
        # 读取可选 completion 细分对象。
        completion_details = getattr(usage, "completion_tokens_details", None)
        # 返回完整可序列化用量，不用零值掩盖服务端缺失字段。
        return {
            # 记录总输入 token，理论上等于命中与未命中之和。
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            # 记录输出 token，用于后续成本核算。
            "completion_tokens": getattr(usage, "completion_tokens", None),
            # 记录输入输出合计 token。
            "total_tokens": getattr(usage, "total_tokens", None),
            # 记录思考模式产生的推理 token。
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
            # 记录 DeepSeek 官方返回的输入缓存命中 token。
            "prompt_cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", None),
            # 记录 DeepSeek 官方返回的输入缓存未命中 token。
            "prompt_cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", None),
        }

    # 计算相邻成功决策的历史哈希是否首尾连续。
    def _history_chain_valid(self) -> bool:
        """验证每个成功决策后的历史恰好成为下一成功决策的前缀。"""
        # 只选择已经通过 JSON 与决策合同校验的响应事件。
        accepted_events = [event for event in self._response_events if event.get("accepted") is True]
        # 逐对检查相邻成功事件；少于两项时循环自然通过。
        for previous_event, current_event in zip(accepted_events, accepted_events[1:]):
            # 若上一轮结束哈希不是下一轮开始哈希，则前缀链已断裂。
            if previous_event.get("history_after_sha256") != current_event.get("history_before_sha256"):
                # 立即返回失败，供工作流缓存 gate 阻断论文签发。
                return False
        # 所有相邻事件均连续时返回成功。
        return True

    # 生成包含完整消息历史的可恢复状态载荷。
    def _state_payload(self) -> dict[str, Any]:
        """构造跨场景恢复所需的完整状态。"""
        # 计算固定系统提示词摘要，作为第零 token 契约。
        prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        # 返回不包含 API 密钥的会话状态。
        return {
            # 标注会话状态格式版本。
            "schema_version": STATE_SCHEMA_VERSION,
            # 标注同一次论文运行的逻辑会话 ID。
            "conversation_id": self._conversation_id,
            # 标注请求模型，防止跨模型复用历史。
            "requested_model": self.model,
            # 标注系统提示词哈希，防止前缀漂移。
            "system_prompt_sha256": prompt_hash,
            # 保存完整多轮消息，这是 DeepSeek 无状态 API 复用上下文的必要输入。
            "messages": self._messages,
            # 保存每个收到响应的 token 与缓存事件。
            "response_events": self._response_events,
            # 保存没有正常响应或合同无效的尝试事件。
            "attempt_errors": self._attempt_errors,
            # 保存全套运行已经发出的 HTTP 请求数。
            "http_requests": self._http_requests,
            # 保存已接受决策数。
            "accepted_decisions": self._accepted_decisions,
            # 保存跨四个隔离场景累计创建的客户端数。
            "client_creations": self._client_creations,
        }

    # 生成不含消息正文的缓存审计收据。
    def _receipt_payload(self) -> dict[str, Any]:
        """汇总缓存命中率、预算、重试和前缀链证据。"""
        # 选择服务端确实返回了整数命中字段的事件。
        complete_usage_events = [event for event in self._response_events if isinstance(event.get("usage", {}).get("prompt_cache_hit_tokens"), int) and isinstance(event.get("usage", {}).get("prompt_cache_miss_tokens"), int)]
        # 汇总所有官方缓存命中 token。
        cache_hit_tokens = sum(int(event["usage"]["prompt_cache_hit_tokens"]) for event in complete_usage_events)
        # 汇总所有官方缓存未命中 token。
        cache_miss_tokens = sum(int(event["usage"]["prompt_cache_miss_tokens"]) for event in complete_usage_events)
        # 计算可审计输入 token 总数。
        observed_prompt_tokens = cache_hit_tokens + cache_miss_tokens
        # 计算总体缓存命中比例；尚无用量时明确记为零。
        cache_hit_ratio = cache_hit_tokens / observed_prompt_tokens if observed_prompt_tokens else 0.0
        # 统计至少命中过一个 token 的响应数量。
        responses_with_cache_hit = sum(1 for event in complete_usage_events if int(event["usage"]["prompt_cache_hit_tokens"]) > 0)
        # 返回供 GitHub Actions gate 使用的收据。
        return {
            # 标注缓存收据格式版本。
            "schema_version": CACHE_SCHEMA_VERSION,
            # 标注逻辑会话 ID，四个场景应保持唯一。
            "conversation_id": self._conversation_id,
            # 明确说明采用跨场景单一增长历史，而不是每轮新会话。
            "conversation_mode": "single-growing-history-across-cases",
            # 标注当前 DeepSeek 模型。
            "requested_model": self.model,
            # 标注当前完整消息数量。
            "history_message_count": len(self._messages),
            # 标注当前完整历史哈希，便于与最终状态文件交叉核验。
            "history_sha256": _json_sha256(self._messages),
            # 标注相邻成功决策历史链是否连续。
            "history_chain_valid": self._history_chain_valid(),
            # 标注 SDK 内部重试被彻底关闭。
            "sdk_max_retries": 0,
            # 标注每个决策允许的显式尝试次数。
            "outer_max_attempts_per_decision": self.retries + 1,
            # 标注单次请求超时秒数。
            "timeout_seconds": self.timeout_seconds,
            # 标注整个论文运行 HTTP 请求硬上限。
            "max_http_requests": self.max_http_requests,
            # 标注已经发出的 HTTP 请求数。
            "http_requests": self._http_requests,
            # 标注收到正常 HTTP 响应的次数。
            "response_count": len(self._response_events),
            # 标注成功交付给 AgentRuntime 的决策数。
            "accepted_decisions": self._accepted_decisions,
            # 标注四个隔离 case 子进程累计创建的 SDK 客户端数。
            "client_creations": self._client_creations,
            # 解释客户端连接与服务端缓存逻辑会话的区别。
            "client_scope": "one SDK client per isolated case process; one serialized DeepSeek conversation for the full suite",
            # 标注具有完整官方缓存 usage 字段的响应数。
            "responses_with_cache_usage": len(complete_usage_events),
            # 标注至少发生一次缓存命中的响应数。
            "responses_with_cache_hit": responses_with_cache_hit,
            # 汇总官方缓存命中 token。
            "prompt_cache_hit_tokens": cache_hit_tokens,
            # 汇总官方缓存未命中 token。
            "prompt_cache_miss_tokens": cache_miss_tokens,
            # 汇总可审计输入 token。
            "observed_prompt_tokens": observed_prompt_tokens,
            # 记录总体缓存命中比例，范围为零至一。
            "cache_hit_ratio": cache_hit_ratio,
            # 保存逐响应事件，供论文复现与成本核查。
            "response_events": self._response_events,
            # 保存传输与合同错误，但不包含 API 密钥。
            "attempt_errors": self._attempt_errors,
        }

    # 同步持久化完整状态与精简收据。
    def _persist(self) -> None:
        """把当前会话和缓存统计原子写入工作流 artifact 目录。"""
        # 配置了状态路径时写入完整可恢复消息历史。
        if self.state_path is not None:
            # 原子提交状态，供下一个 case 子进程继续同一会话。
            _atomic_write_json(self.state_path, self._state_payload())
        # 配置了收据路径时写入不含消息正文的缓存统计。
        if self.receipt_path is not None:
            # 原子提交收据，供独立 audit 脚本和最终 gate 使用。
            _atomic_write_json(self.receipt_path, self._receipt_payload())

    # 执行一次 DeepSeek 决策，并把成功响应接到增长消息链末尾。
    def decide(self, packet: dict[str, Any], *, io_dir: Path, iteration: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """发送当前证据包，返回合同化决策及可审计缓存元数据。"""
        # 创建当前 case 的模型 I/O 目录。
        io_dir.mkdir(parents=True, exist_ok=True)
        # 将当前证据包编码为紧凑中文 JSON，作为本轮新增 user 消息。
        packet_text = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
        # 计算加入本轮 user 消息之前的完整历史哈希。
        history_before_sha256 = _json_sha256(self._messages)
        # 记录加入本轮 user 消息之前的历史消息数。
        history_message_count_before = len(self._messages)
        # 构造当前 user 消息。
        user_message = {"role": "user", "content": packet_text}
        # 从持久历史复制工作消息，并把当前证据包追加在末尾。
        working_messages = [dict(message) for message in self._messages] + [user_message]
        # 构造每轮请求记录路径。
        request_path = io_dir / f"iteration_{iteration:02d}_request.json"
        # 保存证据包及前缀哈希，而不在每轮文件中重复全部增长历史正文。
        _atomic_write_json(request_path, {"schema_version": "deepseek-request-audit/1.0", "conversation_id": self._conversation_id, "history_before_sha256": history_before_sha256, "history_message_count_before": history_message_count_before, "request_message_count": len(working_messages), "packet": packet})
        # 获取当前 case 进程内唯一 SDK 客户端。
        client = self._client_instance()
        # 初始化本轮全部错误摘要。
        errors: list[str] = []
        # 初始化最后一次模型文本，供失败工件保留。
        last_text = ""
        # 初始化最后一次可解析 JSON，供失败工件保留。
        raw_payload: Any = None
        # 在明确的外层尝试上限内执行请求。
        for attempt in range(self.retries + 1):
            # 在每次真实请求前检查跨场景总预算。
            if self._http_requests >= self.max_http_requests:
                # 构造不含秘密的预算耗尽信息。
                budget_error = f"DeepSeek HTTP request budget exhausted at {self._http_requests}/{self.max_http_requests}"
                # 保存预算错误供审计。
                errors.append(budget_error)
                # 跳出尝试循环，统一写入失败响应。
                break
            # 计算本次实际发送消息的完整哈希。
            request_sha256 = _json_sha256(working_messages)
            # 在发请求前累计计数，使网络中断也不会丢失已消费尝试。
            self._http_requests += 1
            # 持久化请求计数与客户端创建数，确保子进程崩溃后仍受总预算约束。
            self._persist()
            # 清空本次尝试的模型文本，区分传输失败与合同失败。
            last_text = ""
            # 调用 DeepSeek Chat Completions；SDK 内部不会自动重试。
            try:
                # 发出带 JSON 输出约束和思考模式的真实模型请求。
                response = client.chat.completions.create(model=self.model, messages=[dict(message) for message in working_messages], response_format={"type": "json_object"}, max_tokens=self.max_tokens, extra_body={"thinking": {"type": "enabled"}})
            # 捕获传输、鉴权、限流和服务端异常。
            except Exception as exc:
                # 生成简短错误文本，不记录请求头或密钥。
                error_text = f"{type(exc).__name__}: {exc}"
                # 把错误加入当前决策摘要。
                errors.append(error_text)
                # 记录本次失败的 case、轮次、尝试和请求哈希。
                self._attempt_errors.append({"kind": "transport_or_api_error", "case_id": io_dir.parent.name, "iteration": iteration, "attempt": attempt + 1, "request_sha256": request_sha256, "error": error_text})
                # 立即持久化已消费请求和错误。
                self._persist()
                # 已达到显式尝试上限时停止重试。
                if attempt >= self.retries:
                    # 跳出循环，统一写失败响应。
                    break
                # 按一、二秒级指数退避等待；默认只有一次一秒退避。
                time.sleep(2**attempt)
                # 传输失败时原样重发同一消息前缀，避免额外制造新上下文。
                continue
            # 提取 DeepSeek 官方 token 与缓存 usage 字段。
            usage_payload = self._usage_payload(response)
            # 读取当前响应所使用的实际模型名。
            actual_model = getattr(response, "model", self.model)
            # 读取输出文本；空值转为空字符串以便合同校验。
            last_text = response.choices[0].message.content or ""
            # 构造该 HTTP 响应的审计事件，先标为尚未通过合同。
            response_event = {"case_id": io_dir.parent.name, "iteration": iteration, "attempt": attempt + 1, "request_sha256": request_sha256, "request_message_count": len(working_messages), "history_before_sha256": history_before_sha256, "history_message_count_before": history_message_count_before, "model": actual_model, "accepted": False, "usage": usage_payload}
            # 追加响应事件，使无效 JSON 也计入真实费用。
            self._response_events.append(response_event)
            # 立即持久化官方 usage，防止后续合同解析失败丢失成本证据。
            self._persist()
            # 尝试解析 JSON 并套用既有决策合同。
            try:
                # 将模型正文解析为 JSON 对象。
                raw_payload = json.loads(last_text)
                # 归一化模型可能使用的兼容字段名并校验动作结构。
                normalized, notes = normalize_model_decision(raw_payload)
            # 捕获 JSON 解析或决策合同错误。
            except Exception as exc:
                # 生成合同错误摘要。
                error_text = f"{type(exc).__name__}: {exc}"
                # 把合同错误加入当前决策摘要。
                errors.append(error_text)
                # 记录本次无效输出错误。
                self._attempt_errors.append({"kind": "json_or_contract_error", "case_id": io_dir.parent.name, "iteration": iteration, "attempt": attempt + 1, "request_sha256": request_sha256, "error": error_text})
                # 已达到显式尝试上限时停止修复请求。
                if attempt >= self.retries:
                    # 持久化最终错误后跳出循环。
                    self._persist()
                    # 跳出循环，统一写失败响应。
                    break
                # 把无效 assistant 文本加入工作历史，保持真实多轮前缀。
                working_messages.append({"role": "assistant", "content": last_text})
                # 构造只要求修复 JSON 合同的 user 消息，不重新发送整包状态。
                correction_text = "上一份JSON无法执行：" + error_text + "。请只返回符合指定结构的JSON；Skill名必须来自skill_catalog，参数必须满足schema。"
                # 把修复消息追加到同一会话，使重试命中此前完整前缀。
                working_messages.append({"role": "user", "content": correction_text})
                # 持久化错误事件后执行短退避。
                self._persist()
                # 默认仅等待一秒，避免无意义占用 runner。
                time.sleep(2**attempt)
                # 继续下一次受预算保护的请求。
                continue
            # 构造成功 assistant 消息；本流程不使用原生 tool_calls，因此无需回传 reasoning_content。
            assistant_message = {"role": "assistant", "content": last_text}
            # 把成功回答追加到工作消息末尾。
            working_messages.append(assistant_message)
            # 将完整工作消息提交为下一轮和下一场景的持久历史。
            self._messages = [dict(message) for message in working_messages]
            # 累加成功决策数。
            self._accepted_decisions += 1
            # 把当前响应事件标记为合同已接受。
            response_event["accepted"] = True
            # 记录成功后完整历史哈希，供下一轮前缀链校验。
            response_event["history_after_sha256"] = _json_sha256(self._messages)
            # 记录成功后完整历史消息数。
            response_event["history_message_count_after"] = len(self._messages)
            # 持久化增长历史和最终缓存统计。
            self._persist()
            # 读取当前响应缓存命中 token。
            current_hit_tokens = usage_payload.get("prompt_cache_hit_tokens")
            # 读取当前响应缓存未命中 token。
            current_miss_tokens = usage_payload.get("prompt_cache_miss_tokens")
            # 仅在两者均为整数时计算本次缓存命中比例。
            current_observed_tokens = current_hit_tokens + current_miss_tokens if isinstance(current_hit_tokens, int) and isinstance(current_miss_tokens, int) else None
            # 有可观察输入 token 时计算比例，否则明确记为空。
            current_hit_ratio = current_hit_tokens / current_observed_tokens if isinstance(current_observed_tokens, int) and current_observed_tokens > 0 else None
            # 构造与原 AgentRuntime 兼容且增加缓存字段的元数据。
            metadata = {"provider": "deepseek", "model": actual_model, "requested_model": self.model, "finish_reason": getattr(response.choices[0], "finish_reason", None), "normalization_notes": notes, "usage": usage_payload, "cache_hit_ratio": current_hit_ratio, "conversation_id": self._conversation_id, "history_message_count": len(self._messages), "http_requests_used": self._http_requests, "http_request_budget": self.max_http_requests, "attempt": attempt + 1}
            # 构造完整成功响应记录。
            response_record = {"raw_text": last_text, "raw_json": raw_payload, "normalized_decision": normalized, "metadata": metadata, "prior_errors": errors}
            # 原子写入当前轮响应工件。
            _atomic_write_json(io_dir / f"iteration_{iteration:02d}_response.json", response_record)
            # 返回合同化决策和缓存审计元数据。
            return normalized, metadata
        # 构造所有尝试失败后的响应工件。
        failure_record = {"raw_text": last_text, "raw_json": raw_payload, "errors": errors, "conversation_id": self._conversation_id, "http_requests_used": self._http_requests, "http_request_budget": self.max_http_requests}
        # 原子写入失败响应，确保论文运行不会静默缺失模型 I/O。
        _atomic_write_json(io_dir / f"iteration_{iteration:02d}_response.json", failure_record)
        # 再次持久化最终错误和预算状态。
        self._persist()
        # 抛出原合同错误类型，由 AgentRuntime 写入 provider_error 事件。
        raise ContractError("DeepSeek decision failed: " + "; ".join(errors))
