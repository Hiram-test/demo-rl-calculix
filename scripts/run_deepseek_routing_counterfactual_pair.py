#!/usr/bin/env python3  # 使用仓库环境中的 Python 运行独立路由反事实实验。
from __future__ import annotations  # 允许使用现代类型注解并保持 Python 3.11 兼容。

import argparse  # 解析模型名称、随机顺序与结果输出路径。
import hashlib  # 计算两次请求共用前缀的可审计摘要。
import json  # 序列化匿名证据包并解析 DeepSeek 的 JSON 决策。
import os  # 从 GitHub Environment 注入的环境变量读取 DeepSeek 凭据。
import random  # 用固定种子打乱两个匿名样本的提交顺序。
import sys  # 把仓库根目录加入模块搜索路径。
from datetime import datetime  # 记录不参与提示词的实验开始时间。
from datetime import timezone  # 使用 UTC 时区生成可复现格式的时间戳。
from pathlib import Path  # 统一处理仓库根目录与结果文件路径。
from typing import Any  # 表示证据包、模型 JSON 与 API 元数据的动态字段。

from openai import OpenAI  # 通过 OpenAI 兼容接口调用 ds Environment 中的 DeepSeek。

ROOT = Path(__file__).resolve().parents[1]  # 定位脚本所属仓库的根目录。
sys.path.insert(0, str(ROOT))  # 允许脚本直接导入仓库内的有限元场景模块。

from bridge_mesh_suite.scenarios import run_crack_case  # 生成具有内部零厚度拓扑不连续的真实有限元证据。
from bridge_mesh_suite.scenarios import run_diaphragm_opening_case  # 生成拓扑连续但含零圆角几何尖点的真实有限元证据。

MAX_HTTP_REQUESTS = 2  # 把本实验的 DeepSeek HTTP 请求硬上限固定为两个匿名样本各一次。
DEFAULT_MODEL = "deepseek-v4-pro"  # 复用 ds Environment 当前已验证可用的 DeepSeek 模型名称。
DEFAULT_SEED = 20260731  # 固定匿名样本顺序的默认种子以便复现实验。
API_TIMEOUT_SECONDS = 180.0  # 允许单次推理最多等待三分钟且不触发 SDK 自动重试。
MAX_COMPLETION_TOKENS = 3200  # 限制每次公开回答与推理的总输出预算以控制费用。
MISLEADING_SAMPLE_ID = "specimen_2718"  # 使用不表达物理类别的匿名编号标识关键词误导样本。
HIDDEN_SAMPLE_ID = "specimen_3141"  # 使用不表达物理类别的匿名编号标识关键词隐藏样本。
MISLEADING_USER_TEXT = "这处裂纹角点的最大应力随着网格加密持续升高。请判断目前应采用哪一种评价路线，并选择一个最有判别力的下一步。"  # 故意保留“裂纹”字样以检验模型是否会忽略相反的连续拓扑事实。
HIDDEN_USER_TEXT = "板中央有一条 40 mm 未连接的内边界，两侧没有连接且都是自由表面，端部附近最大应力随网格加密持续升高。请判断目前应采用哪一种评价路线，并选择一个最有判别力的下一步。"  # 不使用目标类别词而只陈述可核验拓扑事实。
HIDDEN_FORBIDDEN_TERMS = ("裂纹", "断裂", "裂尖", "crack", "fracture", "J积分", "K因子", "应力强度因子")  # 定义隐藏样本不得泄露的目标类别与专用量词汇。
ROUTE_CATALOG = {  # 定义两例完全共用的中性分析路线目录。
    "R1": "连续材料域中的几何局部化路线；圆角、开孔或零半径尖角控制局部场，点峰值可以有界也可以具有几何奇异性，宜比较固定物理位置、固定区域或受控几何变体。",  # 描述连续拓扑下同时容纳有界集中与几何奇异点的评价路线。
    "R2": "内部零厚度拓扑不连续端点路线；优先采用路径量、能量型驱动力或适用理论量进行评价。",  # 描述内部不连续端点的驱动力评价路线。
    "R3": "材料非线性主导路线；需要真实材料曲线并执行相应非线性分析。",  # 描述材料屈服或硬化可能主导时的路线。
    "R4": "现有证据不足以确定路线；应先请求能区分竞争假设的最小补充事实。",  # 描述缺少关键事实时的保守路线。
}  # 完成中性路线目录定义。
TOOL_CATALOG = {  # 定义两例完全共用且顺序固定的中性工具目录。
    "T1": "在几何、材料、载荷和边界不变时增加一档同模型网格证据。",  # 提供同模型网格细化工具。
    "T2": "比较多个网格在相同固定物理位置的响应。",  # 提供固定位置取样工具。
    "T3": "比较多个网格在相同固定物理区域内的平均响应。",  # 提供固定区域平均工具。
    "T4": "只改变一个小的实体几何尺度并比较局部与整体响应。",  # 提供受控实体几何变体工具。
    "T5": "在物理前置条件满足时计算内部不连续端点的路径量或能量型驱动力。",  # 提供内部不连续端点专用评价工具但不点名目标类别。
    "T6": "检查局部边界链、节点共享、相对自由面与连接连续性。",  # 提供独立拓扑核验工具。
    "T7": "请求真实应力—塑性应变曲线、硬化模型及必要材料参数。",  # 提供材料非线性信息请求工具。
    "T8": "在材料数据充分后执行材料非线性分析。",  # 提供材料非线性求解工具。
    "T9": "现有证据足够时停止并给出带适用边界的暂定结论。",  # 提供无需额外计算的主动结束工具。
}  # 完成中性工具目录定义。
OUTPUT_SCHEMA = {  # 定义模型必须返回的公开 JSON 字段。
    "route_id": "R1、R2、R3 或 R4 中唯一一个字符串",  # 要求模型明确选择当前分析路线。
    "selected_tool_id": "T1 至 T9 中唯一一个字符串",  # 要求模型明确选择最有判别力的下一步。
    "competing_hypotheses": "至少两个字符串组成的数组",  # 要求模型显式保留竞争解释。
    "evidence_refs": "引用当前个案包字段路径的字符串数组",  # 要求模型把判断锚定到输入事实。
    "uncertainties": "仍未解决的不确定性字符串数组",  # 要求模型保留证据边界。
    "engineering_reason": "不披露隐藏思维链的简短公开工程理由字符串",  # 要求模型给出可审计理由而非私密推理。
    "provisional_answer": "面向用户的暂定回答字符串",  # 要求模型直接回应工程问题。
    "confidence": "零到一之间的数值",  # 要求模型量化当前判断信心。
}  # 完成输出结构定义。
COMMON_PROTOCOL = {  # 组织两次请求逐字节共用的任务协议。
    "task": "只依据当前匿名个案包中的可核验事实，选择一个分析路线和一个最有判别力的下一步。",  # 定义一次性分类与动作选择任务。
    "decision_principles": [  # 定义不针对任何单一场景的通用工程原则。
        "区分用户用词、直接模型事实、数值观察与尚未验证的解释。",  # 要求模型区分语言和证据层级。
        "当自然语言与几何、拓扑或连接事实可能冲突时，以可审计的模型事实为判断依据。",  # 要求模型处理事实冲突但不泄露任何正确标签。
        "最大离散点值随网格变化并不单独决定物理机制；必须同时检查评价位置、拓扑、整体响应与能量。",  # 提供通用有限元判断原则。
        "形成至少两个竞争假设，并选择最能区分它们的一项下一步。",  # 要求一次性给出判别性动作。
        "不得补造未提供的材料、连接、几何半径、载荷或边界事实。",  # 禁止模型用虚构事实完成分类。
    ],  # 完成通用工程原则列表。
    "route_catalog": ROUTE_CATALOG,  # 向两个匿名样本提供同一份路线目录。
    "tool_catalog": TOOL_CATALOG,  # 向两个匿名样本提供同一份工具目录。
    "output_schema": OUTPUT_SCHEMA,  # 向两个匿名样本提供同一份 JSON 输出合同。
    "response_rule": "只输出一个合法 JSON 对象，不输出 Markdown，不请求或等待第二轮。",  # 明确本实验没有工具循环或修复重试。
}  # 完成共用协议定义。
POSTHOC_EXPECTED_ROUTES = {MISLEADING_SAMPLE_ID: "R1", HIDDEN_SAMPLE_ID: "R2"}  # 仅供两次回答冻结后的离线评分使用且绝不写入请求。
POSTHOC_ACCEPTABLE_TOOLS = {  # 定义只在回答冻结后使用的宽松动作适配集合。
    MISLEADING_SAMPLE_ID: {"T1", "T2", "T3", "T4", "T6", "T9"},  # 连续几何样本允许细化、固定量、几何变体、拓扑核验或停止。
    HIDDEN_SAMPLE_ID: {"T5", "T6", "T9"},  # 内部不连续样本允许专用量、拓扑核验或已有证据下停止。
}  # 完成事后动作适配集合定义。


def _canonical_json(value: Any) -> str:  # 生成字段顺序稳定且不含无意义空白的 UTF-8 JSON 文本。
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)  # 使用稳定键序和分隔符支持前缀缓存与哈希审计。


def _common_messages() -> list[dict[str, str]]:  # 构造两次请求逐字节一致的系统消息与协议消息。
    system_prompt = (  # 组合不包含个案标签或期望答案的系统提示。
        "你是结构有限元证据分类代理。"  # 定义模型承担工程证据分类职责。
        "你只输出公开、简短、可审计的工程判断，不输出隐藏思维链。"  # 限制输出为可交付工程理由。
        "你必须根据提供的几何、拓扑、材料、载荷和数值事实作答。"  # 要求判断锚定在可核验事实。
    )  # 完成系统提示组合。
    protocol_prompt = "以下协议对随后每个匿名个案完全相同：\n" + _canonical_json(COMMON_PROTOCOL)  # 把固定目录与结构合同放在可缓存公共前缀中。
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": protocol_prompt}]  # 返回固定顺序的两条公共消息。


def _continuous_geometry_case() -> dict[str, Any]:  # 用真实有限元求解构造含误导词但拓扑连续的匿名样本。
    scenario = run_diaphragm_opening_case()  # 执行三档零圆角开孔模型及仓库现有确定性有限元求解。
    mesh_history: list[dict[str, Any]] = []  # 初始化只含原始数值观察的网格历史。
    for row in scenario.level_rows:  # 遍历三档真实求解结果而不读取既有诊断结论。
        mesh_history.append({  # 选择模型可见且不含预设路线的原始字段。
            "level": int(row["level"]),  # 记录离散层级编号。
            "elements": int(row["elements"]),  # 记录当前二维单元总数。
            "h_local_mm": float(row["h_local"]),  # 记录角点附近特征网格尺寸。
            "local_peak_mpa": float(row["peak_stress"]),  # 记录随网格变化的局部最大等效应力。
            "fixed_15mm_mpa": float(row["fixed_qoi"]),  # 记录距角点固定十五毫米处的响应。
            "remote_displacement_mm": float(row["remote_displacement"]),  # 记录远场两侧相对位移。
            "far_field_stress_mpa": float(row["far_field_stress"]),  # 记录远离局部特征的平均应力。
            "strain_energy_n_mm": float(row["strain_energy"]),  # 记录全模型应变能。
            "energy_balance_relative": float(row["energy_balance_rel"]),  # 记录内外功相对平衡误差。
        })  # 完成当前网格层级的可见证据。
    return {  # 返回不包含场景名、诊断、Skill 轨迹或圆角变体答案的匿名个案。
        "sample_id": MISLEADING_SAMPLE_ID,  # 提供不表达类别含义的匿名样本编号。
        "user_text": MISLEADING_USER_TEXT,  # 提供含“裂纹”误导词的自然语言问题。
        "model_facts": {  # 组织可核验的模型事实。
            "analysis": "二维平面应力线弹性静力分析",  # 记录当前求解假设。
            "domain_mm": {"width": 260.0, "height": 180.0, "thickness": 12.0},  # 记录板域宽、高和厚度毫米值。
            "material": {"young_mpa": 210000.0, "poisson": 0.30, "plastic_curve": None},  # 记录线弹性材料并明确没有塑性曲线。
            "loading": {"type": "左右边界对称均布拉伸", "nominal_stress_mpa": 90.0},  # 记录远场名义拉应力。
            "feature_geometry": {"shape": "闭合矩形内边界", "half_width_mm": 45.0, "half_height_mm": 25.0, "documented_corner_radius_mm": 0.0},  # 记录闭合开孔的零圆角几何。
            "local_topology": {"closed_internal_boundary_count": 1, "boundary_chain_is_continuous": True, "nodes_are_shared_across_boundary_chain": True, "opposed_coincident_free_face_pair": False, "connectivity_discontinuity": False},  # 明确该局部是共享节点的连续边界而非两张相对自由面。
        },  # 完成连续几何样本的模型事实。
        "mesh_history": mesh_history,  # 附加三档真实求解的原始网格趋势。
    }  # 完成连续几何匿名个案构造。


def _internal_discontinuity_case() -> dict[str, Any]:  # 用真实有限元求解构造无目标词但具有内部不连续拓扑的匿名样本。
    scenario = run_crack_case()  # 执行三档内部不连续模型及仓库现有确定性有限元求解。
    mesh_history: list[dict[str, Any]] = []  # 初始化已移除专用派生量的网格历史。
    for row in scenario.level_rows:  # 遍历三档真实求解结果而不读取既有诊断结论。
        mesh_history.append({  # 只保留峰值、能量和平衡等通用原始观察。
            "level": int(row["level"]),  # 记录离散层级编号。
            "mesh": str(row["mesh"]),  # 记录当前规则网格划分。
            "elements": int(row["elements"]),  # 记录当前二维单元总数。
            "h_local_mm": float(row["h_local"]),  # 记录端部附近特征网格尺寸。
            "local_peak_mpa": float(row["peak_stress"]),  # 记录端部附近随网格变化的局部峰值。
            "strain_energy_n_mm": float(row["strain_energy"]),  # 记录未派生专用量的全模型应变能。
            "energy_balance_relative": float(row["energy_balance_rel"]),  # 记录原模型与微扰模型中的最大平衡误差。
        })  # 完成当前网格层级的可见证据。
    case = {  # 组织不包含目标类别词、专用派生量或既有诊断的匿名个案。
        "sample_id": HIDDEN_SAMPLE_ID,  # 提供不表达类别含义的匿名样本编号。
        "user_text": HIDDEN_USER_TEXT,  # 只用几何和连接事实表达自然语言问题。
        "model_facts": {  # 组织可核验的模型事实。
            "analysis": "二维平面应力线弹性静力分析",  # 记录当前求解假设。
            "domain_mm": {"width": 200.0, "height": 200.0, "thickness": 10.0},  # 记录板域宽、高和厚度毫米值。
            "material": {"young_mpa": 210000.0, "poisson": 0.30, "plastic_curve": None},  # 记录线弹性材料并明确没有塑性曲线。
            "loading": {"type": "上下边界对称均布拉伸", "nominal_stress_mpa": 100.0},  # 记录远场名义拉应力。
            "feature_geometry": {"type": "板内零厚度直线型边界对", "total_length_mm": 40.0, "endpoint_radius_mm": 0.0},  # 只陈述内部边界对的几何事实。
            "local_topology": {"coincident_opposed_boundary_chain_count": 2, "both_chains_are_traction_free": True, "interior_nodes_are_duplicated_across_pair": True, "interior_nodes_are_shared_across_pair": False, "endpoint_nodes_are_shared_between_chains": True, "connectivity_discontinuity_between_endpoints": True, "chains_meet_at_zero_radius_endpoints": True},  # 明确边界内部两张相对自由面不共享节点而两个端点闭合。
        },  # 完成内部不连续样本的模型事实。
        "mesh_history": mesh_history,  # 附加三档真实求解的通用原始趋势。
    }  # 完成内部不连续匿名个案构造。
    serialized = _canonical_json(case).lower()  # 序列化整个运行时个案以检查关键词泄漏。
    leaked = [term for term in HIDDEN_FORBIDDEN_TERMS if term.lower() in serialized]  # 找出任何意外出现的目标类别或专用量词汇。
    if leaked:  # 在 DeepSeek 调用前检查隐藏样本是否仍然盲化。
        raise ValueError(f"hidden sample leaks forbidden terms: {leaked}")  # 拒绝发送带目标词泄漏的隐藏样本。
    return case  # 返回通过词汇盲化检查的内部不连续个案。


def _build_cases(seed: int) -> list[dict[str, Any]]:  # 构造两个真实求解个案并以固定种子打乱匿名顺序。
    misleading_case = _continuous_geometry_case()  # 先执行含误导词的连续几何真实求解。
    hidden_case = _internal_discontinuity_case()  # 再执行无目标词的内部不连续真实求解。
    if "裂纹" not in str(misleading_case["user_text"]):  # 验证误导样本确实包含待检验的关键词。
        raise ValueError("misleading sample must contain the intended keyword")  # 拒绝失去反事实反转条件的样本。
    cases = [misleading_case, hidden_case]  # 把两个匿名个案放入同一顺序池。
    random.Random(seed).shuffle(cases)  # 用局部固定随机数生成器打乱提交顺序且不污染全局状态。
    return cases  # 返回只包含模型可见事实的两个匿名个案。


def _audit_runtime_messages(common_messages: list[dict[str, str]], cases: list[dict[str, Any]]) -> None:  # 在 API 调用前审计最终消息的盲化与独立性。
    common_serialized = _canonical_json(common_messages).lower()  # 序列化真正发送的公共前缀以检查评分答案泄漏。
    forbidden_markers = ("expected_route", "oracle", "positive", "negative", "counterfactual", "pass", "fail")  # 定义不应出现在模型消息中的实验角色或评分标记。
    leaked_markers = [marker for marker in forbidden_markers if marker in common_serialized]  # 搜索公共协议中的事后评分语义。
    if leaked_markers:  # 检查公共前缀是否意外暴露实验设计。
        raise ValueError(f"common prefix leaks posthoc markers: {leaked_markers}")  # 在任何付费调用前拒绝带答案暗示的协议。
    hidden_cases = [case for case in cases if case["sample_id"] == HIDDEN_SAMPLE_ID]  # 从随机顺序中定位无目标词匿名样本。
    if len(hidden_cases) != 1:  # 检查隐藏样本编号是否唯一且未丢失。
        raise ValueError("exactly one hidden sample is required")  # 拒绝不满足双样本设计的运行包。
    hidden_messages = [dict(message) for message in common_messages] + [{"role": "user", "content": "当前匿名个案如下：\n" + _canonical_json(hidden_cases[0])}]  # 重建隐藏样本将要发送的完整 messages。
    hidden_serialized = _canonical_json(hidden_messages).lower()  # 序列化完整请求以检查协议与个案共同造成的词汇泄漏。
    leaked_terms = [term for term in HIDDEN_FORBIDDEN_TERMS if term.lower() in hidden_serialized]  # 搜索完整隐藏请求中的目标类别与专用量词汇。
    if leaked_terms:  # 检查隐藏样本的最终请求是否仍然盲化。
        raise ValueError(f"hidden request leaks forbidden terms: {leaked_terms}")  # 在任何付费调用前拒绝泄漏目标词的请求。


def _validate_decision(value: Any) -> list[str]:  # 只验证模型输出结构而不在运行时判断物理路线是否正确。
    errors: list[str] = []  # 初始化结构错误列表。
    if not isinstance(value, dict):  # 检查顶层响应是否为 JSON 对象。
        return ["response must be a JSON object"]  # 顶层类型错误时立即返回。
    if value.get("route_id") not in ROUTE_CATALOG:  # 检查路线编号是否属于共用目录。
        errors.append("route_id must name one registered route")  # 记录无效路线编号。
    if value.get("selected_tool_id") not in TOOL_CATALOG:  # 检查工具编号是否属于共用目录。
        errors.append("selected_tool_id must name one registered tool")  # 记录无效工具编号。
    hypotheses = value.get("competing_hypotheses")  # 读取竞争假设数组。
    if not isinstance(hypotheses, list) or len(hypotheses) < 2 or not all(isinstance(item, str) and item.strip() for item in hypotheses):  # 检查至少两个非空字符串假设。
        errors.append("competing_hypotheses must contain at least two strings")  # 记录竞争假设结构错误。
    evidence_refs = value.get("evidence_refs")  # 读取证据字段路径数组。
    if not isinstance(evidence_refs, list) or not evidence_refs or not all(isinstance(item, str) and item.strip() for item in evidence_refs):  # 检查至少一个非空证据引用。
        errors.append("evidence_refs must contain at least one string")  # 记录证据引用结构错误。
    if not isinstance(value.get("uncertainties"), list):  # 检查不确定性字段是否为数组。
        errors.append("uncertainties must be an array")  # 记录不确定性字段错误。
    if not isinstance(value.get("engineering_reason"), str) or not value.get("engineering_reason", "").strip():  # 检查公开工程理由是否为非空字符串。
        errors.append("engineering_reason must be a non-empty string")  # 记录工程理由字段错误。
    if not isinstance(value.get("provisional_answer"), str) or not value.get("provisional_answer", "").strip():  # 检查暂定回答是否为非空字符串。
        errors.append("provisional_answer must be a non-empty string")  # 记录暂定回答字段错误。
    confidence = value.get("confidence")  # 读取模型给出的置信度。
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0.0 <= float(confidence) <= 1.0:  # 检查置信度类型和零到一范围。
        errors.append("confidence must be a number between zero and one")  # 记录置信度字段错误。
    return errors  # 返回与路线正确性无关的纯结构错误列表。


def _usage_summary(usage: dict[str, Any]) -> dict[str, int]:  # 从 DeepSeek 兼容 usage 中提取费用与缓存关键计数。
    details = usage.get("prompt_tokens_details", {}) if isinstance(usage.get("prompt_tokens_details", {}), dict) else {}  # 兼容 OpenAI 风格的嵌套输入明细。
    cache_hit = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0))  # 优先读取 DeepSeek 顶层缓存命中字段。
    cache_miss = usage.get("prompt_cache_miss_tokens", 0)  # 读取 DeepSeek 顶层缓存未命中字段。
    return {  # 返回统一整数口径的 token 统计。
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),  # 记录总输入 token 数。
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),  # 记录总输出 token 数。
        "total_tokens": int(usage.get("total_tokens", 0) or 0),  # 记录输入与输出 token 总数。
        "prompt_cache_hit_tokens": int(cache_hit or 0),  # 记录公共前缀缓存命中 token 数。
        "prompt_cache_miss_tokens": int(cache_miss or 0),  # 记录本次未命中缓存的输入 token 数。
    }  # 完成统一 usage 摘要。


def _call_case(client: OpenAI, model: str, common_messages: list[dict[str, str]], case: dict[str, Any], request_index: int) -> dict[str, Any]:  # 对一个匿名个案发起且只发起一次 DeepSeek 请求。
    if request_index < 1 or request_index > MAX_HTTP_REQUESTS:  # 在网络调用前校验显式请求编号不超过硬上限。
        raise RuntimeError("request index exceeds the two-request hard limit")  # 拒绝任何第三次或更多请求。
    case_prompt = "当前匿名个案如下：\n" + _canonical_json(case)  # 把唯一变化的个案放在两条公共消息之后。
    messages = [dict(message) for message in common_messages] + [{"role": "user", "content": case_prompt}]  # 为本个案创建独立消息历史且不包含另一例回答。
    started_at = datetime.now(timezone.utc).isoformat()  # 记录调用时间但不把时间戳写入可缓存提示词。
    try:  # 捕获单次网络或服务错误并冻结为实验结果。
        response = client.chat.completions.create(  # 发起本个案唯一一次实际 DeepSeek 调用。
            model=model,  # 使用命令行或环境变量选择的已验证模型。
            messages=messages,  # 发送相同公共前缀和当前匿名个案。
            response_format={"type": "json_object"},  # 要求服务返回 JSON 对象以便无重试冻结。
            max_tokens=MAX_COMPLETION_TOKENS,  # 限制本次回答的最大输出预算。
            reasoning_effort="medium",  # 使用中等推理预算兼顾分类能力与费用。
            extra_body={"thinking": {"type": "enabled"}},  # 显式启用模型支持的推理模式但不保存隐藏思维链。
            temperature=0.0,  # 使用确定性较高的采样设置减少无关波动。
            stream=False,  # 使用一次性响应以便完整记录 usage 和结束原因。
        )  # 完成本个案唯一网络请求。
    except Exception as exc:  # 捕获 SDK 或服务端异常且绝不自动重试。
        return {"sample_id": case["sample_id"], "request_index": request_index, "started_at_utc": started_at, "transport_error": {"type": type(exc).__name__, "message": str(exc)}, "raw_content": "", "decision": None, "validation_errors": ["transport error"], "metadata": {}}  # 冻结失败信息供 artifact 审计并允许第二个独立样本继续。
    content = response.choices[0].message.content or ""  # 读取公开回答文本而不读取隐藏推理内容。
    usage = response.usage.model_dump() if response.usage is not None else {}  # 保存服务返回的完整 usage 字段。
    try:  # 尝试把公开回答解析为 JSON。
        decision = json.loads(content)  # 解析模型公开决策且不进行任何修复请求。
        parsing_error = ""  # 标记 JSON 解析成功。
    except json.JSONDecodeError as exc:  # 捕获非法 JSON 并把它当作一次冻结结果。
        decision = None  # 记录没有可评分的结构化决策。
        parsing_error = str(exc)  # 保存解析失败位置供离线审计。
    validation_errors = _validate_decision(decision) if decision is not None else [f"invalid JSON: {parsing_error}"]  # 对有效 JSON 只做结构校验且不做路线门禁。
    metadata = {  # 组织不含凭据和隐藏思维链的调用元数据。
        "requested_model": model,  # 记录请求中的模型名称。
        "response_model": response.model,  # 记录服务实际返回的模型名称。
        "finish_reason": response.choices[0].finish_reason,  # 记录模型结束原因以识别截断。
        "usage": usage,  # 保存 API 返回的原始 token 使用详情。
        "usage_summary": _usage_summary(usage),  # 保存便于比较缓存命中的统一摘要。
    }  # 完成调用元数据组织。
    return {"sample_id": case["sample_id"], "request_index": request_index, "started_at_utc": started_at, "transport_error": None, "raw_content": content, "decision": decision, "validation_errors": validation_errors, "metadata": metadata}  # 返回单次请求的完整冻结记录。


def _score_result(result: dict[str, Any]) -> dict[str, Any]:  # 在两次回答完成后按隐藏映射执行确定性离线评分。
    sample_id = str(result["sample_id"])  # 读取匿名样本编号以查询未发送给模型的评分映射。
    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}  # 读取可评分决策或使用空对象。
    route_id = decision.get("route_id")  # 读取模型选择的路线编号。
    tool_id = decision.get("selected_tool_id")  # 读取模型选择的下一步工具编号。
    refs = decision.get("evidence_refs", []) if isinstance(decision.get("evidence_refs", []), list) else []  # 读取证据字段路径列表。
    normalized_refs = [str(item).lower() for item in refs]  # 统一证据路径大小写以执行宽松覆盖检查。
    return {  # 返回不影响运行时模型决策的事后评分项。
        "expected_route_id": POSTHOC_EXPECTED_ROUTES[sample_id],  # 记录该匿名样本的隐藏期望路线。
        "selected_route_id": route_id,  # 记录模型实际选择路线。
        "route_match": route_id == POSTHOC_EXPECTED_ROUTES[sample_id],  # 检查模型是否按物理事实而非词面选择路线。
        "selected_tool_id": tool_id,  # 记录模型实际选择工具。
        "tool_is_compatible": tool_id in POSTHOC_ACCEPTABLE_TOOLS[sample_id],  # 单独报告动作是否落入宽松适配集合。
        "cites_topology": any("local_topology" in ref for ref in normalized_refs),  # 检查理由是否引用局部拓扑事实路径。
        "cites_mesh_history": any("mesh_history" in ref for ref in normalized_refs),  # 检查理由是否引用网格历史事实路径。
        "structure_valid": not result.get("validation_errors"),  # 检查公开回答是否满足共用 JSON 合同。
    }  # 完成单个匿名样本的离线评分。


def _write_json(path: Path, value: dict[str, Any]) -> None:  # 把当前实验状态原子性不足但可读地写入 artifact 路径。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建结果目录以支持 GitHub artifact 上传。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 使用 UTF-8 和缩进保存可人工复核的 JSON。


def _partial_output(model: str, seed: int, common_messages: list[dict[str, str]], cases: list[dict[str, Any]], results: list[dict[str, Any]], started_at: str) -> dict[str, Any]:  # 在请求间保存不含事后评分的恢复记录。
    common_serialized = _canonical_json(common_messages)  # 序列化真正发送的公共消息以计算摘要。
    return {  # 组织可在中途失败时保留首个回答的检查点。
        "experiment": "deepseek_routing_counterfactual_pair",  # 标识独立反事实实验名称。
        "status": "in_progress",  # 标记当前仍可能缺少第二个匿名样本。
        "started_at_utc": started_at,  # 记录实验开始时间。
        "model": model,  # 记录请求模型名称。
        "seed": seed,  # 记录匿名顺序随机种子。
        "client_creations": 1,  # 明确整个实验只创建一个 API 客户端。
        "http_request_hard_limit": MAX_HTTP_REQUESTS,  # 明确网络请求硬上限为两次。
        "http_requests_attempted": len(results),  # 记录当前已经尝试的请求数量。
        "sdk_max_retries": 0,  # 明确 SDK 隐式重试已关闭。
        "common_prefix_sha256": hashlib.sha256(common_serialized.encode("utf-8")).hexdigest(),  # 记录两次请求逐字节共用前缀的摘要。
        "common_messages": common_messages,  # 冻结两次请求共用的实际系统与协议消息。
        "case_order": [case["sample_id"] for case in cases],  # 记录匿名样本实际提交顺序。
        "case_packets": cases,  # 冻结模型实际看到且已去除既有诊断的两个证据包。
        "results": results,  # 保存当前已经完成的原始 DeepSeek 回答。
    }  # 完成中途检查点组织。


def _final_output(partial: dict[str, Any]) -> dict[str, Any]:  # 在两次请求完成后附加隐藏评分与缓存汇总。
    results = partial["results"]  # 读取两个已经冻结且相互独立的回答记录。
    scores = [_score_result(result) for result in results]  # 对两个回答执行不参与提示词的离线评分。
    selected_routes = [score["selected_route_id"] for score in scores]  # 提取模型为两个反事实样本选择的路线。
    aggregate_pass = len(scores) == 2 and all(score["route_match"] for score in scores) and len(set(selected_routes)) == 2  # 只有两例路线都正确且确实不同时才算反事实主指标通过。
    usage_rows = [result.get("metadata", {}).get("usage_summary", {}) for result in results]  # 提取两个调用的统一 token 摘要。
    total_cache_hit = sum(int(row.get("prompt_cache_hit_tokens", 0)) for row in usage_rows)  # 汇总两次请求的缓存命中 token。
    total_cache_miss = sum(int(row.get("prompt_cache_miss_tokens", 0)) for row in usage_rows)  # 汇总两次请求的缓存未命中 token。
    cache_denominator = total_cache_hit + total_cache_miss  # 计算 DeepSeek 缓存统计字段的有效分母。
    transport_ok = all(result.get("transport_error") is None for result in results)  # 检查两次请求是否都获得服务响应。
    output = dict(partial)  # 复制中途记录以保留完整请求与原始回答。
    output.update({  # 附加只在两个回答冻结后计算的最终审计字段。
        "status": "completed" if transport_ok else "completed_with_transport_error",  # 区分完整响应和传输层失败。
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),  # 记录最终冻结时间。
        "http_requests_attempted": len(results),  # 确认最终实际尝试请求数。
        "posthoc_scores": scores,  # 保存隐藏映射产生的逐例评分。
        "aggregate_counterfactual_pass": aggregate_pass,  # 保存不受工具选择影响的路线反事实主指标。
        "routes_are_different": len(selected_routes) == 2 and len(set(selected_routes)) == 2,  # 单独报告模型是否对事实相反的两例给出不同路线。
        "cache_audit": {"prompt_cache_hit_tokens": total_cache_hit, "prompt_cache_miss_tokens": total_cache_miss, "hit_ratio": (total_cache_hit / cache_denominator) if cache_denominator else None},  # 汇总公共前缀缓存命中情况且缓存失败不触发重试。
    })  # 完成最终字段合并。
    return output  # 返回完整独立反事实实验记录。


def main() -> int:  # 解析参数、执行真实有限元证据生成并严格发起两次模型请求。
    parser = argparse.ArgumentParser()  # 创建独立反事实实验命令行解析器。
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))  # 允许 ds Environment 覆盖默认模型名称。
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)  # 允许显式复现或改变匿名样本顺序。
    parser.add_argument("--output", type=Path, default=Path("artifacts/deepseek_routing_counterfactual_pair.json"))  # 设置可由 GitHub Actions 上传的结果文件。
    args = parser.parse_args()  # 读取并冻结命令行参数。
    api_key = os.environ.get("DEEPSEEK_API_KEY")  # 读取 ds Environment 注入的 DeepSeek 凭据。
    if not api_key:  # 在任何真实求解和网络调用前检查凭据存在性。
        raise RuntimeError("DEEPSEEK_API_KEY is required for the live counterfactual pair")  # 拒绝生成伪造模型结果。
    started_at = datetime.now(timezone.utc).isoformat()  # 记录实验开始时间但不发送给模型。
    cases = _build_cases(args.seed)  # 执行两个真实有限元场景并生成盲化证据包。
    common_messages = _common_messages()  # 只构造一次两请求共用的逐字节前缀。
    _audit_runtime_messages(common_messages, cases)  # 在创建客户端前审计最终请求不含评分答案或隐藏目标词。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", max_retries=0, timeout=API_TIMEOUT_SECONDS)  # 创建唯一客户端并关闭 SDK 默认隐式重试。
    results: list[dict[str, Any]] = []  # 初始化最多容纳两个单次回答的结果列表。
    print(json.dumps({"status": "requesting", "request_index": 1, "sample_id": cases[0]["sample_id"]}, ensure_ascii=False))  # 在日志中报告第一个匿名请求但不输出凭据或答案。
    first_result = _call_case(client, args.model, common_messages, cases[0], 1)  # 对第一个匿名样本执行唯一一次请求。
    results.append(first_result)  # 冻结第一个回答以防第二次请求发生传输故障。
    _write_json(args.output, _partial_output(args.model, args.seed, common_messages, cases, results, started_at))  # 在第二次请求前保存首个回答检查点。
    print(json.dumps({"status": "requesting", "request_index": 2, "sample_id": cases[1]["sample_id"]}, ensure_ascii=False))  # 在日志中报告第二个匿名请求但不输出首个回答。
    second_result = _call_case(client, args.model, common_messages, cases[1], 2)  # 对第二个匿名样本执行唯一一次且独立的请求。
    results.append(second_result)  # 冻结第二个回答并达到请求硬上限。
    partial = _partial_output(args.model, args.seed, common_messages, cases, results, started_at)  # 组织包含两个原始回答的完整运行记录。
    final = _final_output(partial)  # 仅在两次回答完成后执行隐藏路线评分与缓存汇总。
    _write_json(args.output, final)  # 保存供 GitHub artifact 上传的最终 JSON。
    summary = {"status": final["status"], "output": str(args.output), "http_requests_attempted": final["http_requests_attempted"], "routes": {row["sample_id"]: row["decision"].get("route_id") if isinstance(row.get("decision"), dict) else None for row in results}, "aggregate_counterfactual_pass": final["aggregate_counterfactual_pass"], "cache_audit": final["cache_audit"]}  # 组织不含隐藏推理的简短日志摘要。
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # 把实际路线与缓存结果写入 GitHub Actions 日志。
    return 0 if all(result.get("transport_error") is None for result in results) else 2  # 仅在传输层失败时返回非零且不因模型判断错误触发重跑。


if __name__ == "__main__":  # 仅在脚本直接执行时启动独立反事实实验。
    raise SystemExit(main())  # 把主函数状态码返回给 GitHub Actions。
