# 启用延迟类型注解，便于声明 JSON 结构而不引入运行时开销。
from __future__ import annotations

# 导入哈希模块，用于固定每个 agent trace 的来源摘要。
import hashlib
# 导入 JSON 模块，用于读取 trace 并构造论文来源收据。
import json
# 导入路径类型，用于定位四个 case 的运行目录与 paper.tex。
from pathlib import Path
# 导入通用类型，用于描述 DeepSeek 决策 JSON。
from typing import Any


# 固定论文决策来源收据格式版本。
PROVENANCE_SCHEMA_VERSION = "deepseek-decision-paper-provenance/1.0"
# 固定插入论文的动态章节标签，便于机器检查章节确实存在。
DECISION_SECTION_LABEL = "sec:deepseek-decisions"
# 固定由补丁写入 LaTeX 模板的唯一插槽，避免依赖通用章节标题定位。
DECISION_INSERTION_SENTINEL = "% DEEPSEEK_FINAL_DECISIONS"


# 定义 UTF-8 JSON 读取函数。
def _load_json(path: Path) -> dict[str, Any]:
    """读取一个必须为对象的 JSON 文件。"""
    # 解析 UTF-8 文件内容。
    payload = json.loads(path.read_text(encoding="utf-8"))
    # 要求顶层是对象，避免数组或标量破坏后续字段访问。
    if not isinstance(payload, dict):
        # 抛出包含路径的明确错误。
        raise ValueError(f"expected JSON object in {path}")
    # 返回通过顶层类型校验的对象。
    return payload


# 定义 LaTeX 文本转义函数，允许原样写入 DeepSeek 中文决策。
def _escape_tex(value: Any) -> str:
    """转义 LaTeX 特殊字符，同时保留中文和普通标点。"""
    # 将任意值转换为字符串并去除两端空白。
    text = str(value if value is not None else "").strip()
    # 将换行折叠为空格，避免模型正文无意创建 LaTeX 段落结构。
    text = " ".join(text.splitlines())
    # 建立逐字符映射，避免连续 replace 再次转义命令自身的花括号。
    replacements = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}
    # 单次遍历原始正文并返回可安全嵌入 LaTeX 的文本。
    return "".join(replacements.get(character, character) for character in text)


# 定义字符串列表归一化函数。
def _string_list(value: Any) -> list[str]:
    """把 JSON 字段归一化为去空白的非空字符串列表。"""
    # 只有列表类型才逐项处理。
    if isinstance(value, list):
        # 返回所有非空字符串化项目。
        return [str(item).strip() for item in value if str(item).strip()]
    # 单个非空字符串也转为单元素列表，兼容模型轻微格式漂移。
    if isinstance(value, str) and value.strip():
        # 返回单元素列表。
        return [value.strip()]
    # 其他类型返回空列表并由来源 gate 报错。
    return []


# 定义列表到 LaTeX itemize 的转换函数。
def _itemize(items: list[str], empty_text: str) -> str:
    """把项目列表转换为紧凑 LaTeX 列表。"""
    # 没有项目时返回显式缺失说明，避免生成空环境。
    if not items:
        # 返回经过转义的缺失说明。
        return _escape_tex(empty_text)
    # 初始化紧凑列表开始标记。
    lines = ["\\begin{itemize}[leftmargin=*,nosep]"]
    # 顺序写入每个来源项目。
    for item in items:
        # 追加经过 LaTeX 转义的项目。
        lines.append("\\item " + _escape_tex(item))
    # 追加列表结束标记。
    lines.append("\\end{itemize}")
    # 用换行拼接完整 LaTeX 片段。
    return "\n".join(lines)


# 定义单个 case 的 DeepSeek 来源提取函数。
def _collect_case(run_dir: Path, case_id: str) -> tuple[dict[str, Any], list[str]]:
    """从 agent_trace 提取最终决策、逐轮 Skill 链和证据引用。"""
    # 定位本场景唯一 trace 文件。
    trace_path = run_dir / "agent_trace.json"
    # 读取完整 trace。
    trace = _load_json(trace_path)
    # 初始化本场景来源错误。
    errors: list[str] = []
    # 读取最终决策对象。
    final_decision = trace.get("final_decision")
    # 最终决策缺失或不是对象时记录错误并使用空对象继续汇总。
    if not isinstance(final_decision, dict):
        # 记录可定位的 case 错误。
        errors.append(f"{case_id}: final_decision is missing")
        # 使用空对象避免后续字段访问异常。
        final_decision = {}
    # 读取最终直接回答。
    answer = str(final_decision.get("answer") or "").strip()
    # 最终回答为空时记录错误。
    if not answer:
        # 记录缺少 answer 字段。
        errors.append(f"{case_id}: final_decision.answer is empty")
    # 归一化已实施动作。
    implemented_actions = _string_list(final_decision.get("implemented_actions"))
    # 已实施动作为空时记录错误。
    if not implemented_actions:
        # 记录缺少实施动作。
        errors.append(f"{case_id}: final_decision.implemented_actions is empty")
    # 归一化可用范围。
    can_use = _string_list(final_decision.get("can_use"))
    # 可用范围为空时记录错误。
    if not can_use:
        # 记录缺少可用范围。
        errors.append(f"{case_id}: final_decision.can_use is empty")
    # 归一化不可用范围。
    cannot_use = _string_list(final_decision.get("cannot_use"))
    # 不可用范围为空时记录错误。
    if not cannot_use:
        # 记录缺少不可用范围。
        errors.append(f"{case_id}: final_decision.cannot_use is empty")
    # 归一化最终决策引用的 artifact ID。
    evidence_refs = _string_list(final_decision.get("evidence_refs"))
    # 证据引用为空时记录错误。
    if not evidence_refs:
        # 记录缺少证据引用。
        errors.append(f"{case_id}: final_decision.evidence_refs is empty")
    # 建立 trace 中真实 artifact ID 集合。
    artifact_ids = {str(artifact.get("artifact_id")) for artifact in trace.get("artifacts", []) if isinstance(artifact, dict) and artifact.get("artifact_id")}
    # 计算引用但不存在的 artifact。
    missing_refs = sorted(set(evidence_refs) - artifact_ids)
    # 有无效引用时记录错误。
    if missing_refs:
        # 记录全部无效 ID。
        errors.append(f"{case_id}: final_decision references missing artifacts {missing_refs}")
    # 选择逐轮模型决策事件。
    model_events = [event for event in trace.get("events", []) if isinstance(event, dict) and event.get("type") == "model_decision"]
    # 没有模型决策事件时记录错误。
    if not model_events:
        # 记录缺少在线模型决策。
        errors.append(f"{case_id}: no model_decision events")
    # 提取所有模型事件的 provider 名称。
    providers = [event.get("provider_metadata", {}).get("provider") for event in model_events]
    # 任一事件不是 DeepSeek 时记录错误。
    if providers and any(provider != "deepseek" for provider in providers):
        # 记录实际 provider 序列。
        errors.append(f"{case_id}: non-DeepSeek provider metadata {providers}")
    # 选择所有实际 Skill 调用事件。
    skill_events = [event for event in trace.get("events", []) if isinstance(event, dict) and event.get("type") == "skill_call"]
    # 构造紧凑逐轮 Skill 链。
    skill_chain = []
    # 遍历 Skill 事件并保留轮次、名称、理由和结果。
    for event in skill_events:
        # 读取原始 Skill 请求对象。
        request = event.get("request") if isinstance(event.get("request"), dict) else {}
        # 读取 Skill 返回对象。
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        # 追加可审计 Skill 链条目。
        skill_chain.append({"iteration": event.get("iteration"), "skill": request.get("skill"), "why": request.get("why"), "status": result.get("status"), "summary": result.get("summary")})
    # 计算 trace 原始字节的 SHA-256。
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    # 构造本场景论文来源记录。
    record = {
        # 保存 case ID。
        "case_id": case_id,
        # 保存普通工程师原始问题。
        "question": str(trace.get("question") or "").strip(),
        # 保存 trace 状态。
        "status": trace.get("status"),
        # 保存最终直接回答。
        "answer": answer,
        # 保存已实施动作。
        "implemented_actions": implemented_actions,
        # 保存可用范围。
        "can_use": can_use,
        # 保存不可用范围。
        "cannot_use": cannot_use,
        # 保存证据引用。
        "evidence_refs": evidence_refs,
        # 保存可选后续动作。
        "follow_up": str(final_decision.get("follow_up") or "").strip(),
        # 保存在线 DeepSeek 决策事件数量。
        "model_decision_count": len(model_events),
        # 保存执行 Skill 链。
        "skill_chain": skill_chain,
        # 保存 provider 序列。
        "providers": providers,
        # 保存 trace 来源相对路径。
        "trace_path": str(trace_path),
        # 保存 trace 内容哈希。
        "trace_sha256": trace_sha256,
    }
    # 返回来源记录和错误列表。
    return record, errors


# 定义单个 case 到论文动态小节的转换函数。
def _case_section(record: dict[str, Any], title: str) -> str:
    """把最终 DeepSeek 决策及用途边界转换为论文小节。"""
    # 从 Skill 链提取已完成或已尝试的 Skill 名称。
    skill_names = [str(item.get("skill")) for item in record.get("skill_chain", []) if item.get("skill")]
    # 构造去重且保持执行顺序的 Skill 名称列表。
    ordered_skill_names = list(dict.fromkeys(skill_names))
    # 构造章节行列表。
    lines = ["\\subsection{" + _escape_tex(title) + "}"]
    # 写入原始工程问题。
    lines.append("\\textbf{Engineer question.} " + _escape_tex(record.get("question")))
    # 写入本场景决策次数及实际 Skill 链。
    lines.append("\\textbf{Audited decision path.} DeepSeek issued " + str(record.get("model_decision_count", 0)) + " model decisions. The executed Skill sequence was: " + _escape_tex(" -> ".join(ordered_skill_names)) + ".")
    # 写入最终回答，明确它来自 trace 而不是模板作者。
    lines.append("\\textbf{Verbatim final DeepSeek answer.} " + _escape_tex(record.get("answer")))
    # 写入已实施动作标题。
    lines.append("\\textbf{Implemented actions recorded by DeepSeek.}")
    # 写入已实施动作列表。
    lines.append(_itemize(record.get("implemented_actions", []), "No implemented action was recorded."))
    # 写入可用范围标题。
    lines.append("\\textbf{Supported uses recorded by DeepSeek.}")
    # 写入可用范围列表。
    lines.append(_itemize(record.get("can_use", []), "No supported use was recorded."))
    # 写入不可用范围标题。
    lines.append("\\textbf{Excluded uses recorded by DeepSeek.}")
    # 写入不可用范围列表。
    lines.append(_itemize(record.get("cannot_use", []), "No excluded use was recorded."))
    # 写入证据引用，使用等宽字体突出机器可查 ID。
    lines.append("\\textbf{Evidence references.} \\texttt{" + _escape_tex(", ".join(record.get("evidence_refs", []))) + "}.")
    # 有后续动作时写入跟进建议。
    if record.get("follow_up"):
        # 写入经过转义的后续动作。
        lines.append("\\textbf{Follow-up recorded by DeepSeek.} " + _escape_tex(record.get("follow_up")))
    # 用空行连接段落，保持双栏论文可读性。
    return "\n\n".join(lines)


# 定义主注入函数，供 payload 中的 Elsevier builder 调用。
def inject_deepseek_decisions(paper_path: Path, suite_dir: Path, case_order: list[str], case_titles: dict[str, str]) -> dict[str, Any]:
    """把四个最终 DeepSeek 决策确定性注入 paper.tex，并返回来源收据。"""
    # 初始化本次论文的四个来源记录。
    case_records: list[dict[str, Any]] = []
    # 初始化全部来源错误。
    errors: list[str] = []
    # 按固定实验顺序读取四个 case。
    for case_id in case_order:
        # 定位场景运行目录。
        run_dir = suite_dir / "runs" / case_id
        # 提取最终决策与逐轮 Skill 链。
        record, case_errors = _collect_case(run_dir, case_id)
        # 追加本场景记录。
        case_records.append(record)
        # 追加本场景错误。
        errors.extend(case_errors)
    # 读取现有 Elsevier LaTeX。
    paper_text = paper_path.read_text(encoding="utf-8")
    # 统计补丁写入的唯一决策章节插槽。
    sentinel_count = paper_text.count(DECISION_INSERTION_SENTINEL)
    # 插槽不是恰好一个时记录错误，避免重复或错位注入。
    if sentinel_count != 1:
        # 记录实际插槽数量供构建日志定位。
        errors.append(f"paper.tex contains {sentinel_count} DeepSeek decision insertion sentinels")
    # 构造动态章节标题与来源说明。
    section_lines = ["\\section{DeepSeek decision-derived engineering findings}", "\\label{" + DECISION_SECTION_LABEL + "}", "This section is generated deterministically from each live \\texttt{agent\\_trace.json}. The wording under ``Verbatim final DeepSeek answer'', the implemented actions, supported uses, excluded uses, and evidence references are not paper-template prose; they are the final decisions produced by the same DeepSeek-controlled run that generated the numerical artifacts."]
    # 为每个场景追加决策来源小节。
    for record in case_records:
        # 使用固定英文标题并写入动态内容。
        section_lines.append(_case_section(record, case_titles.get(record["case_id"], record["case_id"])))
    # 拼接完整动态章节。
    decision_section = "\n\n".join(section_lines) + "\n\n"
    # 只有唯一插槽存在时执行一次替换。
    if sentinel_count == 1:
        # 用动态来源章节替换显式插槽。
        paper_text = paper_text.replace(DECISION_INSERTION_SENTINEL, decision_section, 1)
    # 定义 XeCJK 包锚点，使中文 DeepSeek 原文能够正确渲染。
    package_marker = "\\usepackage{lineno}"
    # 构造中文字体配置；workflow 已安装 Noto CJK 字体。
    cjk_packages = "\\usepackage{lineno}\n\\usepackage{xeCJK}\n\\setCJKmainfont{Noto Serif CJK SC}\n\\setCJKsansfont{Noto Sans CJK SC}"
    # 缺少包锚点时记录错误。
    if package_marker not in paper_text:
        # 记录无法配置中文字体。
        errors.append("paper.tex does not contain the lineno package marker")
    # 只有尚未配置 xeCJK 时才执行替换，保证函数幂等。
    elif "\\usepackage{xeCJK}" not in paper_text:
        # 注入中文字体包与固定字体。
        paper_text = paper_text.replace(package_marker, cjk_packages, 1)
    # 定义摘要结束锚点。
    abstract_marker = "\\end{abstract}"
    # 构造论文来源声明，明确动态结论来自 DeepSeek 最终决策。
    provenance_sentence = " The case-specific engineering findings and use boundaries are populated directly from the final DeepSeek decisions and their evidence references, rather than authored as fixed report-template conclusions."
    # 摘要锚点存在且尚未注入时添加来源声明。
    if abstract_marker in paper_text and provenance_sentence.strip() not in paper_text:
        # 在摘要结束前插入来源声明。
        paper_text = paper_text.replace(abstract_marker, provenance_sentence + "\n" + abstract_marker, 1)
    # 摘要锚点缺失时记录错误。
    elif abstract_marker not in paper_text:
        # 记录无法添加来源声明。
        errors.append("paper.tex does not contain the abstract end marker")
    # 写回已注入决策来源的 LaTeX。
    paper_path.write_text(paper_text, encoding="utf-8")
    # 重新读取文本并检查动态章节标签存在。
    inserted = "\\label{" + DECISION_SECTION_LABEL + "}" in paper_path.read_text(encoding="utf-8")
    # 动态章节未出现时记录错误。
    if not inserted:
        # 记录注入失败。
        errors.append("DeepSeek decision-derived section was not inserted")
    # 构造机器可读来源收据。
    return {
        # 标注来源收据格式版本。
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        # 只有无错误且章节已插入时才通过。
        "valid": not errors and inserted,
        # 保存全部来源或结构错误。
        "errors": errors,
        # 标注论文正文完全由既有 trace 后处理，未产生额外模型调用。
        "additional_deepseek_calls_for_paper": 0,
        # 标注动态章节标签。
        "decision_section_label": DECISION_SECTION_LABEL,
        # 标注场景数量。
        "scenario_count": len(case_records),
        # 保存四个场景的完整决策来源记录。
        "cases": case_records,
    }
