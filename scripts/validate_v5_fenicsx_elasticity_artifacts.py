#!/usr/bin/env python3
# 本脚本使用 Python 标准库在宿主 Actions runner 上独立校验 FEniCSx 结构演示 artifact。
"""严格验证结构模拟回执、分析门、解验证边界以及全部记录文件的大小和 SHA-256。"""  # 明确本脚本不导入也不信任数值 runner 的实现逻辑。
from __future__ import annotations  # 启用现代类型注解语义并避免运行时提前解析复合注解。

import argparse  # 解析唯一必填的结构模拟 artifact 根目录参数。
import hashlib  # 独立重算 JSON、PNG、XDMF 和 HDF5 文件的 SHA-256。
import json  # 严格读取模拟与工程工件并写出独立验证回执。
import sys  # 输出 Actions 机器摘要并返回零或一进程退出码。
import xml.etree.ElementTree as ElementTree  # 使用标准库解析 XDMF XML 以拒绝伪装文本文件。
from datetime import datetime, timezone  # 记录带显式 UTC 时区的验证起止时间。
from pathlib import Path  # 解析 artifact 文件并防止记录路径逃逸 artifact 根。
from typing import Any  # 表达尚未完成结构验证的递归 JSON 值。

EXPECTED_EXECUTION_FAMILY = "FEniCS/FEniCSx"  # 冻结结构演示必须使用的执行技术族。
EXPECTED_SIMULATION_STATUS = "structural_simulation_passed"  # 冻结真实结构数值链通过时的专用状态枚举。
EXPECTED_RESEARCH_STATUS = "not_executed_missing_current_evidence"  # 冻结原 FEN 研究案例仍因缺证而未执行的状态。
EXPECTED_VALIDATION_KIND = "fenicsx_elasticity_artifact_validation_evidence"  # 定义本独立校验回执的证据类别。
EXPECTED_MESH_DIVISIONS = ((16, 4), (32, 8), (64, 16), (128, 32))  # 冻结四层矩形网格从粗到细的两个方向等分数。
EXPECTED_LINEAR_SOLVE_CALLS = 5  # 要求四层结构求解加一次 XDMF 重读求解共五次。
EXPECTED_ANALYSIS_JSON_FILES: dict[str, str] = {  # 冻结必须随模拟发布的十二个分析与验证 JSON 工件。
    "analysis_charter": "analysis_charter.json",  # 记录分析用途、G0 质量门和演示边界。
    "standards_manifest": "standards_manifest.json",  # 记录本演示未冻结任何正式适用规范。
    "response_metric_register": "response_metric_register.json",  # 记录位移、能量和应力指标合同。
    "approval_matrix": "approval_matrix.json",  # 记录允许与未获得的工程批准状态。
    "scope_exclusion_register": "scope_exclusion_register.json",  # 记录塑性、接触和规范验算等排除范围。
    "solution_verification_report": "solution_verification_report.json",  # 记录平衡、残差和能量解验证结论。
    "verified_result_set": "verified_result_set.json",  # 记录仅可用于演示的已验证数值结果集合。
    "global_equilibrium_report": "global_equilibrium_report.json",  # 记录全局力与力矩平衡检查。
    "substructure_free_body_report": "substructure_free_body_report.json",  # 记录子结构自由体或边界反力审计。
    "mesh_and_step_convergence_report": "mesh_and_step_convergence_report.json",  # 记录四层网格与固定指标变化。
    "solver_warning_disposition": "solver_warning_disposition.json",  # 记录求解器警告及其处置状态。
    "solution_issues": "solution_issues.json",  # 记录尚未解决事项与解释限制。
}  # 结束十二个固定 JSON 工件映射。
EXPECTED_RECORDED_FILE_KINDS: dict[str, tuple[str, str]] = {  # 冻结 simulation_receipt.files 至少必须包含的七类可视化和场文件。
    "summary_png": (".png", "png"),  # 要求由真实求解数据绘制的结构摘要 PNG。
    "mesh_tags_xdmf": (".xdmf", "xdmf"),  # 要求包含最细层命名网格和标签的 XDMF 索引。
    "mesh_tags_h5": (".h5", "hdf5"),  # 要求与网格标签 XDMF 配套的 HDF5 数据文件。
    "displacement_xdmf": (".xdmf", "xdmf"),  # 要求位移向量场的 XDMF 索引。
    "displacement_h5": (".h5", "hdf5"),  # 要求位移向量场的 HDF5 数据文件。
    "von_mises_xdmf": (".xdmf", "xdmf"),  # 要求 DG0 von Mises 应力场的 XDMF 索引。
    "von_mises_h5": (".h5", "hdf5"),  # 要求 DG0 von Mises 应力场的 HDF5 数据文件。
}  # 结束七类强制记录文件映射。
LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")  # 定义合法小写 SHA-256 十六进制字符集合。
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"  # 冻结 PNG 文件开头必须具有的八字节标准签名。
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"  # 冻结本工作流生成的 HDF5 文件开头八字节签名。
FORBIDDEN_COMPLIANCE_KEYS = frozenset({"code_compliance_passed", "standards_compliance_passed", "normative_compliance_passed", "design_code_passed", "code_check_passed"})  # 列出不得在演示解验证中为真的规范通过字段。
FORBIDDEN_PASS_TEXT = frozenset({"pass", "passed", "compliant", "approved", "satisfied"})  # 定义规范相关字段不得使用的肯定通过文本。


def utc_now() -> str:  # 不接收输入并返回带 UTC 偏移的 ISO 8601 验证时间文本。
    return datetime.now(timezone.utc).isoformat()  # 使用 Actions runner 的真实系统时钟形成审计时间。


def parse_args() -> argparse.Namespace:  # 定义 --output-dir 必填参数并返回通过 argparse 语法校验的命名空间。
    parser = argparse.ArgumentParser(description="Independently validate FEniCSx elasticity simulation artifacts.")  # 创建不暗示原研究案例或规范验算通过的解析器。
    parser.add_argument("--output-dir", required=True, help="Elasticity simulation artifact root.")  # 要求显式提供 simulation_receipt 和全部工件所在目录。
    return parser.parse_args()  # 返回包含 output_dir 字符串的参数对象供主流程使用。


def write_json(path: Path, payload: dict[str, Any]) -> None:  # 将 payload 以严格 UTF-8 JSON 写入 path，并拒绝 NaN 与 Infinity。
    path.parent.mkdir(parents=True, exist_ok=True)  # 在成功和失败路径均确保验证回执父目录存在。
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"  # 使用稳定键序、两空格缩进和单个末尾换行。
    path.write_text(serialized, encoding="utf-8")  # 由单个宿主进程写入完整验证回执。


def reject_nonfinite_constant(token: str) -> None:  # 接收 JSON 非有限数字标记并总是抛错以实现严格 JSON。
    raise ValueError("non-finite JSON number is forbidden: " + token)  # 拒绝标准外的 NaN、Infinity 和负 Infinity。


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:  # 接收单个 JSON object 的键值对并拒绝重复键。
    result: dict[str, Any] = {}  # 创建当前 object 的独立结果映射。
    for key, value in pairs:  # 按源文件顺序检查每个键值对。
        if key in result:  # 检测同一 object 内已出现的重复键。
            raise ValueError("duplicate JSON key is forbidden: " + key)  # 防止后值覆盖前值造成审计歧义。
        result[key] = value  # 只将首次出现的合法键加入结果映射。
    return result  # 返回已证明无重复键的 JSON object。


def strict_json_loads(text: str) -> Any:  # 严格解析 UTF-8 解码后的 JSON 文本并返回递归标准库值。
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_nonfinite_constant)  # 同时拒绝重复键和非有限数字扩展。


def sha256_file(path: Path) -> str:  # 接收普通文件路径并返回其完整字节内容的六十四字符 SHA-256。
    digest = hashlib.sha256()  # 为当前文件创建独立 SHA-256 状态。
    with path.open("rb") as source:  # 使用二进制只读模式避免换行或编码转换。
        for chunk in iter(lambda: source.read(1024 * 1024), b""):  # 按一 MiB 数据块读取大型 HDF5 场文件。
            digest.update(chunk)  # 将当前非空数据块加入摘要状态。
    return digest.hexdigest()  # 返回小写十六进制摘要供严格等值比较。


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:  # 向 checks 追加稳定名称、严格布尔结果和可审计细节。
    checks.append({"name": name, "passed": bool(passed), "details": details})  # 规范化 passed 并保留实际与期望上下文。


def all_checks_pass(checks: list[dict[str, Any]]) -> bool:  # 仅在 checks 非空且每项 passed 严格为真时返回真。
    return bool(checks) and all(check.get("passed") is True for check in checks)  # 拒绝空检查集合、缺失状态或任一失败项。


def safe_file_identity(path: Path) -> dict[str, Any]:  # 返回 path 的存在性、大小和摘要，I/O 错误转为可序列化 error 字段。
    try:  # 捕获不存在、权限和读取期间的文件系统错误。
        exists = path.is_file()  # 仅把普通文件视为可接受工件。
        return {  # 返回稳定文件身份对象供检查和最终回执使用。
            "path": str(path),  # 保存解析后的文件路径文本。
            "exists": exists,  # 保存普通文件存在性布尔值。
            "size_bytes": int(path.stat().st_size) if exists else 0,  # 对存在文件记录字节数，否则记零。
            "sha256": sha256_file(path) if exists else "",  # 对存在文件独立重算摘要，否则留空。
        }  # 结束正常或缺失文件身份对象。
    except OSError as error:  # 将预期文件系统错误保留到验证证据。
        return {"path": str(path), "exists": False, "size_bytes": 0, "sha256": "", "error": type(error).__name__ + ": " + str(error)}  # 返回明确失败身份而不中断其他检查。


def load_strict_json_object(checks: list[dict[str, Any]], label: str, path: Path) -> dict[str, Any]:  # 读取 label 对应 path 并要求严格 JSON object。
    identity = safe_file_identity(path)  # 在解析前记录文件存在性、大小和摘要。
    try:  # 捕获文件、编码、严格 JSON 和顶层类型错误。
        text = path.read_text(encoding="utf-8")  # 使用严格 UTF-8 解码工件文本。
        payload = strict_json_loads(text)  # 使用拒绝重复键和非有限数字的解析器。
        if not isinstance(payload, dict):  # 只接受顶层 JSON object，拒绝数组、标量和 null。
            raise TypeError("top-level JSON value must be an object")  # 形成稳定顶层类型错误消息。
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:  # 将全部预期输入错误转换为失败检查。
        add_check(checks, label + "_strict_json_object", False, {"file": identity, "error": type(error).__name__ + ": " + str(error)})  # 保存文件身份和严格解析错误。
        return {}  # 返回空对象使其余边界检查继续形成确定性失败。
    add_check(checks, label + "_strict_json_object", True, {"file": identity})  # 记录文件存在且是无重复键、无非有限数的 JSON object。
    return payload  # 返回已完成严格顶层结构验证的映射。


def validate_simulation_header(checks: list[dict[str, Any]], receipt: dict[str, Any]) -> None:  # 校验 simulation_receipt 的状态、执行族和反冒充边界。
    add_check(checks, "simulation_status", receipt.get("status") == EXPECTED_SIMULATION_STATUS, {"expected": EXPECTED_SIMULATION_STATUS, "actual": receipt.get("status")})  # 只接受结构模拟专用通过状态。
    add_check(checks, "simulation_execution_family", receipt.get("execution_family") == EXPECTED_EXECUTION_FAMILY, {"expected": EXPECTED_EXECUTION_FAMILY, "actual": receipt.get("execution_family")})  # 防止结构模拟串接 CalculiX 或其他后端。
    add_check(checks, "simulation_scientific_claim_forbidden", receipt.get("scientific_claim_allowed") is False, {"expected": False, "actual": receipt.get("scientific_claim_allowed")})  # 要求严格 JSON false 禁止论文或工程结论。
    add_check(checks, "simulation_research_case_status", receipt.get("research_case_execution_status") == EXPECTED_RESEARCH_STATUS, {"expected": EXPECTED_RESEARCH_STATUS, "actual": receipt.get("research_case_execution_status")})  # 要求原 FEN 案例继续缺证阻断。
    add_check(checks, "simulation_uses_no_original_inputs", receipt.get("uses_original_research_inputs") is False, {"expected": False, "actual": receipt.get("uses_original_research_inputs")})  # 要求明确声明全部结构输入是演示合成输入。
    counts_value = receipt.get("execution_counts")  # 读取结构模拟的工具和求解调用计数对象。
    counts = counts_value if isinstance(counts_value, dict) else {}  # 类型错误时使用空对象产生确定性失败。
    counts_pass = counts.get("linear_solve_calls") == EXPECTED_LINEAR_SOLVE_CALLS and counts.get("calculix_calls") == 0 and counts.get("model_calls") == 0  # 要求五次 FEniCSx 线性求解且不调用 CalculiX 或模型服务。
    add_check(checks, "simulation_execution_counts", counts_pass, {"expected_linear_solve_calls": EXPECTED_LINEAR_SOLVE_CALLS, "actual": counts})  # 保存实际计数供失败定位。
    receipt_checks_value = receipt.get("checks")  # 读取数值 runner 的预冻结检查列表。
    receipt_checks_are_array = isinstance(receipt_checks_value, list)  # 只接受 JSON array 作为检查集合。
    receipt_checks = receipt_checks_value if receipt_checks_are_array else []  # 类型错误时使用空数组形成失败。
    receipt_checks_pass = bool(receipt_checks) and all(isinstance(row, dict) and row.get("passed") is True for row in receipt_checks)  # 要求非空且每项严格通过。
    failed_or_invalid = [row for row in receipt_checks if not isinstance(row, dict) or row.get("passed") is not True]  # 收集 runner 自报失败或结构非法行。
    add_check(checks, "simulation_checks_nonempty_all_passed", receipt_checks_are_array and receipt_checks_pass, {"count": len(receipt_checks), "failed_or_invalid": failed_or_invalid})  # 防止空 checks 或字符串真值误绿。


def validate_mesh_sequence(checks: list[dict[str, Any]], receipt: dict[str, Any]) -> None:  # 校验四层 divisions、规模递增和每层 PETSc 正收敛原因码。
    levels_value = receipt.get("mesh_levels")  # 读取粗到细的四层结构网格结果数组。
    levels_are_array = isinstance(levels_value, list)  # 只接受 JSON array 作为网格序列。
    levels = levels_value if levels_are_array else []  # 类型错误时使用空数组继续形成失败。
    expected_divisions = [list(pair) for pair in EXPECTED_MESH_DIVISIONS]  # 将冻结元组转换为 JSON 风格二维数组供比较。
    actual_divisions = [row.get("divisions") for row in levels if isinstance(row, dict)]  # 从合法层级对象读取两个方向等分数。
    sequence_pass = levels_are_array and len(levels) == len(expected_divisions) and actual_divisions == expected_divisions  # 要求恰好四层且顺序和值完全一致。
    add_check(checks, "mesh_four_level_sequence", sequence_pass, {"expected": expected_divisions, "actual": actual_divisions, "row_count": len(levels)})  # 保存实际序列供错配审计。
    global_cells = [row.get("global_cells") for row in levels if isinstance(row, dict)]  # 读取每层全局三角形数量。
    global_dofs = [row.get("global_dofs") for row in levels if isinstance(row, dict)]  # 读取每层全局位移自由度数量。
    cells_increase = len(global_cells) == len(expected_divisions) and all(type(value) is int and value > 0 for value in global_cells) and all(global_cells[index] > global_cells[index - 1] for index in range(1, len(global_cells)))  # 要求正整数单元数严格递增。
    dofs_increase = len(global_dofs) == len(expected_divisions) and all(type(value) is int and value > 0 for value in global_dofs) and all(global_dofs[index] > global_dofs[index - 1] for index in range(1, len(global_dofs)))  # 要求正整数自由度数严格递增。
    add_check(checks, "mesh_cells_and_dofs_increase", cells_increase and dofs_increase, {"global_cells": global_cells, "global_dofs": global_dofs})  # 防止重复同一网格伪造收敛序列。
    ksp_reasons = [row.get("ksp_converged_reason") for row in levels if isinstance(row, dict)]  # 读取四层 PETSc KSP 原始收敛原因码。
    reasons_pass = len(ksp_reasons) == len(expected_divisions) and all(type(reason) is int and reason > 0 for reason in ksp_reasons)  # 要求四个严格整数原因码均为 PETSc 正成功值。
    add_check(checks, "mesh_level_ksp_reasons_positive", reasons_pass, {"expected_positive_count": len(expected_divisions), "actual": ksp_reasons})  # 独立拒绝只写 pass 文本而没有正原因码的回执。


def validate_roundtrip(checks: list[dict[str, Any]], receipt: dict[str, Any]) -> None:  # 校验 XDMF 重读比较通过且第五次导入网格求解具有正 KSP 原因码。
    roundtrip_value = receipt.get("roundtrip_comparison")  # 读取最细层原始与重读模型的差分比较对象。
    roundtrip = roundtrip_value if isinstance(roundtrip_value, dict) else {}  # 类型错误时使用空对象形成失败。
    add_check(checks, "roundtrip_comparison_pass", roundtrip.get("outcome") == "pass", {"expected": "pass", "actual": roundtrip.get("outcome"), "actual_type": type(roundtrip_value).__name__})  # 要求往返比较明确报告通过。
    imported_reason = roundtrip.get("imported_ksp_converged_reason")  # 读取第五次重读网格 PETSc 收敛原因码。
    add_check(checks, "roundtrip_imported_ksp_reason_positive", type(imported_reason) is int and imported_reason > 0, {"actual": imported_reason})  # 要求第五次真实线性求解成功而非只比较缓存值。


def validate_engineering_gates(checks: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]]) -> None:  # 校验 G0、规范空清单、演示解验证和 verified-result 使用边界。
    charter = artifacts.get("analysis_charter", {})  # 读取分析章程对象供 G0 门验证。
    charter_pass = charter.get("gate_status") == "BLOCKED" and charter.get("intended_use_class") == "demonstration_only"  # 要求 G0 阻断且用途仅为演示。
    add_check(checks, "analysis_charter_g0_blocked_demo_only", charter_pass, {"gate_status": charter.get("gate_status"), "intended_use_class": charter.get("intended_use_class")})  # 防止真实模型缺失时误开放工程用途。
    standards = artifacts.get("standards_manifest", {})  # 读取适用规范清单对象。
    standards_pass = standards.get("gate_status") == "BLOCKED" and standards.get("standards") == []  # 要求规范门阻断且正式规范列表严格为空。
    add_check(checks, "standards_manifest_g0_blocked_empty", standards_pass, {"gate_status": standards.get("gate_status"), "standards": standards.get("standards")})  # 防止凭空宣称已冻结适用规范。
    verification = artifacts.get("solution_verification_report", {})  # 读取解验证报告对象。
    verification_pass = verification.get("verification_outcome") == "pass_for_demonstration_only" and verification.get("engineering_acceptance_allowed") is False  # 只接受演示层数值解验证且禁止工程接受。
    add_check(checks, "solution_verification_demo_only", verification_pass, {"verification_outcome": verification.get("verification_outcome"), "engineering_acceptance_allowed": verification.get("engineering_acceptance_allowed")})  # 区分方程解验证和规范验算。
    verified_results = artifacts.get("verified_result_set", {})  # 读取已验证结果集合对象。
    verified_results_pass = verified_results.get("allowed_use") == "demonstration_only" and verified_results.get("scientific_claim_allowed") is False  # 要求结果仅用于演示且禁止科学结论。
    add_check(checks, "verified_result_set_demo_only", verified_results_pass, {"allowed_use": verified_results.get("allowed_use"), "scientific_claim_allowed": verified_results.get("scientific_claim_allowed")})  # 防止 verified 一词被误解为工程合格。


def collect_forbidden_compliance_claims(value: Any, path: str = "$") -> list[dict[str, Any]]:  # 递归扫描 value 并返回规范、标准或设计验算被肯定通过的路径和值。
    findings: list[dict[str, Any]] = []  # 创建当前递归分支发现的违规声明列表。
    if isinstance(value, dict):  # 对 JSON object 逐键检查并递归处理子值。
        for key, child in value.items():  # 保留源对象键名以构造可审计 JSON 路径。
            child_path = path + "." + str(key)  # 使用点号拼接当前字段路径供失败回执定位。
            normalized_key = str(key).strip().lower()  # 将键规范化为小写以执行不区分大小写的规则。
            compliance_related = any(token in normalized_key for token in ("standard", "code_compliance", "normative", "design_code", "code_check"))  # 识别与规范通过声明直接相关的字段。
            approval_signal_key = any(token in normalized_key for token in ("pass", "approved", "compliant", "satisfied"))  # 识别字段名自身是否表达肯定通过或批准语义。
            text_is_forbidden_pass = isinstance(child, str) and child.strip().lower() in FORBIDDEN_PASS_TEXT  # 判断字符串是否为肯定通过枚举。
            explicit_boolean_violation = normalized_key in FORBIDDEN_COMPLIANCE_KEYS and child is True  # 判断显式通过布尔字段是否错误为真。
            contextual_text_violation = compliance_related and text_is_forbidden_pass  # 判断规范相关字段是否使用肯定通过文本。
            contextual_boolean_violation = compliance_related and approval_signal_key and child is True  # 仅在规范相关肯定字段为真时判定违规，允许“需规范审查”等保守布尔字段为真。
            if explicit_boolean_violation or contextual_text_violation or contextual_boolean_violation:  # 任一禁止模式命中时记录完整路径和值。
                findings.append({"path": child_path, "value": child})  # 保存可 JSON 序列化的违规声明。
            findings.extend(collect_forbidden_compliance_claims(child, child_path))  # 继续扫描嵌套 object 或 array。
    elif isinstance(value, list):  # 对 JSON array 按稳定索引递归扫描每个元素。
        for index, child in enumerate(value):  # 使用零基索引构造唯一子路径。
            findings.extend(collect_forbidden_compliance_claims(child, path + "[" + str(index) + "]"))  # 递归收集当前数组元素中的违规声明。
    return findings  # 返回当前值及全部后代的规范通过违规列表。


def validate_no_code_compliance_claim(checks: list[dict[str, Any]], artifacts: dict[str, dict[str, Any]]) -> None:  # 扫描解验证和 verified results，确保没有声称正式规范通过。
    verification_findings = collect_forbidden_compliance_claims(artifacts.get("solution_verification_report", {}), "$.solution_verification_report")  # 扫描解验证报告全部嵌套字段。
    result_findings = collect_forbidden_compliance_claims(artifacts.get("verified_result_set", {}), "$.verified_result_set")  # 扫描已验证结果集合全部嵌套字段。
    findings = verification_findings + result_findings  # 合并两类高风险工件中的所有违规声明。
    add_check(checks, "no_standards_or_code_pass_claim", not findings, {"forbidden_claims": findings})  # 只有完全没有规范肯定通过声明时接受。


def validate_file_format(path: Path, format_kind: str) -> tuple[bool, str]:  # 根据 format_kind 验证 path 的 PNG、HDF5 签名或 XDMF XML 根元素。
    try:  # 捕获格式读取和 XML 解析错误并返回可审计消息。
        if format_kind == "png":  # 对摘要图检查固定八字节 PNG 签名。
            with path.open("rb") as source:  # 以二进制只读方式打开摘要图并避免把完整图片载入内存。
                actual_signature = source.read(len(PNG_SIGNATURE))  # 只读取 PNG 标准签名所需的八个字节。
            return actual_signature == PNG_SIGNATURE, "PNG signature " + actual_signature.hex()  # 返回签名比较结果和实际十六进制值。
        if format_kind == "hdf5":  # 对场和网格数据检查固定八字节 HDF5 签名。
            with path.open("rb") as source:  # 以二进制只读方式打开大型 HDF5 文件并限制内存占用。
                actual_signature = source.read(len(HDF5_SIGNATURE))  # 只读取 HDF5 标准签名所需的八个字节。
            return actual_signature == HDF5_SIGNATURE, "HDF5 signature " + actual_signature.hex()  # 返回签名比较结果和实际十六进制值。
        if format_kind == "xdmf":  # 对 XDMF 索引执行标准库 XML 解析和根元素检查。
            root = ElementTree.parse(path).getroot()  # 解析完整 XML 并取得根元素。
            local_name = root.tag.rsplit("}", 1)[-1].lower()  # 去除可选 XML 命名空间后规范化根标签。
            return local_name == "xdmf", "XML root " + str(root.tag)  # 只接受 Xdmf 根元素并报告实际标签。
        return True, "no specialized format rule"  # 对额外记录文件仅执行大小和摘要检查。
    except (OSError, ElementTree.ParseError, ValueError) as error:  # 将读取和 XML 语法错误转换为格式失败消息。
        return False, type(error).__name__ + ": " + str(error)  # 返回假和稳定错误文本而不中断其他文件校验。


def validate_recorded_files(checks: list[dict[str, Any]], output_root: Path, files_value: Any) -> None:  # 校验 files 中全部路径、大小和摘要，并强制七类结构场文件存在。
    files_are_object = isinstance(files_value, dict)  # 只接受以逻辑键索引文件记录的 JSON object。
    add_check(checks, "recorded_files_object", files_are_object, {"actual_type": type(files_value).__name__})  # 明确报告缺失 files 或错误类型。
    files = files_value if files_are_object else {}  # 类型错误时使用空对象继续生成确定性失败。
    required_keys = set(EXPECTED_RECORDED_FILE_KINDS)  # 构造七个强制逻辑文件键集合。
    actual_keys = set(files.keys())  # 读取回执实际提供的所有文件键。
    add_check(checks, "required_structural_file_keys", required_keys.issubset(actual_keys), {"required": sorted(required_keys), "actual": sorted(str(key) for key in actual_keys), "missing": sorted(required_keys - actual_keys)})  # 要求 PNG、网格、位移和应力成对文件齐全。
    for file_key, record_value in files.items():  # 对 files 中每个记录而非仅七个强制键执行完整性验证。
        record_is_object = isinstance(record_value, dict)  # 只接受含 path、size_bytes 和 sha256 的 JSON object。
        if not record_is_object:  # 对非法记录生成失败并继续下一个文件。
            add_check(checks, "recorded_file_" + str(file_key), False, {"reason": "file record must be an object", "actual_type": type(record_value).__name__})  # 保存逻辑键和错误类型。
            continue  # 避免对非法值执行路径与摘要访问。
        record = record_value  # 经过类型门后将当前值作为文件记录映射使用。
        recorded_path = record.get("path")  # 读取 runner 声称位于 artifact 根内的相对路径。
        path_is_relative_text = isinstance(recorded_path, str) and bool(recorded_path.strip()) and not Path(recorded_path).is_absolute()  # 拒绝空字符串和绝对路径。
        candidate_path = (output_root / recorded_path).resolve() if path_is_relative_text else output_root  # 仅对合法文本解析候选路径，否则使用根目录占位。
        try:  # 通过 relative_to 检查规范化路径没有使用点点段或符号链接逃逸根目录。
            candidate_path.relative_to(output_root)  # 成功时证明候选路径仍位于 artifact 根内。
            path_within_root = path_is_relative_text  # 只有原路径相对且规范化后在根内才接受。
        except ValueError:  # 捕获候选路径不属于 artifact 根的情况。
            path_within_root = False  # 明确拒绝越界路径以防读取仓库或系统文件。
        actual_identity = safe_file_identity(candidate_path) if path_within_root else {"path": str(candidate_path), "exists": False, "size_bytes": 0, "sha256": "", "error": "path escapes artifact root"}  # 对安全路径重算身份，对越界路径返回失败对象。
        recorded_size = record.get("size_bytes")  # 读取 runner 记录的正整数字节数。
        recorded_sha256 = record.get("sha256")  # 读取 runner 记录的小写 SHA-256。
        size_is_valid = type(recorded_size) is int and recorded_size > 0  # 拒绝布尔、字符串、零和负数大小。
        digest_is_valid = isinstance(recorded_sha256, str) and len(recorded_sha256) == 64 and set(recorded_sha256).issubset(LOWERCASE_HEX_DIGITS)  # 要求完整小写十六进制摘要。
        required_kind = EXPECTED_RECORDED_FILE_KINDS.get(str(file_key))  # 读取当前逻辑键的扩展名和格式签名合同。
        expected_suffix = required_kind[0] if required_kind else ""  # 对额外文件不冻结扩展名，对七类文件冻结后缀。
        format_kind = required_kind[1] if required_kind else "other"  # 对额外文件只校验身份，不执行专门格式解析。
        suffix_pass = not expected_suffix or str(recorded_path).lower().endswith(expected_suffix)  # 要求七类文件的路径后缀与逻辑键一致。
        format_pass, format_details = validate_file_format(candidate_path, format_kind) if actual_identity.get("exists") is True else (False, "file is missing")  # 对存在文件验证真实格式签名或 XML。
        identity_pass = path_within_root and actual_identity.get("exists") is True and size_is_valid and digest_is_valid and recorded_size == actual_identity.get("size_bytes") and recorded_sha256 == actual_identity.get("sha256")  # 要求路径、存在性、大小和摘要全部一致。
        file_passed = identity_pass and suffix_pass and format_pass  # 七类强制文件还必须满足扩展名和内容格式检查。
        add_check(checks, "recorded_file_" + str(file_key), file_passed, {"recorded": record, "actual": actual_identity, "path_within_root": path_within_root, "suffix_expected": expected_suffix, "suffix_passed": suffix_pass, "format": format_details})  # 保存全部独立比较细节。


def main() -> int:  # 组织严格读取、结构验证、文件复核和始终写回执的完整主流程。
    args = parse_args()  # 读取 Actions 显式传入的 artifact 根目录。
    output_root = Path(args.output_dir).resolve()  # 规范化根目录供路径边界检查和回执写入。
    validation_receipt_path = output_root / "artifact_validation_receipt.json"  # 冻结成功和失败共同使用的验证回执文件名。
    started_at = utc_now()  # 在任何 artifact 读取前记录验证开始时间。
    checks: list[dict[str, Any]] = []  # 收集严格 JSON、数值状态、工程门和文件完整性检查。
    artifacts: dict[str, dict[str, Any]] = {}  # 保存十二个已解析分析与验证工件供交叉检查。
    unexpected_error = ""  # 默认没有 validator 自身异常，兜底路径会写入类型和消息。
    try:  # 捕获所有非预期错误以保证失败回执仍被尝试写出。
        output_root.mkdir(parents=True, exist_ok=True)  # 确保 artifact 根存在并可写验证回执。
        simulation_receipt_path = output_root / "simulation_receipt.json"  # 定位结构数值 runner 的主回执。
        simulation_receipt = load_strict_json_object(checks, "simulation_receipt", simulation_receipt_path)  # 严格读取主回执并拒绝重复键和非有限数。
        for artifact_id, filename in EXPECTED_ANALYSIS_JSON_FILES.items():  # 按冻结清单逐个读取全部分析与验证工件。
            artifacts[artifact_id] = load_strict_json_object(checks, artifact_id, output_root / filename)  # 将严格 JSON object 保存到稳定工件标识下。
        validate_simulation_header(checks, simulation_receipt)  # 校验专用状态、反冒充边界和五次求解计数。
        validate_mesh_sequence(checks, simulation_receipt)  # 校验四层固定网格、规模递增和四个 KSP 正原因码。
        validate_roundtrip(checks, simulation_receipt)  # 校验往返比较和第五次重读求解原因码。
        validate_engineering_gates(checks, artifacts)  # 校验 G0 阻断、空规范清单和演示限定结果。
        validate_no_code_compliance_claim(checks, artifacts)  # 独立扫描并拒绝任何正式规范通过声明。
        validate_recorded_files(checks, output_root, simulation_receipt.get("files"))  # 复核 files 中所有路径、大小、摘要与七类必备输出。
    except BaseException as error:  # 兜底捕获编程、资源和意外输入组合错误，避免无回执失败。
        unexpected_error = type(error).__name__ + ": " + str(error)  # 保存异常类型和消息供 Actions 调试。
        add_check(checks, "validator_unexpected_exception", False, {"error": unexpected_error})  # 将 validator 自身异常转换为明确失败检查。
    passed = all_checks_pass(checks) and not unexpected_error  # 只有全部检查通过且无兜底异常时接受 artifact。
    failed_check_names = [str(check.get("name", "unnamed_check")) for check in checks if check.get("passed") is not True]  # 收集全部失败检查稳定名称供快速定位。
    input_identities = {"simulation_receipt": safe_file_identity(output_root / "simulation_receipt.json")}  # 首先记录主模拟回执的实际文件身份。
    for artifact_id, filename in EXPECTED_ANALYSIS_JSON_FILES.items():  # 为十二个分析与验证工件补充实际文件身份。
        input_identities[artifact_id] = safe_file_identity(output_root / filename)  # 保存工件路径、大小和独立摘要。
    receipt = {  # 构造成功和失败共同使用的独立结构 artifact 验证回执。
        "schema_version": "v5-fenicsx-elasticity-artifact-validation/1.0",  # 冻结 validator 输出合同版本。
        "status": "artifact_validation_passed" if passed else "artifact_validation_failed",  # 使用不会被误读为工程合格的专用状态。
        "validation_outcome": "pass" if passed else "fail",  # 提供 Actions 和后续脚本可稳定消费的枚举。
        "artifact_kind": EXPECTED_VALIDATION_KIND,  # 明确该文件是独立验证证据而非结构计算结果。
        "started_at_utc": started_at,  # 保存验证活动开始时间。
        "finished_at_utc": utc_now(),  # 保存全部检查完成后的结束时间。
        "output_dir": str(output_root),  # 保存被验证 artifact 根的规范化路径。
        "input_files": input_identities,  # 保存主回执和十二个 JSON 工件的实际大小与摘要。
        "check_count": len(checks),  # 保存实际执行的独立检查数量。
        "failed_check_count": len(failed_check_names),  # 保存失败检查总数。
        "failed_check_names": failed_check_names,  # 保存全部失败检查名称。
        "checks": checks,  # 保存完整通过与失败细节供 artifact 审计。
        "unexpected_error": unexpected_error,  # 无异常时为空字符串，有异常时保存类型和消息。
        "execution_family": EXPECTED_EXECUTION_FAMILY,  # 重申被验证数值链属于 FEniCS/FEniCSx。
        "scientific_claim_allowed": False,  # validator 本身不能支持工程或论文结论。
        "research_case_execution_status": EXPECTED_RESEARCH_STATUS,  # 重申独立校验不改变原 FEN 案例状态。
        "allowed_use": "仅用于确认 FEniCSx 结构演示 artifact 的机器合同、解验证边界和文件完整性。",  # 限定验证回执的允许用途。
        "disallowed_use": "不得据此宣称原研究案例、真实结构或任何正式规范验算已经通过。",  # 明确禁止工程和规范冒充。
    }  # 结束独立结构 artifact 验证回执。
    try:  # 独立捕获最终回执写入错误并确保进程返回失败。
        write_json(validation_receipt_path, receipt)  # 无论 passed 为真或假都写到固定 artifact 路径。
    except BaseException as error:  # 文件系统不可写时无法满足留证合同，必须输出错误并失败。
        print(json.dumps({"status": "artifact_validation_receipt_write_failed", "path": str(validation_receipt_path), "error": type(error).__name__ + ": " + str(error)}, ensure_ascii=False), file=sys.stderr)  # 在 Actions stderr 保留最后可用证据。
        return 1  # 回执写入失败无条件返回一，禁止工作流误绿。
    print(json.dumps({"status": receipt["status"], "validation_outcome": receipt["validation_outcome"], "failed_check_names": failed_check_names, "receipt": str(validation_receipt_path)}, ensure_ascii=False))  # 在 Actions stdout 输出精简机器摘要。
    return 0 if passed else 1  # 全部严格检查通过返回零，否则返回一使 Actions 失败。


if __name__ == "__main__":  # 仅在 Actions 或维护者直接调用本文件时执行主流程。
    raise SystemExit(main())  # 将 main 的零或一精确传递给调用 shell和 GitHub Actions。
