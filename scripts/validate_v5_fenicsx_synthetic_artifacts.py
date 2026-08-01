#!/usr/bin/env python3
# 上一行选择 GitHub Actions 自带的 Python 3；本脚本仅用标准库独立校验 FEniCSx 合成数值 artifact。
"""独立复核合成数值回执、原案例边界与记录文件的字节大小和 SHA-256。"""  # 明确本脚本不导入或信任 DOLFINx runner 的判定逻辑。
from __future__ import annotations  # 启用现代类型注解语义并避免运行时提前解析复杂注解。

import argparse  # 解析 artifact 根目录与两个权威 profile 路径。
import hashlib  # 独立重算 artifact 文件和输入 JSON 文件的 SHA-256。
import json  # 读取 runner 回执与 profile，并写出验证回执。
import sys  # 将验证结论转换为 Actions 可消费的进程退出码和标准输出。
from datetime import datetime, timezone  # 生成带显式时区的 UTC 审计时间。
from pathlib import Path  # 安全解析输入路径、artifact 相对路径和防目录穿越边界。
from typing import Any  # 表达解析后尚未完成结构验证的 JSON 值。

EXPECTED_EXECUTION_FAMILY = "FEniCS/FEniCSx"  # 冻结两个 FEN profile 和数值回执必须使用的执行技术族。
EXPECTED_PROFILE_STATUS = "draft_not_executed"  # 冻结原研究案例在合成测试后仍未执行的 profile 状态。
EXPECTED_RESEARCH_STATUS = "not_executed_missing_current_evidence"  # 冻结回执必须保留的原研究案例阻断状态。
EXPECTED_ARTIFACT_KIND = "synthetic_numerical_contract_evidence"  # 冻结合成数值证据类型，防止冒充原案例结果。
EXPECTED_VALIDATION_KIND = "synthetic_artifact_validation_evidence"  # 定义本独立校验回执自身的证据类型。
EXPECTED_CASE_SPECS: dict[str, dict[str, Any]] = {  # 为两个能力参考案例冻结互不混淆的校验合同。
    "FEN-003": {  # 定义制造解网格收敛合成合同的固定要求。
        "receipt_relative_path": "fen-003/numerical_contract_receipt.json",  # 指向 FEN-003 runner 回执的 artifact 相对路径。
        "missing_fact_count": 5,  # 要求原 FEN-003 profile 仍保留五项当前缺失事实。
        "benchmark_id": "SYN-FEN-003-MMS-POISSON-P1",  # 冻结制造解基准标识并使用 SYN 前缀。
        "linear_solve_calls": 4,  # 要求四个受控网格层级各完成一次真实线性求解。
        "expected_divisions": (8, 16, 32, 64),  # 冻结粗到细四个结构化网格等分数。
        "file_keys": ("solution_xdmf", "solution_h5"),  # 要求最细层 XDMF 索引和 HDF5 场文件都被记录。
    },  # 结束 FEN-003 固定合同。
    "FEN-014": {  # 定义 XDMF 与 MeshTags 往返合成合同的固定要求。
        "receipt_relative_path": "fen-014/numerical_contract_receipt.json",  # 指向 FEN-014 runner 回执的 artifact 相对路径。
        "missing_fact_count": 7,  # 要求原 FEN-014 profile 仍保留七项当前缺失事实。
        "benchmark_id": "SYN-FEN-014-XDMF-MESHTAGS-ROUNDTRIP",  # 冻结往返基准标识并使用 SYN 前缀。
        "linear_solve_calls": 2,  # 要求内存参考网格与重读网格各完成一次真实线性求解。
        "file_keys": ("import_xdmf", "import_h5", "solution_xdmf", "solution_h5"),  # 要求输入包和重读解的四个文件都被记录。
    },  # 结束 FEN-014 固定合同。
}  # 结束两个案例固定合同映射。
LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")  # 定义合法小写 SHA-256 十六进制字符集合。


def utc_now() -> str:  # 返回带 UTC 偏移的 ISO 8601 文本，不接收输入且供验证回执记录时间。
    return datetime.now(timezone.utc).isoformat()  # 使用 Actions runner 的真实系统时钟形成审计时间。


def parse_args() -> argparse.Namespace:  # 定义三个必填路径参数并返回通过 argparse 语法校验的命名空间。
    parser = argparse.ArgumentParser(description="Independently validate V5 FEniCSx synthetic numerical artifacts.")  # 创建不暗示原研究案例通过的命令行解析器。
    parser.add_argument("--output-dir", required=True, help="Synthetic artifact root containing campaign and case receipts.")  # 要求显式提供可读取且可写验证回执的 artifact 根目录。
    parser.add_argument("--fen003-profile", required=True, help="Authoritative FEN-003 profile JSON path.")  # 要求显式提供当前提交中的 FEN-003 profile。
    parser.add_argument("--fen014-profile", required=True, help="Authoritative FEN-014 profile JSON path.")  # 要求显式提供当前提交中的 FEN-014 profile。
    return parser.parse_args()  # 返回三个字符串路径参数供主流程解析为绝对路径。


def write_json(path: Path, payload: dict[str, Any]) -> None:  # 将 payload 写到 path，并拒绝 NaN 或 Infinity 破坏严格 JSON。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建验证回执父目录，允许失败路径首次建立 artifact 根。
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"  # 使用 UTF-8 友好、稳定排序和单个末尾换行的表示。
    path.write_text(serialized, encoding="utf-8")  # 原子性要求较低的单进程 Actions 环境直接写入完整文本。


def sha256_file(path: Path) -> str:  # 接收普通文件路径并返回其完整字节内容的六十四字符 SHA-256。
    digest = hashlib.sha256()  # 为当前文件创建独立摘要状态，避免跨文件复用污染。
    with path.open("rb") as source:  # 以二进制只读模式打开文件，确保不受换行或编码转换影响。
        for chunk in iter(lambda: source.read(1024 * 1024), b""):  # 按一 MiB 块读取以限制大型 HDF5 文件的内存占用。
            digest.update(chunk)  # 将当前非空字节块追加到 SHA-256 状态。
    return digest.hexdigest()  # 返回小写十六进制摘要供严格等值比较。


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:  # 向 checks 追加名称、严格布尔结果和可审计细节。
    checks.append({"name": name, "passed": bool(passed), "details": details})  # 规范化 passed 为 JSON 布尔值并保留失败上下文。


def validation_passes(checks: list[dict[str, Any]]) -> bool:  # 接收全部独立检查并仅在非空且每项严格通过时返回真。
    return bool(checks) and all(check.get("passed") is True for check in checks)  # 拒绝空检查集合、缺失状态和任意失败项。


def safe_file_identity(path: Path) -> dict[str, Any]:  # 返回输入文件的存在性、大小和摘要，读取异常转为 error 字段而不终止回执写入。
    try:  # 捕获不存在、权限不足和 I/O 错误以便失败回执仍可形成。
        exists = path.is_file()  # 只把普通文件视为可接受输入，目录和设备均不接受。
        return {  # 返回可由 json 模块稳定序列化的文件身份对象。
            "path": str(path),  # 保存调用方解析后的绝对或明确路径文本。
            "exists": exists,  # 保存普通文件存在性布尔值。
            "size_bytes": int(path.stat().st_size) if exists else 0,  # 对存在文件记录字节大小，否则使用零表示缺失。
            "sha256": sha256_file(path) if exists else "",  # 对存在文件重算摘要，否则使用空字符串表示不可验证。
        }  # 结束成功或缺失文件身份对象。
    except OSError as error:  # 将文件系统读取异常保留到验证回执而不是吞掉。
        return {"path": str(path), "exists": False, "size_bytes": 0, "sha256": "", "error": type(error).__name__ + ": " + str(error)}  # 返回明确失败身份供检查细节使用。


def load_json_object(checks: list[dict[str, Any]], label: str, path: Path) -> dict[str, Any]:  # 读取 label 对应 path，验证其为 JSON object，并把结果加入 checks。
    identity = safe_file_identity(path)  # 在解析前记录文件存在性、大小和摘要供审计。
    try:  # 捕获文件缺失、编码错误、JSON 语法错误和顶层类型错误。
        text = path.read_text(encoding="utf-8")  # 以严格 UTF-8 读取当前提交或 artifact 文件。
        payload = json.loads(text)  # 使用标准库解析 JSON，不允许注释或尾随逗号。
        if not isinstance(payload, dict):  # 只接受顶层 JSON object，拒绝数组、标量和 null。
            raise TypeError("top-level JSON value must be an object")  # 生成稳定错误消息供失败检查记录。
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:  # 将所有预期输入错误转换为失败检查。
        add_check(checks, label + "_readable_json_object", False, {"file": identity, "error": type(error).__name__ + ": " + str(error)})  # 保存文件身份和解析错误。
        return {}  # 返回空对象使其余独立检查继续产生明确失败而非中断。
    add_check(checks, label + "_readable_json_object", True, {"file": identity})  # 记录文件存在、可解码且顶层为 object。
    return payload  # 返回已完成顶层结构验证的 JSON 映射。


def validate_profile(checks: list[dict[str, Any]], case_id: str, profile: dict[str, Any], spec: dict[str, Any], profile_path: Path) -> None:  # 校验 case_id 的 profile、spec 和来源 path 是否保持原案例边界。
    add_check(checks, case_id + "_profile_case_id", profile.get("case_id") == case_id, {"expected": case_id, "actual": profile.get("case_id"), "path": str(profile_path)})  # 拒绝路径与案例内容错配。
    add_check(checks, case_id + "_profile_execution_family", profile.get("execution_family") == EXPECTED_EXECUTION_FAMILY, {"expected": EXPECTED_EXECUTION_FAMILY, "actual": profile.get("execution_family")})  # 防止 FEN 再次被接到 CalculiX 或其他技术族。
    add_check(checks, case_id + "_profile_status", profile.get("status") == EXPECTED_PROFILE_STATUS, {"expected": EXPECTED_PROFILE_STATUS, "actual": profile.get("status")})  # 防止合成测试静默把原案例标成已执行。
    missing_facts = profile.get("missing_facts")  # 读取原案例当前证据缺口列表供类型和数量检查。
    missing_is_list = isinstance(missing_facts, list)  # 仅接受 JSON array 作为 missing_facts 合同。
    add_check(checks, case_id + "_profile_missing_facts_is_array", missing_is_list, {"actual_type": type(missing_facts).__name__})  # 明确报告缺失字段或错误类型。
    missing_rows = missing_facts if missing_is_list else []  # 类型错误时使用空列表继续生成确定性失败检查。
    expected_count = int(spec["missing_fact_count"])  # 从冻结案例合同读取五项或七项期望数量。
    add_check(checks, case_id + "_profile_missing_fact_count", len(missing_rows) == expected_count, {"expected": expected_count, "actual": len(missing_rows)})  # 要求缺失事实不能被合成结果消项。
    paths = [row.get("path") for row in missing_rows if isinstance(row, dict)]  # 仅从 object 行提取缺失事实路径供完整性检查。
    rows_valid = len(paths) == len(missing_rows) and all(isinstance(path_value, str) and bool(path_value.strip()) for path_value in paths)  # 要求每项都是带非空 path 的 object。
    add_check(checks, case_id + "_profile_missing_fact_rows", rows_valid, {"paths": paths})  # 防止用空对象填满数量门。
    add_check(checks, case_id + "_profile_missing_fact_paths_unique", rows_valid and len(set(paths)) == len(paths), {"paths": paths})  # 防止重复同一路径伪造五项或七项缺口。


def validate_recorded_files(checks: list[dict[str, Any]], case_id: str, output_root: Path, files_value: Any, expected_keys: tuple[str, ...]) -> None:  # 重算 files_value 中每个 expected_keys 文件的大小和摘要，并限制路径在 output_root 内。
    files_are_object = isinstance(files_value, dict)  # 只接受以稳定键名索引文件记录的 JSON object。
    add_check(checks, case_id + "_files_object", files_are_object, {"actual_type": type(files_value).__name__})  # 明确报告缺失 files 或错误类型。
    files = files_value if files_are_object else {}  # 类型错误时使用空对象继续逐键生成失败检查。
    expected_key_set = set(expected_keys)  # 将冻结元组转换为集合以执行无顺序键覆盖比较。
    actual_key_set = set(files.keys())  # 读取回执实际记录的所有文件键。
    add_check(checks, case_id + "_file_keys_exact", actual_key_set == expected_key_set, {"expected": sorted(expected_key_set), "actual": sorted(str(key) for key in actual_key_set)})  # 拒绝缺文件和未审计额外文件键。
    for file_key in expected_keys:  # 按冻结顺序逐一校验每个必须留存的 XDMF 或 HDF5 文件。
        record_value = files.get(file_key)  # 读取当前逻辑文件键对应的记录对象。
        record_is_object = isinstance(record_value, dict)  # 只接受包含 path、size_bytes 和 sha256 的 JSON object。
        if not record_is_object:  # 对缺失或错误类型记录生成单项失败并继续下一个文件。
            add_check(checks, case_id + "_file_" + file_key, False, {"reason": "file record must be an object", "actual_type": type(record_value).__name__})  # 保存当前键的结构错误。
            continue  # 不对没有合法记录的当前键执行路径和哈希访问。
        record = record_value  # 经过 isinstance 门后把当前值作为文件记录映射使用。
        recorded_path = record.get("path")  # 读取 runner 声称位于 artifact 根内的相对路径。
        path_is_relative_text = isinstance(recorded_path, str) and bool(recorded_path.strip()) and not Path(recorded_path).is_absolute()  # 拒绝空路径和任何绝对路径。
        candidate_path = (output_root / recorded_path).resolve() if path_is_relative_text else output_root  # 仅在路径文本合法时解析候选文件，否则使用根目录占位。
        try:  # 使用 relative_to 检查规范化后的路径没有通过点点段逃逸 artifact 根。
            candidate_path.relative_to(output_root)  # 成功时证明候选文件仍在解析后的 artifact 根内。
            path_within_root = path_is_relative_text  # 只有原文本相对且规范化后在根内才接受。
        except ValueError:  # 捕获目录穿越导致候选路径不属于 artifact 根的情况。
            path_within_root = False  # 明确拒绝越界文件，避免 validator 读取仓库或系统其他位置。
        actual_identity = safe_file_identity(candidate_path) if path_within_root else {"path": str(candidate_path), "exists": False, "size_bytes": 0, "sha256": "", "error": "path escapes artifact root"}  # 对安全路径重算身份，对越界路径返回失败对象。
        recorded_size = record.get("size_bytes")  # 读取 runner 记录的文件字节数。
        recorded_sha256 = record.get("sha256")  # 读取 runner 记录的小写 SHA-256。
        size_is_valid = type(recorded_size) is int and recorded_size > 0  # 拒绝布尔值、字符串、零和负数大小。
        digest_is_valid = isinstance(recorded_sha256, str) and len(recorded_sha256) == 64 and set(recorded_sha256).issubset(LOWERCASE_HEX_DIGITS)  # 要求完整六十四字符小写十六进制摘要。
        file_passed = path_within_root and actual_identity.get("exists") is True and size_is_valid and digest_is_valid and recorded_size == actual_identity.get("size_bytes") and recorded_sha256 == actual_identity.get("sha256")  # 只有路径、存在性、大小和摘要全部一致才通过。
        add_check(checks, case_id + "_file_" + file_key, file_passed, {"recorded": record, "actual": actual_identity, "path_within_root": path_within_root})  # 保存记录值和独立重算值供差异审计。


def validate_case_common(checks: list[dict[str, Any]], case_id: str, receipt: dict[str, Any], spec: dict[str, Any], output_root: Path) -> None:  # 对 case_id 回执执行共同边界、检查项、调用计数和文件完整性验证。
    add_check(checks, case_id + "_receipt_case_id", receipt.get("case_id") == case_id, {"expected": case_id, "actual": receipt.get("case_id")})  # 防止两个案例回执路径互换。
    benchmark_id = receipt.get("benchmark_id")  # 读取 runner 声明的合成基准标识。
    benchmark_has_syn_prefix = isinstance(benchmark_id, str) and benchmark_id.startswith("SYN-")  # 要求标识使用不可省略的 SYN 前缀。
    add_check(checks, case_id + "_benchmark_syn_prefix", benchmark_has_syn_prefix, {"actual": benchmark_id})  # 防止 artifact 被命名成原案例结果。
    add_check(checks, case_id + "_benchmark_exact", benchmark_id == spec["benchmark_id"], {"expected": spec["benchmark_id"], "actual": benchmark_id})  # 防止两个合成基准之间错配或静默改题。
    add_check(checks, case_id + "_artifact_kind", receipt.get("artifact_kind") == EXPECTED_ARTIFACT_KIND, {"expected": EXPECTED_ARTIFACT_KIND, "actual": receipt.get("artifact_kind")})  # 要求机器证据类型明确为合成合同。
    add_check(checks, case_id + "_execution_family", receipt.get("execution_family") == EXPECTED_EXECUTION_FAMILY, {"expected": EXPECTED_EXECUTION_FAMILY, "actual": receipt.get("execution_family")})  # 要求真实数值回执仍属于 FEniCS/FEniCSx。
    add_check(checks, case_id + "_contract_test_outcome", receipt.get("contract_test_outcome") == "pass", {"expected": "pass", "actual": receipt.get("contract_test_outcome")})  # 独立 validator 只接受 runner 明确报告通过的合同。
    add_check(checks, case_id + "_research_case_execution_status", receipt.get("research_case_execution_status") == EXPECTED_RESEARCH_STATUS, {"expected": EXPECTED_RESEARCH_STATUS, "actual": receipt.get("research_case_execution_status")})  # 原研究案例必须继续标为缺证阻断。
    add_check(checks, case_id + "_scientific_claim_forbidden", receipt.get("scientific_claim_allowed") is False, {"expected": False, "actual": receipt.get("scientific_claim_allowed")})  # 仅接受严格 JSON false，不接受缺失或假字符串。
    add_check(checks, case_id + "_synthetic_results_generated", receipt.get("synthetic_numeric_results_generated") is True, {"expected": True, "actual": receipt.get("synthetic_numeric_results_generated")})  # 要求通过回执明确声明已生成合成数值结果。
    provenance_value = receipt.get("provenance")  # 读取案例输入来源和当前提交身份对象。
    provenance = provenance_value if isinstance(provenance_value, dict) else {}  # 类型错误时使用空对象形成确定性失败。
    add_check(checks, case_id + "_uses_no_original_research_inputs", provenance.get("uses_original_research_inputs") is False, {"expected": False, "actual": provenance.get("uses_original_research_inputs"), "provenance_type": type(provenance_value).__name__})  # 要求每案来源明确否认使用原研究输入。
    case_checks_value = receipt.get("checks")  # 读取 runner 的预冻结数值检查数组。
    case_checks_are_array = isinstance(case_checks_value, list)  # 只接受 JSON array 作为检查列表。
    case_checks = case_checks_value if case_checks_are_array else []  # 类型错误时使用空数组继续形成失败。
    all_case_checks_pass = bool(case_checks) and all(isinstance(row, dict) and row.get("passed") is True for row in case_checks)  # 要求非空且每项是严格通过的 object。
    add_check(checks, case_id + "_runner_checks_nonempty_all_passed", case_checks_are_array and all_case_checks_pass, {"count": len(case_checks), "failed_or_invalid": [row for row in case_checks if not isinstance(row, dict) or row.get("passed") is not True]})  # 保存失败或非法行供 Actions 审计。
    execution_counts_value = receipt.get("execution_counts")  # 读取 runner 记录的实际调用计数。
    execution_counts = execution_counts_value if isinstance(execution_counts_value, dict) else {}  # 类型错误时使用空对象形成确定性失败。
    expected_solve_calls = int(spec["linear_solve_calls"])  # 从冻结合同读取 FEN-003 四次或 FEN-014 两次求解要求。
    add_check(checks, case_id + "_linear_solve_calls", execution_counts.get("linear_solve_calls") == expected_solve_calls, {"expected": expected_solve_calls, "actual": execution_counts.get("linear_solve_calls")})  # 防止只生成 receipt 而没有完成规定求解次数。
    add_check(checks, case_id + "_no_calculix_or_model_calls", execution_counts.get("calculix_calls") == 0 and execution_counts.get("model_calls") == 0, {"calculix_calls": execution_counts.get("calculix_calls"), "model_calls": execution_counts.get("model_calls")})  # 要求本合同没有串用 CalculiX 或模型服务。
    expected_file_keys = tuple(str(value) for value in spec["file_keys"])  # 将冻结文件键转换为不可变字符串元组供逐项校验。
    validate_recorded_files(checks, case_id, output_root, receipt.get("files"), expected_file_keys)  # 独立重算所有记录 artifact 的大小和摘要。


def validate_fen003_numerics(checks: list[dict[str, Any]], receipt: dict[str, Any], spec: dict[str, Any]) -> None:  # 校验 FEN-003 四层结果和每次 PETSc 收敛原因码。
    levels_value = receipt.get("level_results")  # 读取制造解四个网格层级的数值结果数组。
    levels_are_array = isinstance(levels_value, list)  # 只接受 JSON array 作为层级结果。
    levels = levels_value if levels_are_array else []  # 类型错误时使用空数组继续形成失败。
    expected_divisions = list(spec["expected_divisions"])  # 将冻结等分数元组转换为 JSON 风格列表供比较和报告。
    actual_divisions = [row.get("divisions_per_axis") for row in levels if isinstance(row, dict)]  # 从合法 object 行读取各层等分数。
    add_check(checks, "FEN-003_level_sequence", levels_are_array and len(levels) == len(expected_divisions) and actual_divisions == expected_divisions, {"expected": expected_divisions, "actual": actual_divisions, "row_count": len(levels)})  # 要求四层完整且顺序固定。
    converged_reasons = [row.get("ksp_converged_reason") for row in levels if isinstance(row, dict)]  # 读取每层 PETSc KSP 收敛原因码。
    reasons_pass = len(converged_reasons) == len(expected_divisions) and all(type(reason) is int and reason > 0 for reason in converged_reasons)  # 要求四个严格整数原因码均为 PETSc 正成功值。
    add_check(checks, "FEN-003_ksp_converged_reasons", reasons_pass, {"expected_positive_count": len(expected_divisions), "actual": converged_reasons})  # 防止仅依靠 runner 汇总通过标志。


def validate_fen014_numerics(checks: list[dict[str, Any]], receipt: dict[str, Any]) -> None:  # 校验 FEN-014 参考与重读两次 PETSc 求解的收敛原因码。
    reference_value = receipt.get("reference")  # 读取内存参考网格的数值诊断对象。
    imported_value = receipt.get("imported")  # 读取 XDMF 重读网格的数值诊断对象。
    reference = reference_value if isinstance(reference_value, dict) else {}  # 类型错误时使用空对象形成失败。
    imported = imported_value if isinstance(imported_value, dict) else {}  # 类型错误时使用空对象形成失败。
    converged_reasons = [reference.get("ksp_converged_reason"), imported.get("ksp_converged_reason")]  # 按参考后重读顺序收集两个 PETSc 原因码。
    reasons_pass = all(type(reason) is int and reason > 0 for reason in converged_reasons)  # 要求两个严格整数原因码均为 PETSc 正成功值。
    add_check(checks, "FEN-014_ksp_converged_reasons", reasons_pass, {"reference": converged_reasons[0], "imported": converged_reasons[1]})  # 防止只写两次调用计数但任一次未收敛。


def validate_campaign(checks: list[dict[str, Any]], campaign: dict[str, Any]) -> None:  # 校验整体 campaign 的合成边界、两案摘要和六次求解汇总。
    add_check(checks, "campaign_status", campaign.get("status") == "synthetic_contract_passed", {"expected": "synthetic_contract_passed", "actual": campaign.get("status")})  # 拒绝含糊的 success 或失败状态。
    add_check(checks, "campaign_contract_test_outcome", campaign.get("contract_test_outcome") == "pass", {"expected": "pass", "actual": campaign.get("contract_test_outcome")})  # 要求整体机器枚举明确为通过。
    add_check(checks, "campaign_artifact_kind", campaign.get("artifact_kind") == EXPECTED_ARTIFACT_KIND, {"expected": EXPECTED_ARTIFACT_KIND, "actual": campaign.get("artifact_kind")})  # 要求整体证据仍明确为合成数值合同。
    add_check(checks, "campaign_execution_family", campaign.get("execution_family") == EXPECTED_EXECUTION_FAMILY, {"expected": EXPECTED_EXECUTION_FAMILY, "actual": campaign.get("execution_family")})  # 要求活动没有串到 CalculiX。
    add_check(checks, "campaign_research_case_execution_status", campaign.get("research_case_execution_status") == EXPECTED_RESEARCH_STATUS, {"expected": EXPECTED_RESEARCH_STATUS, "actual": campaign.get("research_case_execution_status")})  # 要求整体重复声明原案例缺证阻断。
    add_check(checks, "campaign_uses_no_original_research_inputs", campaign.get("uses_original_research_inputs") is False, {"expected": False, "actual": campaign.get("uses_original_research_inputs")})  # 要求整体严格声明输入由 CI 合成。
    add_check(checks, "campaign_scientific_claim_forbidden", campaign.get("scientific_claim_allowed") is False, {"expected": False, "actual": campaign.get("scientific_claim_allowed")})  # 禁止绿色 campaign 被解释为科学或工程结论。
    add_check(checks, "campaign_research_cases_remain_blocked", campaign.get("research_cases_remain_blocked") is True, {"expected": True, "actual": campaign.get("research_cases_remain_blocked")})  # 要求原 profiles 的缺失事实仍具约束力。
    summaries_value = campaign.get("case_summaries")  # 读取两个合成案例的整体摘要数组。
    summaries_are_array = isinstance(summaries_value, list)  # 只接受 JSON array 作为摘要集合。
    summaries = summaries_value if summaries_are_array else []  # 类型错误时使用空数组继续形成失败。
    summaries_by_case = {row.get("case_id"): row for row in summaries if isinstance(row, dict) and isinstance(row.get("case_id"), str)}  # 按 case_id 建立摘要索引并忽略非法行。
    summaries_pass = summaries_are_array and len(summaries) == len(EXPECTED_CASE_SPECS) and set(summaries_by_case) == set(EXPECTED_CASE_SPECS)  # 要求恰好包含 FEN-003 与 FEN-014 两项。
    for case_id, spec in EXPECTED_CASE_SPECS.items():  # 逐案检查摘要的 SYN 标识、合同结果和原案例边界。
        summary = summaries_by_case.get(case_id, {})  # 缺失案例时使用空对象生成失败条件。
        summary_pass = summary.get("benchmark_id") == spec["benchmark_id"] and summary.get("contract_test_outcome") == "pass" and summary.get("research_case_execution_status") == EXPECTED_RESEARCH_STATUS and summary.get("scientific_claim_allowed") is False  # 要求摘要与详细回执边界一致。
        summaries_pass = summaries_pass and summary_pass  # 将当前案例结果合并到整体摘要检查。
    add_check(checks, "campaign_case_summaries", summaries_pass, {"actual": summaries})  # 保存原始摘要供失败时定位错配字段。
    execution_counts_value = campaign.get("execution_counts")  # 读取整体调用计数汇总。
    execution_counts = execution_counts_value if isinstance(execution_counts_value, dict) else {}  # 类型错误时使用空对象形成失败。
    campaign_counts_pass = execution_counts.get("linear_solve_calls") == 6 and execution_counts.get("calculix_calls") == 0 and execution_counts.get("model_calls") == 0  # 要求四加二共六次 FEniCSx 求解且无其他技术调用。
    add_check(checks, "campaign_execution_counts", campaign_counts_pass, {"expected_linear_solve_calls": 6, "actual": execution_counts})  # 防止整体成功与案例计数不一致。


def main() -> int:  # 组织全部独立读取和验证，始终尝试写出 validation receipt，并返回零或一。
    args = parse_args()  # 读取 Actions 显式传入的三个路径参数。
    output_root = Path(args.output_dir).resolve()  # 规范化 artifact 根供目录穿越检查和回执写入。
    validation_receipt_path = output_root / "artifact_validation_receipt.json"  # 冻结成功和失败共同使用的验证回执文件名。
    fen003_profile_path = Path(args.fen003_profile).resolve()  # 规范化当前提交的 FEN-003 profile 路径。
    fen014_profile_path = Path(args.fen014_profile).resolve()  # 规范化当前提交的 FEN-014 profile 路径。
    started_at = utc_now()  # 在任何文件读取前记录验证活动开始时间。
    checks: list[dict[str, Any]] = []  # 收集全部输入、边界、数值和文件完整性检查。
    unexpected_error = ""  # 默认没有 validator 自身异常，异常路径会写入类型和消息。
    try:  # 捕获所有非预期验证器错误以确保失败回执仍被尝试写出。
        output_root.mkdir(parents=True, exist_ok=True)  # 确保 artifact 根存在并可放置独立验证回执。
        campaign_path = output_root / "campaign_receipt.json"  # 定位 runner 生成的整体活动回执。
        fen003_receipt_path = output_root / str(EXPECTED_CASE_SPECS["FEN-003"]["receipt_relative_path"])  # 定位 FEN-003 数值回执。
        fen014_receipt_path = output_root / str(EXPECTED_CASE_SPECS["FEN-014"]["receipt_relative_path"])  # 定位 FEN-014 数值回执。
        campaign = load_json_object(checks, "campaign_receipt", campaign_path)  # 独立读取整体活动回执。
        fen003_receipt = load_json_object(checks, "fen003_numerical_contract_receipt", fen003_receipt_path)  # 独立读取 FEN-003 回执。
        fen014_receipt = load_json_object(checks, "fen014_numerical_contract_receipt", fen014_receipt_path)  # 独立读取 FEN-014 回执。
        fen003_profile = load_json_object(checks, "fen003_profile", fen003_profile_path)  # 独立读取当前权威 FEN-003 profile。
        fen014_profile = load_json_object(checks, "fen014_profile", fen014_profile_path)  # 独立读取当前权威 FEN-014 profile。
        validate_profile(checks, "FEN-003", fen003_profile, EXPECTED_CASE_SPECS["FEN-003"], fen003_profile_path)  # 检查 FEN-003 技术族、草稿状态和五项缺口。
        validate_profile(checks, "FEN-014", fen014_profile, EXPECTED_CASE_SPECS["FEN-014"], fen014_profile_path)  # 检查 FEN-014 技术族、草稿状态和七项缺口。
        validate_case_common(checks, "FEN-003", fen003_receipt, EXPECTED_CASE_SPECS["FEN-003"], output_root)  # 检查 FEN-003 合成边界、四次求解和文件记录。
        validate_case_common(checks, "FEN-014", fen014_receipt, EXPECTED_CASE_SPECS["FEN-014"], output_root)  # 检查 FEN-014 合成边界、两次求解和文件记录。
        validate_fen003_numerics(checks, fen003_receipt, EXPECTED_CASE_SPECS["FEN-003"])  # 独立检查四个 KSP 正收敛原因码。
        validate_fen014_numerics(checks, fen014_receipt)  # 独立检查参考和重读两个 KSP 正收敛原因码。
        validate_campaign(checks, campaign)  # 检查整体 campaign 与两个详细回执保持一致边界。
    except BaseException as error:  # 兜底捕获编程错误、资源错误和意外输入组合，避免无回执失败。
        unexpected_error = type(error).__name__ + ": " + str(error)  # 保存异常类型和消息供 Actions 调试。
        add_check(checks, "validator_unexpected_exception", False, {"error": unexpected_error})  # 将 validator 自身异常转换为明确失败检查。
    passed = validation_passes(checks) and not unexpected_error  # 只有全部检查通过且没有兜底异常时接受 artifact。
    failed_check_names = [str(check.get("name", "unnamed_check")) for check in checks if check.get("passed") is not True]  # 收集所有失败检查名称供主索引快速定位。
    receipt = {  # 构造成功和失败共同使用的独立 artifact 验证回执。
        "schema_version": "v5-fenicsx-synthetic-artifact-validation/1.0",  # 冻结 validator 输出合同版本。
        "status": "artifact_validation_passed" if passed else "artifact_validation_failed",  # 使用不会被误读为原案例成功的状态文本。
        "validation_outcome": "pass" if passed else "fail",  # 提供 Actions 和后续脚本可稳定消费的枚举。
        "artifact_kind": EXPECTED_VALIDATION_KIND,  # 明确该文件是独立验证证据而非数值结果。
        "started_at_utc": started_at,  # 保存验证活动开始时间。
        "finished_at_utc": utc_now(),  # 保存完成全部检查后的结束时间。
        "output_dir": str(output_root),  # 保存被验证 artifact 根的规范化路径。
        "profile_inputs": {"FEN-003": safe_file_identity(fen003_profile_path), "FEN-014": safe_file_identity(fen014_profile_path)},  # 保存两个权威 profile 的实际大小和摘要。
        "check_count": len(checks),  # 保存实际执行的独立检查数量。
        "failed_check_count": len(failed_check_names),  # 保存失败检查总数供快速门控。
        "failed_check_names": failed_check_names,  # 保存所有失败检查稳定名称。
        "checks": checks,  # 保存完整通过与失败细节供 artifact 审计。
        "unexpected_error": unexpected_error,  # 无异常时为空字符串，有异常时保存类型和消息。
        "scientific_claim_allowed": False,  # validator 本身同样不能支持原研究或工程结论。
        "research_case_execution_status": EXPECTED_RESEARCH_STATUS,  # 重申独立校验不改变两个原案例状态。
        "allowed_use": "仅用于确认合成数值 artifact 与当前未执行 profiles 的机器边界和文件完整性。",  # 限定验证回执的唯一允许用途。
        "disallowed_use": "不得据此宣称原 FEN-003 或 FEN-014 研究案例已执行、通过或形成科学结论。",  # 明确禁止冒充原案例复现。
    }  # 结束独立验证回执。
    try:  # 独立捕获最终回执写入错误并确保进程返回失败。
        write_json(validation_receipt_path, receipt)  # 无论 passed 为真或假都写到固定 artifact 路径。
    except BaseException as error:  # 文件系统不可写时无法满足留证合同，必须输出错误并失败。
        print(json.dumps({"status": "artifact_validation_receipt_write_failed", "path": str(validation_receipt_path), "error": type(error).__name__ + ": " + str(error)}, ensure_ascii=False), file=sys.stderr)  # 在 Actions stderr 保留最后可用证据。
        return 1  # 回执写入失败无条件返回一，禁止工作流误绿。
    print(json.dumps({"status": receipt["status"], "validation_outcome": receipt["validation_outcome"], "failed_check_names": failed_check_names, "receipt": str(validation_receipt_path)}, ensure_ascii=False))  # 在 Actions stdout 输出精简机器摘要。
    return 0 if passed else 1  # 全部严格检查通过返回零，否则返回一使 Actions 失败。


if __name__ == "__main__":  # 仅在 Actions 或维护者直接调用本文件时执行主流程。
    raise SystemExit(main())  # 将 main 的零或一精确传递给调用 shell 和 GitHub Actions。
