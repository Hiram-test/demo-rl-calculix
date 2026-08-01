#!/usr/bin/env python3
"""验证 V5 轻量前端、提高层档案和既有九项合同测试，并始终写出审计 receipt。"""  # 说明脚本只做静态与合同验证，不执行工程计算。
from __future__ import annotations  # 启用现代类型注解行为，兼容 Python 三点十二运行环境。

import argparse  # 解析工作流传入的 Node 退出码和 receipt 路径。
import hashlib  # 计算源文件 SHA-256 以形成可追溯快照。
import io  # 捕获 unittest 文本输出并写入 receipt。
import json  # 读取 profile 与 Skill JSON 并写出验证 receipt。
import os  # 切换到仓库根目录并读取 GitHub Actions 环境字段。
import re  # 检查前端标签、禁止能力和带单位数值。
import shutil  # 将已验证静态文件复制到单一 artifact 目录。
import sys  # 配置源码导入路径并返回明确退出状态。
import unittest  # 精确运行现有 ProblemManifest 与 DecisionLoop 九项测试。
from pathlib import Path  # 使用跨平台路径对象定位仓库文件。
from typing import Any  # 描述递归 JSON 值与检查结果结构。

REPO_ROOT = Path(__file__).resolve().parents[1]  # 以脚本所在 scripts 目录的父目录作为仓库根。
EXPECTED_TEST_COUNT = 9  # 当前基线由五项 ProblemManifest 测试和四项 DecisionLoop 测试组成。
PROFILE_PATHS = [  # 冻结本切片恰好包含的三个提高层档案。
    Path("examples/v5_high_layer/fen-003.json"),  # 指向网格收敛与误差判断档案。
    Path("examples/v5_high_layer/fen-014.json"),  # 指向外部导入网格可信性档案。
    Path("examples/v5_high_layer/ccx-015.json"),  # 指向受质量约束自动细化档案。
]  # 结束提高层档案路径列表。
FRONTEND_PATHS = [  # 冻结轻量前端的四个无依赖文件。
    Path("frontend-lite/index.html"),  # 指向语义化单页结构。
    Path("frontend-lite/styles.css"),  # 指向本地响应式样式。
    Path("frontend-lite/app.js"),  # 指向无网络经典浏览器脚本。
    Path("frontend-lite/README.md"),  # 指向字段、状态和使用边界说明。
]  # 结束轻量前端路径列表。
DOCUMENTATION_PATHS = [Path("docs/v5_high_layer/HIGH_LAYER_LITE.md")]  # 冻结三个 JSON 档案的逐项配套说明。
SUPPORT_PATHS = [  # 冻结复现本验证所需的验证器和新工作流。
    Path("scripts/validate_v5_frontend_high_layer_lite.py"),  # 包含当前纯标准库验证逻辑。
    Path(".github/workflows/v5-frontend-high-layer-lite.yml"),  # 包含当前 GitHub Actions 执行合同。
]  # 结束复现支持文件路径列表。
REQUIRED_PROFILE_FIELDS = {  # 冻结 profile 的十四个顶层字段，避免悄悄加入案例参数。
    "profile_version",  # 记录静态档案合同版本。
    "case_id",  # 记录研究计划案例标识。
    "execution_family",  # 记录用户明确确认的 FEniCS/FEniCSx 或 CalculiX 执行技术族。
    "status",  # 记录档案尚未执行的固定状态。
    "objective",  # 记录要验证的工程判断能力。
    "decision_question",  # 记录最终要回答的工程决策问题。
    "applicable_skill_ids",  # 记录当前仓库已存在的通用 Skill 映射。
    "skill_coverage_gaps",  # 记录尚未实现的通用能力缺口。
    "required_evidence",  # 记录证据类型、用途和来源要求。
    "minimal_experiment",  # 记录未执行的最小区分性实验设计。
    "fixed_qoi_contract_fields",  # 记录未来必须冻结的 QoI 字段名。
    "stop_condition_fields",  # 记录未来必须建立的停止与回退字段名。
    "missing_facts",  # 记录当前输入尚未建立的事实和用户问题。
    "interpretation_limits",  # 记录不得越过的解释边界。
}  # 结束 profile 顶层字段集合。
EXPECTED_SKILL_MAPPINGS = {  # 冻结三个档案只引用当前已有 Skill 的精确映射。
    "FEN-003": {"problem-definition-source-audit", "mesh-convergence-and-singularity"},  # FEN-003 使用来源审计和网格收敛 Skill。
    "FEN-014": {"problem-definition-source-audit"},  # FEN-014 当前只能使用来源审计 Skill。
    "CCX-015": {"problem-definition-source-audit", "mesh-convergence-and-singularity", "optimization-readiness"},  # CCX-015 使用三个现有通用 Skill。
}  # 结束 Skill 精确映射。
EXPECTED_EXECUTION_FAMILIES = {  # 冻结用户明确确认的案例与执行技术族映射。
    "FEN-003": "FEniCS/FEniCSx",  # FEN-003 必须使用 FEniCS/FEniCSx 路线。
    "FEN-014": "FEniCS/FEniCSx",  # FEN-014 必须使用 FEniCS/FEniCSx 路线。
    "CCX-015": "CalculiX",  # CCX-015 必须使用 CalculiX 路线。
}  # 结束执行技术族精确映射。
EXPECTED_SKILL_GAPS = {  # 冻结当前三个 profile 公开声明的能力缺口。
    "FEN-003": [],  # FEN-003 当前通用 Skill 已覆盖入口推理框架。
    "FEN-014": ["fenics-imported-mesh-integrity-and-tag-mapping-diagnosis"],  # FEN-014 缺少 FEniCS 网格与标签映射诊断。
    "CCX-015": ["mesh-quality-constrained-refinement-and-rollback"],  # CCX-015 缺少质量约束细化和回退能力。
}  # 结束能力缺口精确映射。
BANNED_EXACT_PLACEHOLDERS = {"tbd", "unknown", "待定"}  # 禁止用模糊占位值替代 missing_facts。
QUANTITY_WITH_UNIT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|mpa|gpa|n|kn|s|ms|%)\b", re.IGNORECASE)  # 拒绝 profile 中任何带常见物理单位的数值。
FORBIDDEN_FRONTEND_TOKENS = [  # 冻结轻量页面不得使用的网络、持久化和不安全渲染能力。
    "fetch(",  # 禁止浏览器发起 fetch 网络请求。
    "XMLHttpRequest",  # 禁止旧式浏览器网络请求。
    "WebSocket",  # 禁止建立实时网络连接。
    "localStorage",  # 禁止持久化用户工程输入。
    "sessionStorage",  # 禁止会话级持久化用户工程输入。
    ".innerHTML",  # 禁止把用户或证据文本作为 HTML 执行。
    "http://",  # 禁止加载明文外部资源。
    "https://",  # 禁止加载加密外部资源或调用外部 API。
]  # 结束前端禁止能力列表。


def sha256_file(path: Path) -> str:  # 定义文件哈希函数，输入仓库文件路径并返回十六进制 SHA-256。
    digest = hashlib.sha256()  # 创建新的 SHA-256 摘要器以避免跨文件混合状态。
    with path.open("rb") as handle:  # 以只读二进制模式打开文件以保持字节稳定。
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):  # 每次读取一 MiB，限制哈希计算的内存峰值。
            digest.update(chunk)  # 将当前文件块加入摘要计算。
    return digest.hexdigest()  # 返回六十四字符十六进制摘要。


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:  # 记录一个命名检查的布尔结果和可审计细节。
    checks.append({"name": name, "passed": bool(passed), "details": details})  # 将标准化检查对象追加到 receipt 列表。


def walk_json(value: Any, path: str = "$") -> list[tuple[str, Any]]:  # 递归展开 JSON，返回每个叶子值及其逻辑路径。
    leaves: list[tuple[str, Any]] = []  # 创建当前 JSON 的叶子值收集列表。
    if isinstance(value, dict):  # 对 JSON 对象递归检查每个键值。
        for key, child in value.items():  # 按文件原有顺序遍历对象字段。
            leaves.extend(walk_json(child, path + "." + str(key)))  # 使用点号路径记录子字段来源。
        return leaves  # 返回对象中全部递归叶子值。
    if isinstance(value, list):  # 对 JSON 数组递归检查每个下标。
        for index, child in enumerate(value):  # 按零基下标遍历数组。
            leaves.extend(walk_json(child, path + "[" + str(index) + "]"))  # 使用方括号路径记录数组来源。
        return leaves  # 返回数组中全部递归叶子值。
    leaves.append((path, value))  # 将标量或 null 作为叶子值记录。
    return leaves  # 返回当前单个叶子值列表。


def validate_required_files(checks: list[dict[str, Any]]) -> list[Path]:  # 检查本切片全部文件存在且非空，并返回实际存在文件。
    required = FRONTEND_PATHS + PROFILE_PATHS + DOCUMENTATION_PATHS + SUPPORT_PATHS  # 合并前端、档案、说明和复现支持路径。
    existing: list[Path] = []  # 创建实际存在文件列表供 artifact 快照使用。
    missing: list[str] = []  # 创建缺失或空文件列表供 receipt 报告。
    for relative in required:  # 逐个检查冻结路径。
        absolute = REPO_ROOT / relative  # 将仓库相对路径解析为绝对路径。
        if absolute.is_file() and absolute.stat().st_size > 0:  # 仅把存在且字节数大于零的普通文件视为有效。
            existing.append(relative)  # 记录可复制和计算哈希的有效文件。
        else:  # 文件缺失、非普通文件或为空时进入失败列表。
            missing.append(relative.as_posix())  # 记录稳定的仓库相对路径。
    add_check(checks, "required_files_exist_and_are_nonempty", not missing, {"required_count": len(required), "missing": missing})  # 写入文件完整性检查。
    return existing  # 返回实际存在文件供后续处理。


def validate_frontend(checks: list[dict[str, Any]], node_check_exit_code: int) -> None:  # 验证 Node 语法退出码和静态页面安全边界。
    add_check(checks, "node_check_app_js", node_check_exit_code == 0, {"exit_code": node_check_exit_code})  # 记录工作流真实 node --check 结果。
    if not all((REPO_ROOT / path).is_file() for path in FRONTEND_PATHS):  # 任一前端文件缺失时无法继续内容检查。
        add_check(checks, "frontend_contract", False, "required frontend file is missing")  # 明确记录前端合同无法检查。
        return  # 避免读取不存在文件导致验证器中断。
    index_text = (REPO_ROOT / FRONTEND_PATHS[0]).read_text(encoding="utf-8")  # 读取 UTF-8 HTML 页面。
    app_text = (REPO_ROOT / FRONTEND_PATHS[2]).read_text(encoding="utf-8")  # 读取 UTF-8 经典浏览器脚本。
    local_assets = 'href="styles.css"' in index_text and 'src="app.js"' in index_text and " defer" in index_text  # 要求 HTML 只引用两个同目录资源且脚本延迟执行。
    classic_script = 'type="module"' not in index_text  # 要求 app.js 保持可由 node --check 检查的经典脚本。
    imp_tag_match = re.search(r"<input[^>]*id=\"imp-files\"[^>]*>", index_text)  # 定位 IMP 文件输入标签。
    imp_tag = imp_tag_match.group(0) if imp_tag_match else ""  # 缺失标签时使用空文本并由后续断言失败。
    imp_contract = bool(imp_tag) and 'type="file"' in imp_tag and " multiple" in imp_tag and " accept=" not in imp_tag  # 要求多文件选择且不预设未知 IMP 格式。
    boundary_texts = all(text in index_text for text in ["本地静态预览", "未连接后端", "未启动网格或求解"])  # 要求页面永久显示三项运行边界。
    forbidden_hits = [token for token in FORBIDDEN_FRONTEND_TOKENS if token in app_text]  # 查找脚本中的网络、持久化和不安全渲染能力。
    passed = local_assets and classic_script and imp_contract and boundary_texts and not forbidden_hits  # 汇总前端合同检查结果。
    details = {"local_assets": local_assets, "classic_script": classic_script, "imp_contract": imp_contract, "boundary_texts": boundary_texts, "forbidden_hits": forbidden_hits}  # 保存每个前端断言的独立结果。
    add_check(checks, "frontend_contract", passed, details)  # 写入前端静态合同检查。


def load_skill_ids() -> set[str]:  # 从当前真实 Skill JSON 目录加载全部 skill_id 并返回集合。
    skill_ids: set[str] = set()  # 创建去重 Skill ID 集合。
    for path in sorted((REPO_ROOT / "skills/engineering").glob("*.json")):  # 按文件名稳定遍历当前通用 Skill。
        payload = json.loads(path.read_text(encoding="utf-8"))  # 解析当前 Skill JSON。
        skill_id = str(payload.get("skill_id", "")).strip()  # 读取并清理 Skill ID。
        if skill_id:  # 仅记录非空 Skill ID。
            skill_ids.add(skill_id)  # 将真实 Skill ID 加入集合。
    return skill_ids  # 返回当前仓库真实 Skill ID 集合。


def validate_profiles(checks: list[dict[str, Any]]) -> None:  # 验证三个提高层档案的闭合合同、Skill 映射和无物理默认值边界。
    if not all((REPO_ROOT / path).is_file() for path in PROFILE_PATHS):  # 任一 profile 缺失时无法继续内容检查。
        add_check(checks, "high_layer_profiles", False, "required profile file is missing")  # 明确记录 profile 合同无法检查。
        return  # 避免读取不存在文件导致验证器中断。
    skill_ids = load_skill_ids()  # 加载当前仓库真实 Skill ID。
    errors: list[str] = []  # 创建全部 profile 错误列表以一次性报告。
    seen_case_ids: set[str] = set()  # 创建已读取案例标识集合。
    for relative in PROFILE_PATHS:  # 逐个验证三个冻结 profile。
        payload = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))  # 解析当前 profile JSON。
        case_id = str(payload.get("case_id", ""))  # 读取案例标识供映射和错误定位。
        seen_case_ids.add(case_id)  # 记录当前案例标识。
        if set(payload) != REQUIRED_PROFILE_FIELDS:  # 要求顶层字段集合精确闭合。
            errors.append(relative.as_posix() + ": top-level fields do not match the frozen contract")  # 记录闭合字段失败。
        if payload.get("profile_version") != "1.1":  # 加入执行技术族后要求 profile 合同版本同步升级。
            errors.append(relative.as_posix() + ": profile_version must be 1.1")  # 记录旧合同版本或无版本升级。
        if payload.get("status") != "draft_not_executed":  # 要求档案始终明确未执行。
            errors.append(relative.as_posix() + ": status must be draft_not_executed")  # 记录状态越界。
        execution_family = str(payload.get("execution_family", ""))  # 读取当前 profile 的机器可读执行技术族。
        expected_execution_family = EXPECTED_EXECUTION_FAMILIES.get(case_id, "")  # 读取用户确认的案例执行技术族映射。
        if execution_family != expected_execution_family:  # 要求 FEN 与 CCX 前缀不再接错执行后端。
            errors.append(relative.as_posix() + ": execution_family does not match the user-confirmed mapping")  # 记录执行技术族映射错误。
        serialized_profile = json.dumps(payload, ensure_ascii=False)  # 序列化 profile 以检查交叉技术族污染。
        if case_id.startswith("FEN-") and ("CalculiX" in serialized_profile or "current input deck" in serialized_profile):  # FEN 案例不得出现 CalculiX 或输入 deck 执行语义。
            errors.append(relative.as_posix() + ": FEN profile contains CalculiX semantics")  # 记录 FEniCS 案例误接 CalculiX。
        if case_id.startswith("CCX-") and ("FEniCS" in serialized_profile or "DOLFIN" in serialized_profile):  # CCX 案例不得出现 FEniCS 或 DOLFIN 执行语义。
            errors.append(relative.as_posix() + ": CCX profile contains FEniCS semantics")  # 记录 CalculiX 案例误接 FEniCS。
        applicable = set(payload.get("applicable_skill_ids", []))  # 读取 profile 声明的现有 Skill 映射。
        if applicable != EXPECTED_SKILL_MAPPINGS.get(case_id, set()):  # 要求映射与冻结案例定位完全一致。
            errors.append(relative.as_posix() + ": applicable_skill_ids do not match the case mapping")  # 记录 Skill 映射错误。
        if not applicable.issubset(skill_ids):  # 要求每个引用都存在于当前真实 Skill 库。
            errors.append(relative.as_posix() + ": applicable_skill_ids contain missing skills")  # 记录不存在的 Skill 引用。
        gaps = payload.get("skill_coverage_gaps", [])  # 读取通用能力缺口列表。
        if gaps != EXPECTED_SKILL_GAPS.get(case_id, []):  # 要求 FEniCS 与 CalculiX 能力缺口不被混淆或静默删除。
            errors.append(relative.as_posix() + ": skill_coverage_gaps do not match the frozen mapping")  # 记录能力缺口映射错误。
        missing_facts = payload.get("missing_facts", [])  # 读取当前案例缺失事实列表。
        missing_paths = {str(item.get("path", "")) for item in missing_facts if isinstance(item, dict)} if isinstance(missing_facts, list) else set()  # 提取当前缺失事实路径供运行时合同检查。
        if case_id.startswith("FEN-") and "runtime.fenics_variant_version_and_entrypoint" not in missing_paths:  # FEN 案例必须显式保留 FEniCS 变体、版本和入口缺口。
            errors.append(relative.as_posix() + ": FEN runtime missing fact is absent")  # 记录 FEniCS 运行时被错误默认。
        if case_id == "CCX-015" and not {"runtime.calculix_version_command_and_input", "mesh.refinement_toolchain"}.issubset(missing_paths):  # CCX 必须分别确认 CalculiX 运行时和细化工具链。
            errors.append(relative.as_posix() + ": CalculiX runtime or refinement toolchain missing fact is absent")  # 记录 CalculiX 或网格器被错误默认。
        if not isinstance(missing_facts, list) or not missing_facts:  # 要求每个未执行档案至少包含一项缺失事实。
            errors.append(relative.as_posix() + ": missing_facts must be a non-empty list")  # 记录缺失事实合同失败。
        else:  # 缺失事实列表存在时逐项验证结构。
            for index, item in enumerate(missing_facts):  # 按零基下标检查每条缺失事实。
                if not isinstance(item, dict) or set(item) != {"path", "reason", "question", "acceptable_sources"}:  # 要求严格沿用现有 MissingFact 四字段结构。
                    errors.append(relative.as_posix() + ": missing_facts[" + str(index) + "] has an invalid structure")  # 记录结构错误位置。
        for field_name in ["fixed_qoi_contract_fields", "stop_condition_fields", "interpretation_limits", "required_evidence"]:  # 检查必须存在且非空的列表字段。
            field_value = payload.get(field_name)  # 读取当前列表字段。
            if not isinstance(field_value, list) or not field_value:  # 空列表不能形成可执行后续合同。
                errors.append(relative.as_posix() + ": " + field_name + " must be a non-empty list")  # 记录空列表错误。
        for leaf_path, leaf_value in walk_json(payload):  # 遍历 profile 的全部叶子值。
            if leaf_value is None:  # null 不能替代显式 missing facts。
                errors.append(relative.as_posix() + ": null leaf at " + leaf_path)  # 记录 null 位置。
            elif isinstance(leaf_value, (int, float, bool)):  # profile 禁止数值和布尔案例默认值。
                errors.append(relative.as_posix() + ": numeric or boolean leaf at " + leaf_path)  # 记录非字符串叶子位置。
            elif isinstance(leaf_value, str):  # 对字符串检查空值、模糊占位和带单位数值。
                stripped = leaf_value.strip()  # 去除首尾空白进行稳定比较。
                if not stripped:  # 空字符串不能代替 missing fact。
                    errors.append(relative.as_posix() + ": empty string at " + leaf_path)  # 记录空字符串位置。
                if stripped.lower() in BANNED_EXACT_PLACEHOLDERS:  # 禁止模糊占位值。
                    errors.append(relative.as_posix() + ": banned placeholder at " + leaf_path)  # 记录模糊占位位置。
                if QUANTITY_WITH_UNIT.search(stripped):  # 禁止带常见物理单位的数值进入 profile。
                    errors.append(relative.as_posix() + ": physical quantity default at " + leaf_path)  # 记录物理数量位置。
    if seen_case_ids != set(EXPECTED_SKILL_MAPPINGS):  # 要求恰好出现三个冻结案例标识。
        errors.append("case IDs do not match FEN-003, FEN-014, and CCX-015")  # 记录案例集合错误。
    documentation_text = (REPO_ROOT / DOCUMENTATION_PATHS[0]).read_text(encoding="utf-8") if (REPO_ROOT / DOCUMENTATION_PATHS[0]).is_file() else ""  # 读取当前配套说明以验证用户确认映射可见。
    if "FEN-* = FEniCS/FEniCSx" not in documentation_text or "CCX-* = CalculiX" not in documentation_text:  # 文档必须同时公开两条执行技术族映射。
        errors.append("documentation does not state both execution-family mappings")  # 记录说明文档仍可能误导执行后端。
    add_check(checks, "high_layer_profiles", not errors, {"profile_count": len(PROFILE_PATHS), "skill_ids": sorted(skill_ids), "errors": errors})  # 写入 profile 综合检查。


def run_existing_v5_tests(checks: list[dict[str, Any]]) -> None:  # 精确运行当前两份 V5 合同测试并记录九项基线。
    os.chdir(REPO_ROOT)  # 切换到仓库根以匹配现有 workflow 的发现语义。
    src_path = str(REPO_ROOT / "src")  # 计算 engineering_agent 源码目录文本。
    if src_path not in sys.path:  # 避免重复插入源码路径。
        sys.path.insert(0, src_path)  # 将 src 放在导入搜索路径首位。
    loader = unittest.TestLoader()  # 创建标准库测试加载器。
    suite = unittest.TestSuite()  # 创建只包含两份 V5 测试的组合套件。
    suite.addTests(loader.discover("tests", pattern="test_problem_manifest.py"))  # 加载五项 ProblemManifest 测试。
    suite.addTests(loader.discover("tests", pattern="test_decision_loop.py"))  # 加载四项 DecisionLoop 测试。
    discovered_count = suite.countTestCases()  # 记录实际发现的测试数量。
    output = io.StringIO()  # 创建内存文本流保存详细测试结果。
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)  # 运行组合套件并保留每项结果。
    passed = discovered_count == EXPECTED_TEST_COUNT and result.wasSuccessful()  # 要求恰好九项且零失败零错误。
    details = {"expected_count": EXPECTED_TEST_COUNT, "discovered_count": discovered_count, "failure_count": len(result.failures), "error_count": len(result.errors), "output": output.getvalue()}  # 汇总测试数量和完整文本证据。
    add_check(checks, "existing_v5_contract_tests", passed, details)  # 写入现有 V5 测试检查。


def copy_source_snapshot(paths: list[Path], artifact_root: Path) -> list[dict[str, Any]]:  # 复制实际存在文件并返回大小与哈希清单。
    snapshot_root = artifact_root / "source"  # 将源码快照放在 artifact 的 source 子目录。
    files: list[dict[str, Any]] = []  # 创建机器可读文件清单。
    for relative in paths:  # 逐个复制已确认存在的仓库文件。
        source = REPO_ROOT / relative  # 解析源文件绝对路径。
        destination = snapshot_root / relative  # 保留仓库相对目录结构作为目标路径。
        destination.parent.mkdir(parents=True, exist_ok=True)  # 创建目标父目录但不触碰仓库源文件。
        shutil.copy2(source, destination)  # 复制文件内容和基础元数据到 Actions artifact。
        files.append({"path": relative.as_posix(), "size_bytes": source.stat().st_size, "sha256": sha256_file(source)})  # 记录源路径、字节数和 SHA-256。
    return files  # 返回源码快照机器清单。


def parse_args() -> argparse.Namespace:  # 定义命令行参数并返回解析结果。
    parser = argparse.ArgumentParser(description="Validate the V5 frontend and high-layer lite slice.")  # 创建只描述静态验证用途的解析器。
    parser.add_argument("--node-check-exit-code", type=int, required=True, help="Exit code from node --check frontend-lite/app.js.")  # 接收工作流真实 Node 语法检查退出码。
    parser.add_argument("--receipt", required=True, help="Path for the always-written validation receipt.")  # 接收验证 receipt 输出路径。
    return parser.parse_args()  # 解析当前命令行参数。


def main() -> int:  # 执行全部静态检查、现有测试、源码快照和 receipt 写入，并返回零或一。
    args = parse_args()  # 读取工作流传入的真实检查状态和输出路径。
    checks: list[dict[str, Any]] = []  # 创建顺序稳定的检查结果列表。
    existing_paths = validate_required_files(checks)  # 检查文件完整性并取得可复制路径。
    try:  # 捕获前端检查异常并继续生成 receipt。
        validate_frontend(checks, args.node_check_exit_code)  # 验证 Node 退出码和前端边界。
    except Exception as exc:  # 将任何检查器异常记录为失败证据。
        add_check(checks, "frontend_contract", False, {"validator_error": type(exc).__name__ + ": " + str(exc)})  # 保存异常类型和文本。
    try:  # 捕获 profile 检查异常并继续生成 receipt。
        validate_profiles(checks)  # 验证三个未执行诊断档案。
    except Exception as exc:  # 将任何 profile 检查器异常记录为失败证据。
        add_check(checks, "high_layer_profiles", False, {"validator_error": type(exc).__name__ + ": " + str(exc)})  # 保存异常类型和文本。
    try:  # 捕获 unittest 加载或执行异常并继续生成 receipt。
        run_existing_v5_tests(checks)  # 精确运行九项现有 V5 合同测试。
    except Exception as exc:  # 将测试基础设施异常记录为失败证据。
        add_check(checks, "existing_v5_contract_tests", False, {"validator_error": type(exc).__name__ + ": " + str(exc)})  # 保存异常类型和文本。
    receipt_path = Path(args.receipt).resolve()  # 将工作流提供的 receipt 路径解析为绝对路径。
    artifact_root = receipt_path.parent  # 使用 receipt 父目录作为单一 artifact 根。
    artifact_root.mkdir(parents=True, exist_ok=True)  # 创建 artifact 目录供失败和成功证据共同使用。
    file_manifest = copy_source_snapshot(existing_paths, artifact_root)  # 复制实际存在文件并生成哈希清单。
    valid = bool(checks) and all(bool(item.get("passed")) for item in checks)  # 仅当存在检查且每项通过时标记静态验证有效。
    receipt = {  # 构造轻量验证机器 receipt。
        "schema_version": "v5-frontend-high-layer-lite-validation/1.0",  # 冻结本 receipt 的静态合同版本。
        "status": "success" if valid else "failure",  # 根据全部检查结果记录成功或失败。
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),  # 记录 GitHub Actions 提供的仓库标识。
        "ref": os.environ.get("GITHUB_REF", ""),  # 记录 GitHub Actions 提供的完整引用。
        "head_sha": os.environ.get("GITHUB_SHA", ""),  # 记录当前验证提交 SHA。
        "checks": checks,  # 保存每项静态检查和九项测试证据。
        "source_files": file_manifest,  # 保存复制到 artifact 的源文件哈希清单。
        "execution_boundaries": {  # 明确本工作流没有执行的能力。
            "dependencies_installed": False,  # 本工作流没有运行 pip、npm 或 apt 安装。
            "deepseek_calls": 0,  # 本工作流没有模型凭证和模型调用。
            "fenics_calls": 0,  # 本静态工作流没有导入或运行 FEniCS/FEniCSx。
            "calculix_calls": 0,  # 本工作流没有安装或调用 CalculiX。
            "mesh_generation_calls": 0,  # 本工作流没有生成任何网格。
            "engineering_results_generated": False,  # 本工作流没有生成工程结果或结论。
            "validation_scope": "static_frontend_profiles_and_existing_v5_contract_tests",  # 说明成功状态只覆盖静态切片和既有合同测试。
        },  # 结束执行边界对象。
    }  # 结束验证 receipt 对象。
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 使用 UTF-8 和两空格缩进写出始终可读的 receipt。
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path), "check_count": len(checks)}, ensure_ascii=False))  # 在 Actions 日志输出精简结果位置和检查数量。
    return 0 if valid else 1  # 全部检查通过时返回零，否则返回一使 workflow 明确失败。


if __name__ == "__main__":  # 仅在脚本直接执行时运行验证入口。
    raise SystemExit(main())  # 将 main 返回值传递给 GitHub Actions 作为真实退出状态。
