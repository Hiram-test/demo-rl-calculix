#!/usr/bin/env python3  # 使用 GitHub Actions 自带的 Python 运行三案例就绪性测试。
"""执行 V5 提高层三案例 T0 就绪性测试，并在任何工程计算前保持 fail-closed。"""  # 说明本脚本只验证输入门和证据边界。
from __future__ import annotations  # 启用现代类型注解行为，兼容 Actions 的 Python 三点十二环境。

import argparse  # 解析工作流显式提供的 profile、Skill 和 artifact 路径。
import hashlib  # 为每个当前仓库来源文件计算 SHA-256 追溯标识。
import json  # 读取 profile 与 Skill JSON，并写出机器可读 receipt。
import os  # 读取 GitHub Actions 提供的仓库、提交和运行元数据。
import shutil  # 将本次实际读取的来源文件复制到 artifact 源码快照。
import sys  # 把仓库 src 目录加入当前进程的模块搜索路径。
from pathlib import Path  # 用跨平台路径对象定位仓库文件与证据目录。
from typing import Any  # 描述 JSON 对象、检查项和 receipt 的递归值。

REPO_ROOT = Path(__file__).resolve().parents[1]  # 以 scripts 目录的父目录作为当前仓库根目录。
SRC_ROOT = REPO_ROOT / "src"  # 指向当前 V5 engineering_agent 源码目录。
if str(SRC_ROOT) not in sys.path:  # 只在源码目录尚未注册时修改导入路径。
    sys.path.insert(0, str(SRC_ROOT))  # 把当前提交的 src 放到搜索路径首位，避免导入外部同名包。

from engineering_agent.problem_manifest import ProblemManifest  # 复用当前 ProblemManifest 的缺失事实和来源合同。
from engineering_agent.skill_contract import SkillLibrary  # 复用当前 Skill 解析和完整性检查合同。

CASE_CONTRACTS: dict[str, dict[str, Any]] = {  # 冻结三个 T0 测试应保留的非数值前置条件。
    "FEN-003": {  # 定义“是否继续加密”案例的入口合同。
        "profile_name": "fen-003.json",  # 指向当前仓库中 FEN-003 的静态 profile 文件名。
        "execution_family": "FEniCS/FEniCSx",  # 冻结用户确认的 FEN-003 执行技术族映射。
        "expected_skills": (  # 冻结 profile 当前必须引用的两个通用 Skill。
            "problem-definition-source-audit",  # 要求先审计当前问题事实和来源。
            "mesh-convergence-and-singularity",  # 要求后续区分收敛、提取和奇异性。
        ),  # 结束 FEN-003 Skill 标识元组。
        "expected_gaps": (),  # FEN-003 当前不声明额外通用 Skill 缺口。
        "missing_paths": (  # 冻结真实数值阶段开始前必须补齐的五个事实路径。
            "runtime.fenics_variant_version_and_entrypoint",  # 要求确认 FEniCS 变体、版本、入口和底层求解配置。
            "model.current_input",  # 要求当前权威模型或求解器输入。
            "qoi.current_definition",  # 要求完整且固定的当前 QoI 提取合同。
            "mesh_sequence.current_evidence",  # 要求同一模型的受控网格序列与真实结果。
            "convergence_criterion.current_source",  # 要求停止或继续规则及其当前来源。
        ),  # 结束 FEN-003 缺失事实路径元组。
        "expected_stop_field_count": 7,  # profile 当前列出七个必须赋值的停止条件字段。
        "phase_names": (  # 定义因来源审计阻断而不得执行的后续阶段。
            "mesh_sequence_audit",  # 未提供当前网格序列时不得比较网格层级。
            "mesh_generation",  # 未冻结输入与判据时不得生成新网格。
            "fenics_execution",  # 未通过前置门时不得运行 FEniCS/FEniCSx 问题。
            "qoi_extraction",  # 没有求解结果时不得声称提取 QoI。
            "engineering_decision",  # 没有证据时不得输出停止或继续的工程结论。
        ),  # 结束 FEN-003 后续阶段元组。
    },  # 结束 FEN-003 入口合同。
    "FEN-014": {  # 定义“外部网格是否可信”案例的入口合同。
        "profile_name": "fen-014.json",  # 指向当前仓库中 FEN-014 的静态 profile 文件名。
        "execution_family": "FEniCS/FEniCSx",  # 冻结用户确认的 FEN-014 执行技术族映射。
        "expected_skills": (  # 冻结 profile 当前唯一可调用的通用 Skill。
            "problem-definition-source-audit",  # 当前只能执行来源审计，不能伪装已有网格诊断器。
        ),  # 结束 FEN-014 Skill 标识元组。
        "expected_gaps": (  # 冻结尚未实现的 FEniCS 外部网格诊断能力标识。
            "fenics-imported-mesh-integrity-and-tag-mapping-diagnosis",  # 记录解析、拓扑、方向、尺度和 tags 映射能力缺口。
        ),  # 结束 FEN-014 Skill 缺口元组。
        "missing_paths": (  # 冻结最小真实差分测试前必须补齐的七个事实路径。
            "runtime.fenics_variant_version_and_entrypoint",  # 要求确认 FEniCS 变体、版本、入口和底层求解配置。
            "imported_mesh.current_file",  # 要求未修改的当前外部网格和原始来源。
            "imported_mesh.format_contract",  # 要求格式版本、连接和标签表达合同。
            "reference_regular_mesh.current_evidence",  # 要求同一物理模型的可运行参考网格证据。
            "model_mapping.current_definition",  # 要求材料、载荷、边界和集合映射定义。
            "units.current_definition",  # 要求坐标、材料、载荷和结果的一致单位定义。
            "runtime.fenics_failure_log",  # 要求真实 FEniCS 失败命令、版本、traceback 和运行产物。
        ),  # 结束 FEN-014 缺失事实路径元组。
        "expected_stop_field_count": 8,  # profile 当前列出八个必须赋值的可信性门字段。
        "phase_names": (  # 定义来源证据不足时必须保持未运行的后续阶段。
            "format_parse",  # 未确认格式合同前不得解析并宣称兼容。
            "topology_check",  # 未获得网格文件前不得声称检查拓扑。
            "element_orientation_check",  # 未获得连接顺序前不得声称检查方向。
            "mapping_and_scale_check",  # 未获得单位和映射合同前不得比较尺度或集合。
            "minimal_fenics_reproduction",  # 静态门未通过时不得运行 FEniCS/FEniCSx 复现。
            "qoi_comparison",  # 没有同物理参考结果时不得比较 QoI。
            "engineering_decision",  # 没有完整证据时不得宣布网格可信或不可信。
        ),  # 结束 FEN-014 后续阶段元组。
    },  # 结束 FEN-014 入口合同。
    "CCX-015": {  # 定义“如何安全自动细化”案例的入口合同。
        "profile_name": "ccx-015.json",  # 指向当前仓库中 CCX-015 的静态 profile 文件名。
        "execution_family": "CalculiX",  # 冻结用户确认的 CCX-015 执行技术族映射。
        "expected_skills": (  # 冻结 profile 当前引用的三个通用 Skill。
            "problem-definition-source-audit",  # 要求先冻结当前物理事实与来源。
            "mesh-convergence-and-singularity",  # 要求区分有效 QoI 改善和局部奇异趋势。
            "optimization-readiness",  # 要求检查优化目标、约束、预算和可复现性。
        ),  # 结束 CCX-015 Skill 标识元组。
        "expected_gaps": (  # 冻结当前尚未实现的质量约束细化能力。
            "mesh-quality-constrained-refinement-and-rollback",  # 记录质量门、过渡、回退和策略切换缺口。
        ),  # 结束 CCX-015 Skill 缺口元组。
        "missing_paths": (  # 冻结单候选真实试验前必须补齐的八个事实路径。
            "runtime.calculix_version_command_and_input",  # 要求确认 CalculiX 版本、命令和输入 deck 入口。
            "mesh.refinement_toolchain",  # 要求确认 CalculiX 之外的候选网格生成与转换工具链。
            "mesh.last_accepted_state",  # 要求可回退的最后接受网格和真实结果。
            "qoi.current_definition",  # 要求当前细化要改善的完整 QoI 合同。
            "mesh.quality_contract",  # 要求当前单元族质量量、硬门和来源。
            "mesh.transition_contract",  # 要求局部尺寸、邻接、过渡和回退约束。
            "budget.current_definition",  # 要求网格、求解、失败和墙钟预算。
            "refinement.current_indicator",  # 要求当前结果驱动指标和目标区域来源。
        ),  # 结束 CCX-015 缺失事实路径元组。
        "expected_stop_field_count": 8,  # profile 当前列出八个必须赋值的接受、回退或切换字段。
        "phase_names": (  # 定义来源证据不足时不得运行的后续阶段。
            "candidate_generation",  # 未冻结接受状态和约束前不得生成细化候选。
            "mesh_generation",  # 未建立过渡规则前不得实际生成网格。
            "quality_precheck",  # 未建立质量合同前不得声称候选通过硬门。
            "calculix_execution",  # 前置质量门未通过时不得调用 CalculiX。
            "qoi_extraction",  # 没有可比结果时不得提取或比较 QoI。
            "accept_or_rollback",  # 没有稳定性、质量和预算证据时不得接受候选。
            "engineering_decision",  # 单候选或缺失事实不能证明优化充分。
        ),  # 结束 CCX-015 后续阶段元组。
    },  # 结束 CCX-015 入口合同。
}  # 结束三个案例的 T0 冻结合同。

EXPECTED_QOI_FIELD_COUNT = 14  # 三个 profile 当前都要求冻结十四个 QoI 合同字段。
ZERO_EXECUTION_COUNTS: dict[str, int] = {  # 冻结 T0 必须保持为零的执行能力计数。
    "model_calls": 0,  # T0 不调用 DeepSeek 或其他语言模型。
    "parser_calls": 0,  # T0 不读取尚未提供的 IMP 或外部网格内容。
    "mesh_checker_calls": 0,  # T0 不伪造网格完整性或质量检查。
    "mesh_generation_calls": 0,  # T0 不生成任何工程网格。
    "fenics_calls": 0,  # T0 不导入或运行 FEniCS/FEniCSx 案例。
    "calculix_calls": 0,  # T0 不安装或调用 CalculiX。
    "optimization_calls": 0,  # T0 不运行 PSO 或任何优化器。
}  # 结束零执行计数合同。
SUPPORT_SOURCE_PATHS = (  # 冻结复现本 T0 活动需要纳入 artifact 的支持文件。
    Path("src/engineering_agent/problem_manifest.py"),  # 记录实际使用的 ProblemManifest 实现。
    Path("src/engineering_agent/skill_contract.py"),  # 记录实际使用的 Skill 解析实现。
    Path("schemas/problem_manifest.schema.json"),  # 记录当前 ProblemManifest JSON 合同。
    Path("scripts/run_v5_high_layer_readiness.py"),  # 记录当前就绪性 runner 自身。
    Path("scripts/qualify_fenicsx_runtime.py"),  # 记录固定官方镜像内执行的 FEniCSx 环境资格脚本。
    Path("tests/test_v5_high_layer_readiness.py"),  # 记录当前 fail-closed 单元测试。
    Path("docs/v5_high_layer/CASE_TEST_READINESS.md"),  # 记录 T0 状态与字段解释。
    Path(".github/workflows/v5-high-layer-case-readiness.yml"),  # 记录当前远程执行工作流。
)  # 结束 T0 支持来源路径元组。


def sha256_file(path: Path) -> str:  # 输入一个现有文件路径并返回稳定的十六进制 SHA-256。
    digest = hashlib.sha256()  # 为当前文件创建独立 SHA-256 摘要器。
    with path.open("rb") as handle:  # 以只读二进制方式打开文件，避免换行转换改变哈希。
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):  # 每次读取一 MiB 以限制内存占用。
            digest.update(chunk)  # 将当前字节块加入摘要计算。
    return digest.hexdigest()  # 返回六十四字符的十六进制文件摘要。


def read_json(path: Path) -> dict[str, Any]:  # 读取一个 UTF-8 JSON 对象并拒绝非对象顶层值。
    payload = json.loads(path.read_text(encoding="utf-8"))  # 使用标准库解析当前仓库 JSON 文本。
    if not isinstance(payload, dict):  # profile 和 Skill 顶层必须是 JSON 对象。
        raise ValueError(path.as_posix() + " must contain a JSON object")  # 用稳定路径报告合同错误。
    return payload  # 返回已确认是对象的 JSON 数据。


def write_json(path: Path, payload: Any) -> None:  # 将机器证据以稳定 UTF-8 格式写入指定路径。
    path.parent.mkdir(parents=True, exist_ok=True)  # 在 artifact 内创建必要父目录。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 使用两空格缩进并保留中文。


def source_record(path: Path) -> dict[str, Any]:  # 为当前仓库文件生成路径、大小和 SHA-256 来源记录。
    absolute = path if path.is_absolute() else REPO_ROOT / path  # 同时支持仓库相对路径和测试绝对路径。
    if not absolute.is_file():  # 来源缺失时立即阻止 receipt 假装完整。
        raise FileNotFoundError(absolute.as_posix())  # 报告实际缺失文件位置。
    try:  # 尝试把来源表示为仓库相对路径以便 GitHub 追溯。
        display_path = absolute.resolve().relative_to(REPO_ROOT.resolve()).as_posix()  # 计算稳定仓库相对路径。
    except ValueError:  # 测试临时文件不在仓库时保留绝对路径。
        display_path = absolute.resolve().as_posix()  # 记录测试来源的完整绝对路径。
    return {  # 返回闭合的来源清单对象。
        "path": display_path,  # 记录实际读取文件的位置。
        "size_bytes": absolute.stat().st_size,  # 记录文件字节数用于完整性核对。
        "sha256": sha256_file(absolute),  # 记录当前内容的 SHA-256。
        "source_kind": "current_authoritative_commit",  # 说明生产运行只接受当前检出提交中的文件。
    }  # 结束来源清单对象。


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:  # 追加一个命名布尔检查和可审计细节。
    checks.append({"name": name, "passed": bool(passed), "details": details})  # 标准化检查结构以便 Actions 和前端读取。


def validate_profile(profile: dict[str, Any], case_id: str, contract: dict[str, Any], skill_library: SkillLibrary) -> list[dict[str, Any]]:  # 验证 profile 身份、缺失事实和 Skill 引用未被静默改变。
    checks: list[dict[str, Any]] = []  # 创建当前案例按顺序记录的检查列表。
    add_check(checks, "profile_case_id", profile.get("case_id") == case_id, {"expected": case_id, "actual": profile.get("case_id")})  # 要求文件内容与预期案例一致。
    add_check(checks, "profile_version", profile.get("profile_version") == "1.1", {"expected": "1.1", "actual": profile.get("profile_version")})  # 要求加入执行技术族后的 profile 使用一点一合同版本。
    expected_execution_family = str(contract["execution_family"])  # 读取用户确认并由 runner 冻结的执行技术族。
    actual_execution_family = str(profile.get("execution_family", ""))  # 读取当前 profile 的机器可读执行技术族。
    add_check(checks, "execution_family_exact", actual_execution_family == expected_execution_family, {"expected": expected_execution_family, "actual": actual_execution_family})  # 阻止 FEN 与 CCX 再次接错执行后端。
    add_check(checks, "profile_remains_unexecuted", profile.get("status") == "draft_not_executed", {"actual": profile.get("status")})  # T0 不允许改写 profile 为已执行工程案例。
    actual_skills = tuple(profile.get("applicable_skill_ids", []))  # 读取 profile 当前声明的通用 Skill 顺序。
    expected_skills = tuple(contract["expected_skills"])  # 读取本 runner 冻结的预期 Skill 顺序。
    add_check(checks, "applicable_skills_exact", actual_skills == expected_skills, {"expected": list(expected_skills), "actual": list(actual_skills)})  # 防止 Skill 被静默增删或替换。
    skill_issues = {skill_id: skill_library.skills[skill_id].issues() if skill_id in skill_library.skills else ["skill is missing"] for skill_id in expected_skills}  # 收集每个预期 Skill 的真实加载问题。
    add_check(checks, "applicable_skills_load", all(skill_id in skill_library.skills and not skill_issues[skill_id] for skill_id in expected_skills), skill_issues)  # 要求引用的当前 Skill 存在且合同完整。
    actual_gaps = tuple(profile.get("skill_coverage_gaps", []))  # 读取当前 profile 声明的通用能力缺口。
    expected_gaps = tuple(contract["expected_gaps"])  # 读取本 T0 冻结的能力缺口。
    add_check(checks, "skill_gaps_exact", actual_gaps == expected_gaps, {"expected": list(expected_gaps), "actual": list(actual_gaps)})  # 防止缺口被误当成已实现 Skill。
    missing_rows = profile.get("missing_facts", [])  # 读取当前 profile 的缺失事实数组。
    actual_missing_paths = tuple(row.get("path") for row in missing_rows if isinstance(row, dict)) if isinstance(missing_rows, list) else ()  # 提取结构有效条目的事实路径。
    expected_missing_paths = tuple(contract["missing_paths"])  # 读取本案例冻结的入口缺失事实路径。
    add_check(checks, "missing_fact_paths_exact", actual_missing_paths == expected_missing_paths, {"expected": list(expected_missing_paths), "actual": list(actual_missing_paths)})  # 要求所有必要缺口原样保留。
    missing_structures_valid = isinstance(missing_rows, list) and bool(missing_rows) and all(isinstance(row, dict) and set(row) == {"path", "reason", "question", "acceptable_sources"} and bool(str(row["reason"]).strip()) and bool(str(row["question"]).strip()) and isinstance(row["acceptable_sources"], list) and bool(row["acceptable_sources"]) for row in missing_rows)  # 检查每条缺参都有原因、问题和来源类型。
    add_check(checks, "missing_fact_structures", missing_structures_valid, {"row_count": len(missing_rows) if isinstance(missing_rows, list) else 0})  # 记录缺参结构完整性。
    qoi_fields = profile.get("fixed_qoi_contract_fields", [])  # 读取尚待当前任务赋值的 QoI 字段名。
    add_check(checks, "qoi_field_contract_present", isinstance(qoi_fields, list) and len(qoi_fields) == EXPECTED_QOI_FIELD_COUNT and len(set(qoi_fields)) == EXPECTED_QOI_FIELD_COUNT, {"expected_count": EXPECTED_QOI_FIELD_COUNT, "actual_count": len(qoi_fields) if isinstance(qoi_fields, list) else 0})  # 要求十四个 QoI 字段完整且不重复。
    stop_fields = profile.get("stop_condition_fields", [])  # 读取尚待当前任务赋值的停止条件字段名。
    expected_stop_count = int(contract["expected_stop_field_count"])  # 读取当前案例的预期停止字段数量。
    add_check(checks, "stop_field_contract_present", isinstance(stop_fields, list) and len(stop_fields) == expected_stop_count and len(set(stop_fields)) == expected_stop_count, {"expected_count": expected_stop_count, "actual_count": len(stop_fields) if isinstance(stop_fields, list) else 0})  # 要求停止合同字段完整且不重复。
    evidence_rows = profile.get("required_evidence", [])  # 读取 profile 要求的当前证据类型。
    evidence_structures_valid = isinstance(evidence_rows, list) and bool(evidence_rows) and all(isinstance(row, dict) and set(row) == {"name", "purpose", "acceptable_sources"} and bool(str(row["name"]).strip()) and bool(str(row["purpose"]).strip()) and isinstance(row["acceptable_sources"], list) and bool(row["acceptable_sources"]) for row in evidence_rows)  # 检查每类证据的名称、用途和来源类型完整。
    add_check(checks, "required_evidence_contract_present", evidence_structures_valid, {"row_count": len(evidence_rows) if isinstance(evidence_rows, list) else 0})  # 记录证据合同完整性。
    return checks  # 返回当前案例全部静态来源检查。


def build_manifest(profile: dict[str, Any], profile_record: dict[str, Any], skill_records: list[dict[str, Any]]) -> ProblemManifest:  # 从 profile 缺失事实生成不含物理事实的当前 ProblemManifest。
    case_id = str(profile["case_id"])  # 读取已验证的案例标识作为稳定任务后缀。
    manifest = ProblemManifest(  # 创建只表示 T0 来源审计状态的 ProblemManifest。
        task_id="v5-high-layer-readiness-" + case_id.lower(),  # 生成稳定且不会冒充真实工程任务的测试标识。
        user_goal=str(profile["objective"]),  # 使用当前 profile 的目标描述，不增加物理参数。
        input_files=[profile_record, *skill_records],  # 仅记录实际读取的 profile 与 Skill 来源元数据。
        observations=[  # 明确记录本 manifest 的执行和解释边界。
            "T0 只执行当前仓库来源审计与缺失事实保留。",  # 说明本阶段的唯一正向能力。
            "没有提供当前案例物理输入，facts 必须保持为空。",  # 防止旧 benchmark 或假设进入事实集合。
            "没有调用模型、解析器、网格检查器、网格生成器、求解器或优化器。",  # 明确全部执行调用为零。
            "Actions 成功只表示系统正确阻断，不能解释为工程案例通过。",  # 防止把软件门通过误读为网格或结果可信。
        ],  # 结束 T0 观察列表。
    )  # 完成 ProblemManifest 初始化。
    for row in profile["missing_facts"]:  # 按 profile 原顺序登记每条必要缺失事实。
        manifest.require_fact(  # 调用当前 V5 合同生成稳定问题清单。
            str(row["path"]),  # 保留 profile 中的缺失字段路径。
            reason=str(row["reason"]),  # 保留缺失原因，禁止改写成默认值说明。
            question=str(row["question"]),  # 保留向用户提出的具体问题。
            acceptable_sources=[str(item) for item in row["acceptable_sources"]],  # 保留允许补齐事实的当前来源类型。
        )  # 完成当前缺失事实登记。
    return manifest  # 返回 facts 与算法配置均为空的 T0 manifest。


def phase_gate_records(contract: dict[str, Any]) -> list[dict[str, str]]:  # 为当前案例生成来源审计后全部未运行的阶段门。
    return [{"stage": "source_audit", "status": "blocked_missing_current_evidence"}, *[{"stage": str(name), "status": "not_run"} for name in contract["phase_names"]]]  # 第一个门明确阻断，后续阶段全部保持未运行。


def process_case(case_id: str, contract: dict[str, Any], profiles_dir: Path, skills_dir: Path, output_dir: Path, skill_library: SkillLibrary) -> dict[str, Any]:  # 处理一个案例并写出 manifest、问题、来源和就绪性 receipt。
    profile_path = profiles_dir / str(contract["profile_name"])  # 根据显式 profile 目录和冻结文件名定位当前案例。
    profile = read_json(profile_path)  # 读取当前权威提交中的案例 profile。
    profile_record = source_record(profile_path)  # 为实际读取的 profile 生成哈希来源记录。
    skill_paths = [skills_dir / (str(skill_id) + ".json") for skill_id in contract["expected_skills"]]  # 由冻结 Skill ID 形成当前仓库路径。
    skill_records = [source_record(path) for path in skill_paths]  # 为每个实际引用 Skill 生成哈希记录。
    checks = validate_profile(profile, case_id, contract, skill_library)  # 验证身份、缺口、字段和 Skill 未被静默改变。
    manifest = build_manifest(profile, profile_record, skill_records)  # 构造当前无物理事实的 fail-closed ProblemManifest。
    manifest_dict = manifest.to_dict()  # 转换为可写入 JSON 的稳定对象。
    provenance = manifest_dict["provenance_report"]  # 读取当前合同计算的来源摘要。
    add_check(checks, "no_current_physical_facts", provenance["fact_count"] == 0, provenance)  # 当前未上传案例输入时 facts 必须严格为空。
    add_check(checks, "no_algorithm_defaults", provenance["algorithm_setting_count"] == 0, provenance)  # T0 不得借算法配置偷带工程阈值或预算。
    add_check(checks, "all_missing_facts_preserved", provenance["missing_fact_count"] == len(contract["missing_paths"]), {"expected": len(contract["missing_paths"]), "actual": provenance["missing_fact_count"]})  # 要求缺参数量与冻结入口合同一致。
    add_check(checks, "no_legacy_or_pending_values", not provenance["has_legacy_fixture_values"] and not provenance["has_pending_assumptions"], {"has_legacy_fixture_values": provenance["has_legacy_fixture_values"], "has_pending_assumptions": provenance["has_pending_assumptions"]})  # 禁止旧夹具和待确认假设混入当前事实。
    evidence_status = [{"name": str(row["name"]), "purpose": str(row["purpose"]), "satisfied": False, "status": "missing_current_evidence"} for row in profile["required_evidence"]]  # 将每类当前证据明确标记为尚未提供。
    qoi_field_status = [{"field": str(field), "supplied": False} for field in profile["fixed_qoi_contract_fields"]]  # 将十四个 QoI 字段明确标记为尚未冻结。
    stop_field_status = [{"field": str(field), "supplied": False} for field in profile["stop_condition_fields"]]  # 将当前案例停止字段明确标记为尚未赋值。
    execution_counts = dict(ZERO_EXECUTION_COUNTS)  # 为 receipt 复制独立的零执行计数对象。
    add_check(checks, "all_required_evidence_missing", bool(evidence_status) and all(not row["satisfied"] for row in evidence_status), {"required_evidence_count": len(evidence_status)})  # T0 只有在没有伪称证据时才通过。
    add_check(checks, "all_qoi_fields_unfrozen", len(qoi_field_status) == EXPECTED_QOI_FIELD_COUNT and all(not row["supplied"] for row in qoi_field_status), {"field_count": len(qoi_field_status)})  # 未提供当前 QoI 时必须全部保持未冻结。
    add_check(checks, "all_stop_fields_unfrozen", len(stop_field_status) == int(contract["expected_stop_field_count"]) and all(not row["supplied"] for row in stop_field_status), {"field_count": len(stop_field_status)})  # 未提供当前判断规则时必须全部保持未赋值。
    add_check(checks, "execution_counts_zero", all(value == 0 for value in execution_counts.values()), execution_counts)  # 防止来源审计阶段发生任何模型或工程执行。
    test_passed = bool(checks) and all(bool(row["passed"]) for row in checks)  # 只有所有静态和 fail-closed 检查通过时 T0 才成功。
    case_root = output_dir / case_id.lower()  # 为当前案例建立独立 artifact 子目录。
    source_manifest = {"case_id": case_id, "sources": [profile_record, *skill_records]}  # 汇总当前案例实际读取的 profile 与 Skill 来源。
    receipt = {  # 构造当前案例的机器可读 T0 receipt。
        "schema_version": "v5-high-layer-readiness-receipt/1.0",  # 冻结本轮就绪性证据合同版本。
        "case_id": case_id,  # 记录研究计划中的稳定案例标识。
        "execution_family": str(contract["execution_family"]),  # 记录用户确认的 FEniCSx/DOLFINx 或 CalculiX 路线。
        "test_level": "T0_readiness_fail_closed",  # 说明这是前置条件测试而不是数值阶段。
        "test_outcome": "pass" if test_passed else "fail",  # 区分测试软件是否正确执行。
        "case_status": "blocked_missing_current_evidence",  # 即使 T0 通过，工程案例仍明确阻断。
        "decision": "unresolved",  # 当前证据不允许输出停止、可信或接受候选结论。
        "next_action": "ask_user",  # 后续只能请求 profile 列出的当前事实。
        "stop_stage": "source_audit",  # 在解析、网格、求解和优化之前停止。
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),  # 记录 Actions 提供的仓库标识。
        "ref": os.environ.get("GITHUB_REF", ""),  # 记录 Actions 提供的完整分支引用。
        "head_sha": os.environ.get("GITHUB_SHA", ""),  # 记录本次验证的不可变提交 SHA。
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),  # 记录本次 GitHub Actions 运行编号。
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),  # 记录本次运行的重跑编号。
        "checks": checks,  # 保存每个入口合同和零执行断言。
        "required_evidence_status": evidence_status,  # 保存当前全部缺失的证据类型。
        "qoi_field_status": qoi_field_status,  # 保存尚未冻结的 QoI 字段覆盖。
        "stop_condition_status": stop_field_status,  # 保存尚未赋值的停止、回退或可信性门。
        "missing_facts": manifest_dict["missing_facts"],  # 保存 ProblemManifest 原样保留的缺失事实。
        "questions_for_user": manifest.questions_for_user(),  # 保存由当前 V5 合同生成的具体问题。
        "provenance_report": provenance,  # 保存零事实、零旧夹具和零待确认假设证明。
        "phase_gates": phase_gate_records(contract),  # 保存 source audit 阻断和后续阶段未运行状态。
        "execution_counts": execution_counts,  # 保存全部模型、解析、网格、求解和优化调用为零。
        "allowed_use": [  # 声明本 receipt 可以支持的结论。
            "证明当前缺失事实被完整保留。",  # 允许用来审计没有默认值填充。
            "证明系统在当前证据不足时于 source_audit 阶段停止。",  # 允许用来审计 fail-closed 行为。
            "形成进入真实数值阶段前的用户问题清单。",  # 允许用来准备下一轮当前输入。
        ],  # 结束允许用途列表。
        "disallowed_use": [  # 声明本 receipt 不能支持的结论。
            "不能证明网格收敛、网格可信或自动细化安全。",  # T0 没有任何网格和结果证据。
            "不能证明 CalculiX、解析器、网格检查器或优化器已运行。",  # 所有执行计数明确为零。
            "不能把 Actions 成功解释为工程案例通过。",  # 软件门通过与工程结论严格分离。
        ],  # 结束禁止用途列表。
        "interpretation_limits": list(profile["interpretation_limits"]),  # 沿用当前 profile 的工程解释边界。
        "source_manifest": source_manifest,  # 内嵌当前案例来源哈希便于单文件审计。
    }  # 结束当前案例 receipt。
    write_json(case_root / "problem_manifest.json", manifest_dict)  # 写出无物理事实、完整缺参的 ProblemManifest。
    write_json(case_root / "required_input_questions.json", {"case_id": case_id, "case_status": receipt["case_status"], "questions": receipt["questions_for_user"]})  # 写出下一阶段用户输入问题。
    write_json(case_root / "source_manifest.json", source_manifest)  # 写出 profile 和 Skill 的来源清单。
    write_json(case_root / "readiness_receipt.json", receipt)  # 写出当前案例的主审计 receipt。
    return receipt  # 返回 receipt 供整体活动汇总和退出判断。


def copy_source_snapshot(paths: list[Path], output_dir: Path) -> list[dict[str, Any]]:  # 复制实际使用的仓库来源并返回统一哈希清单。
    snapshot_root = output_dir / "source"  # 把只读来源快照放在 artifact 的 source 子目录。
    records: list[dict[str, Any]] = []  # 创建去重前的来源记录列表。
    seen: set[str] = set()  # 创建仓库相对路径去重集合。
    for relative in paths:  # 按调用方提供的稳定顺序处理来源。
        normalized = relative.as_posix()  # 将仓库路径转换为跨平台斜杠形式。
        if normalized in seen:  # 同一 Skill 被多个案例引用时只复制一次。
            continue  # 跳过已复制来源以保持 artifact 紧凑。
        seen.add(normalized)  # 将当前来源路径标记为已处理。
        source = REPO_ROOT / relative  # 解析当前仓库来源文件绝对路径。
        record = source_record(source)  # 在复制前计算实际内容哈希。
        destination = snapshot_root / relative  # 保留仓库相对目录结构作为快照路径。
        destination.parent.mkdir(parents=True, exist_ok=True)  # 创建 artifact 中的目标父目录。
        shutil.copy2(source, destination)  # 复制当前提交文件内容和基础元数据。
        records.append(record)  # 将来源记录加入整体清单。
    return records  # 返回已复制来源的稳定哈希清单。


def run_campaign(profiles_dir: Path, skills_dir: Path, output_dir: Path) -> dict[str, Any]:  # 执行三个 T0 案例并写出整体活动 receipt。
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建成功和失败都可使用的 artifact 根目录。
    skill_library = SkillLibrary.load_json_directory(skills_dir)  # 用当前 V5 合同加载真实 Skill 目录。
    receipts: list[dict[str, Any]] = []  # 创建三个案例 receipt 收集列表。
    for case_id, contract in CASE_CONTRACTS.items():  # 按研究计划递进顺序处理 FEN-003、FEN-014 和 CCX-015。
        try:  # 捕获单个案例的来源或合同异常并继续为其他案例留证。
            receipt = process_case(case_id, contract, profiles_dir, skills_dir, output_dir, skill_library)  # 执行当前案例 T0 检查。
        except Exception as exc:  # 将任何 runner 异常转为显式失败 receipt。
            case_root = output_dir / case_id.lower()  # 定位当前案例失败证据目录。
            receipt = {  # 构造不会冒充正确阻断的失败 receipt。
                "schema_version": "v5-high-layer-readiness-receipt/1.0",  # 保持与正常 receipt 相同的合同版本。
                "case_id": case_id,  # 记录发生 runner 异常的案例标识。
                "execution_family": str(contract["execution_family"]),  # 即使 runner 失败也保留正确执行技术族映射。
                "test_level": "T0_readiness_fail_closed",  # 说明异常发生在入口测试阶段。
                "test_outcome": "fail",  # runner 异常必须使软件测试失败。
                "case_status": "runner_error",  # 区分执行器错误和预期的证据阻断。
                "decision": "unresolved",  # 异常不能产生工程决策。
                "stop_stage": "source_audit",  # 保持在任何工程执行之前停止。
                "error": type(exc).__name__ + ": " + str(exc),  # 保存异常类型和消息供 Actions 调试。
                "execution_counts": dict(ZERO_EXECUTION_COUNTS),  # 即使失败也明确没有工程执行调用。
            }  # 结束单案例失败 receipt。
            write_json(case_root / "readiness_receipt.json", receipt)  # 始终写出可上传的失败证据。
        receipts.append(receipt)  # 将成功阻断或 runner 失败 receipt 加入整体活动。
    snapshot_paths = list(SUPPORT_SOURCE_PATHS)  # 从 runner、测试、说明、工作流和现有合同开始构造快照列表。
    snapshot_paths.extend(Path("examples/v5_high_layer") / str(contract["profile_name"]) for contract in CASE_CONTRACTS.values())  # 加入三个当前 profile。
    snapshot_paths.extend(Path("skills/engineering") / (str(skill_id) + ".json") for contract in CASE_CONTRACTS.values() for skill_id in contract["expected_skills"])  # 加入全部实际引用 Skill。
    source_snapshot = copy_source_snapshot(snapshot_paths, output_dir)  # 复制去重来源并生成统一哈希清单。
    campaign_passed = len(receipts) == len(CASE_CONTRACTS) and all(receipt.get("test_outcome") == "pass" and receipt.get("case_status") == "blocked_missing_current_evidence" for receipt in receipts)  # 三案都正确阻断时整体 T0 才通过。
    campaign = {  # 构造三案例整体活动 receipt。
        "schema_version": "v5-high-layer-readiness-campaign/1.0",  # 冻结整体活动机器合同版本。
        "status": "success" if campaign_passed else "failure",  # 区分 T0 软件测试整体成功或失败。
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),  # 记录 GitHub Actions 仓库标识。
        "ref": os.environ.get("GITHUB_REF", ""),  # 记录权威分支完整引用。
        "head_sha": os.environ.get("GITHUB_SHA", ""),  # 记录当前不可变提交 SHA。
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),  # 记录 GitHub Actions 运行编号。
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),  # 记录运行尝试编号。
        "case_count": len(receipts),  # 记录本活动实际生成 receipt 的案例数量。
        "passed_readiness_count": sum(receipt.get("test_outcome") == "pass" for receipt in receipts),  # 记录正确执行 T0 的案例数。
        "blocked_case_count": sum(receipt.get("case_status") == "blocked_missing_current_evidence" for receipt in receipts),  # 记录因当前证据不足而正确阻断的案例数。
        "case_summaries": [{"case_id": receipt.get("case_id"), "execution_family": receipt.get("execution_family"), "test_outcome": receipt.get("test_outcome"), "case_status": receipt.get("case_status"), "next_action": receipt.get("next_action", "inspect_runner_error")} for receipt in receipts],  # 提供三案精简状态索引。
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),  # 冻结整体活动的模型、网格、求解和优化调用为零。
        "engineering_results_generated": False,  # 明确本活动没有生成任何有限元工程结果。
        "success_meaning": "三个案例均完整保留当前缺失事实，并在 source_audit 阶段正确停止。",  # 定义 Actions success 的唯一含义。
        "source_snapshot": source_snapshot,  # 保存 artifact 中全部复现来源的路径、大小和哈希。
    }  # 结束整体活动 receipt。
    write_json(output_dir / "campaign_receipt.json", campaign)  # 写出三案例主索引和总体边界。
    return campaign  # 返回整体 receipt 供单元测试和 CLI 退出判断。


def parse_args() -> argparse.Namespace:  # 定义所有路径参数并要求工作流显式提供。
    parser = argparse.ArgumentParser(description="Run fail-closed readiness tests for the three V5 high-layer cases.")  # 创建只描述 T0 行为的命令行解析器。
    parser.add_argument("--profiles-dir", required=True, help="Directory containing the three current high-layer profiles.")  # 要求显式传入当前 profile 目录。
    parser.add_argument("--skills-dir", required=True, help="Directory containing the current engineering Skill JSON files.")  # 要求显式传入当前 Skill 目录。
    parser.add_argument("--output-dir", required=True, help="Artifact directory for manifests, questions, receipts, and source snapshots.")  # 要求显式传入证据输出目录。
    return parser.parse_args()  # 解析调用方提供的全部路径参数。


def main() -> int:  # 执行 T0 活动、打印精简状态并返回真实成功或失败退出码。
    args = parse_args()  # 读取工作流显式提供的三个目录参数。
    campaign = run_campaign(Path(args.profiles_dir), Path(args.skills_dir), Path(args.output_dir))  # 运行三案来源审计和 fail-closed 测试。
    print(json.dumps({"status": campaign["status"], "case_count": campaign["case_count"], "blocked_case_count": campaign["blocked_case_count"], "output_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False))  # 在 Actions 日志输出紧凑且不冒充工程结论的状态。
    return 0 if campaign["status"] == "success" else 1  # 三案正确阻断时返回零，否则使 workflow 明确失败。


if __name__ == "__main__":  # 仅在脚本由工作流直接执行时运行入口。
    raise SystemExit(main())  # 将真实活动状态传递给 GitHub Actions。
