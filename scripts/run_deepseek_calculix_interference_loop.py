#!/usr/bin/env python3
# 允许使用现代类型注解，同时保持 Python 3.11 运行兼容性。
from __future__ import annotations

# 导入命令行解析模块，用于接收模型、求解器、轮数和输出目录配置。
import argparse
# 导入 HTML 转义模块，用于把模型文本安全写入 ReportLab 段落。
import html
# 导入哈希模块，用于冻结输入、deck、trace 和论文来源。
import hashlib
# 导入 JSON 模块，用于读取 Skill/source audit 并保存完整决策轨迹。
import json
# 导入操作系统模块，用于读取 ds Environment 中的模型凭据和配置。
import os
# 导入正则表达式模块，用于从 CalculiX 日志提取收敛和位移指标。
import re
# 导入可执行文件定位模块，用于寻找长期化环境中的 ccx 命令。
import shutil
# 导入子进程模块，用于执行真实 CalculiX 求解。
import subprocess
# 导入系统模块，用于返回明确的进程退出状态。
import sys
# 导入计时模块，用于记录每次微型求解的墙钟时间。
import time
# 导入 UTC 时间类型，用于生成与本地时区无关的审计时间戳。
from datetime import datetime
# 导入 UTC 时区常量，用于规范化审计时间。
from datetime import timezone
# 导入路径类型，用于跨 Windows 和 Linux 管理仓库及 artifact 路径。
from pathlib import Path
# 导入通用类型，用于描述动态 JSON、SDK 响应和 ReportLab 对象。
from typing import Any

# 定位当前脚本所属仓库根目录，避免依赖调用者的当前目录。
ROOT = Path(__file__).resolve().parents[1]
# 固定原帖 source audit 的仓库相对位置。
SOURCE_AUDIT_PATH = ROOT / "experiments" / "calculix_interference_2747" / "original_deck_audit.json"
# 固定应用级接触诊断 Skill 的仓库相对位置。
SKILL_PATH = ROOT / "skills" / "engineering" / "nonlinear-contact-diagnosis.json"
# 固定最多三次 DeepSeek HTTP 请求，以控制费用并避免开放循环。
MAX_HTTP_REQUESTS = 3
# 固定 DeepSeek 单次公开响应和推理的最大 token 预算。
MAX_COMPLETION_TOKENS = 2400
# 固定单次 DeepSeek 请求的最长等待时间为三分钟。
API_TIMEOUT_SECONDS = 180.0
# 固定微型 CalculiX 求解的最长等待时间为三十秒。
SOLVER_TIMEOUT_SECONDS = 30.0
# 固定默认模型名称，并允许 ds Environment 通过命令行或环境变量覆盖。
DEFAULT_MODEL = "deepseek-v4-pro"
# 固定代表模型的主接触面高度，所有初始过闭合都相对此值定义。
MASTER_SURFACE_Z = 1.0
# 固定搜索案例的边中间节点高度，使其产生 0.05 长度单位初始过闭合。
SEARCH_MIDSIDE_Z = 0.95
# 固定未参与初始选择的留出高度，使其产生更大的 0.25 长度单位过闭合。
HOLDOUT_MIDSIDE_Z = 0.75
# 固定代表模型的默认初始增量，与 CalculiX 官方 contact4 测试保持一致。
DEFAULT_INITIAL_INCREMENT = 0.5
# 固定工具缺省建议的显式 penalty 数值，用于允许模型选择该实验但不预告结果。
DEFAULT_EXPLICIT_PENALTY = 100000.0
# 固定正式代表模型的 node-to-surface 接触形式，以获得可用的健康对照。
DEFAULT_CONTACT_TYPE = "NODE TO SURFACE"
# 固定 trace 的机器可读格式版本。
TRACE_SCHEMA_VERSION = "deepseek-calculix-contact-loop/1.0"
# 固定问题清单的格式版本，并与仓库 problem manifest schema 对齐。
MANIFEST_VERSION = "engineering-problem-manifest/1.0"
# 固定论文来源收据的格式版本。
PROVENANCE_SCHEMA_VERSION = "deepseek-calculix-paper-provenance/1.0"


# 定义稳定 JSON 序列化函数，使缓存前缀、哈希和审计输出可复现。
def _canonical_json(value: Any) -> str:
    # 使用固定键序、紧凑分隔符和 UTF-8 中文输出。
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# 定义普通 JSON 写入函数，统一 artifact 的编码和可读缩进。
def _write_json(path: Path, value: Any) -> None:
    # 创建目标父目录，确保中间 artifact 可以随时冻结。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写入 UTF-8、中文可读且以换行结尾的 JSON。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 定义必须返回 JSON 对象的读取函数。
def _read_json(path: Path) -> dict[str, Any]:
    # 解析指定 UTF-8 JSON 文件。
    value = json.loads(path.read_text(encoding="utf-8"))
    # 顶层不是对象时拒绝继续，避免把无效 catalog 当作工程事实。
    if not isinstance(value, dict):
        # 抛出带精确路径的结构错误。
        raise ValueError(f"expected a JSON object in {path}")
    # 返回通过顶层类型检查的对象。
    return value


# 定义文件 SHA-256 计算函数，用于来源收据和求解工件清单。
def _sha256_file(path: Path) -> str:
    # 创建独立 SHA-256 累加器。
    digest = hashlib.sha256()
    # 以二进制只读方式打开文件。
    with path.open("rb") as stream:
        # 逐块读取文件，避免对较大日志产生额外内存峰值。
        while True:
            # 每次读取一 MiB 数据，兼顾吞吐和内存占用。
            block = stream.read(1024 * 1024)
            # 没有新字节时结束循环。
            if not block:
                # 跳出文件读取循环。
                break
            # 把当前数据块加入摘要。
            digest.update(block)
    # 返回大写十六进制摘要，和 source audit 中的格式一致。
    return digest.hexdigest().upper()


# 定义带来源的事实构造函数，与 problem manifest schema 保持一致。
def _fact(value: Any, provenance: str, source_ref: str | None, derivation: str | None = None, notes: str | None = None) -> dict[str, Any]:
    # 返回包含值、来源、确认状态和可选推导说明的事实对象。
    return {"value": value, "provenance": provenance, "source_ref": source_ref, "derivation": derivation, "user_confirmed": provenance in {"provided_by_user", "parsed_from_user_model"}, "notes": notes}


# 定义问题清单构造函数，把原帖事实和实验配置严格分开。
def _build_problem_manifest(source_audit: dict[str, Any], solver_version: dict[str, Any], initial_evidence: dict[str, Any], model: str) -> dict[str, Any]:
    # 读取 source audit 中两个原始附件记录。
    source_entries = source_audit.get("entries", [])
    # 构造符合仓库 schema 的当前问题清单。
    return {
        # 写入问题清单格式版本。
        "manifest_version": MANIFEST_VERSION,
        # 写入稳定任务标识。
        "task_id": "calculix-forum-2747-contact-diagnosis",
        # 写入用户要求解决的工程目标。
        "user_goal": "根据原始 CalculiX 论坛问题、附件事实和真实受控求解，诊断网格细化后过盈接触不收敛的机制并给出有适用边界的答复。",
        # 列出本次未搬入仓库但已冻结哈希的两个原始输入。
        "input_files": source_entries,
        # 记录来自附件解析、论坛和求解环境的事实。
        "facts": {
            # 记录原模型的 C3D20R 单元族。
            "original_element_family": _fact(source_audit.get("shared_deck_facts", {}).get("element_family"), "parsed_from_user_model", "original_deck_audit.json#/shared_deck_facts/element_family"),
            # 记录原模型的 surface-to-surface 接触形式。
            "original_contact_formulation": _fact(source_audit.get("shared_deck_facts", {}).get("contact_formulation"), "parsed_from_user_model", "original_deck_audit.json#/shared_deck_facts/contact_formulation"),
            # 记录原模型显式声明的线性 penalty 数值。
            "original_linear_penalty": _fact(source_audit.get("shared_deck_facts", {}).get("linear_penalty_first_value"), "parsed_from_user_model", "original_deck_audit.json#/shared_deck_facts/linear_penalty_first_value"),
            # 记录两个原附件存在接触激活步骤混杂。
            "source_pair_is_confounded": _fact(True, "derived_by_agent", "original_deck_audit.json#/audit_limitations/0", "两个附件同时改变网格、接触面数量和 MODEL CHANGE 路径。"),
            # 记录代表模型真实使用的求解器版本。
            "representative_solver_version": _fact(solver_version, "derived_by_agent", "solver_version_stdout"),
            # 只在 manifest 中引用初始对照 ID，详细数值由同一请求的 initial_solver_evidence 提供以避免重复 token。
            "representative_initial_evidence": _fact({"control_solve_id": initial_evidence.get("control", {}).get("solve_id"), "target_solve_id": initial_evidence.get("target", {}).get("solve_id")}, "derived_by_agent", "agent_trace.json#/initial_solver_evidence"),
        },
        # 列出当前资源下仍缺失且不能静默补造的原模型事实。
        "missing_facts": [
            # 记录没有执行原始大型模型的资源限制。
            {"path": "original_large_model.independent_run", "reason": "原模型约一千一百六十万自由度，当前标准 runner 未证明具有足够内存和 Pardiso 后端。", "question": "候选修复能否在原始圆柱孔模型和批准后端上保持位移、反力、接触传力与穿透合理？", "acceptable_sources": ["大内存 Pardiso CalculiX 运行", "作者原环境复验", "批准的第二求解器复核"]},
            # 记录论坛没有作者确认的最终答案。
            {"path": "forum.confirmed_solution", "reason": "冻结时论坛没有作者确认的后续修复。", "question": "作者是否能在原模型上确认候选修改？", "acceptable_sources": ["论坛作者复验", "原始 deck 修订与求解日志"]},
        ],
        # 把调用预算和代表模型参数明确标成算法配置。
        "algorithm_configuration": {
            # 记录实际请求的 DeepSeek 模型。
            "deepseek_model": _fact(model, "algorithm_configuration", "cli_or_ds_environment", notes="模型名称影响推理行为和费用。"),
            # 记录最多三次模型调用的费用边界。
            "max_http_requests": _fact(MAX_HTTP_REQUESTS, "algorithm_configuration", "runner_constant", notes="三次是硬费用上限，不是工程完成门禁。"),
            # 记录代表模型的搜索过闭合幅度。
            "representative_search_midside_z": _fact(SEARCH_MIDSIDE_Z, "algorithm_configuration", "runner_constant", derivation="相对主面 z=1.0 形成 0.05 长度单位初始过闭合。"),
            # 记录代表模型的留出过闭合幅度。
            "representative_holdout_midside_z": _fact(HOLDOUT_MIDSIDE_Z, "algorithm_configuration", "runner_constant", derivation="相对主面 z=1.0 形成 0.25 长度单位初始过闭合。"),
            # 记录代表模型为建立健康对照使用的接触形式替代。
            "representative_contact_formulation": _fact(DEFAULT_CONTACT_TYPE, "algorithm_configuration", "runner_constant", notes="这是机制实验设置，不是原帖的预设修复。"),
        },
        # 记录清单生成时已有的客观观察。
        "observations": [
            # 记录原始 deck 对照不是单因素。
            "原始粗/细附件同时改变网格规模、PIN2 接触面数量和接触激活历史。",
            # 记录代表模型只改变边中间节点即可形成成功/失败对照。
            "代表模型中仅改变四个接触边中间节点高度，即可形成健康对照与 cutback 失败目标。",
            # 记录该代表模型不能替代原始圆柱孔大型模型验证。
            "代表模型提供机制判别证据，但不构成原始大型模型已解决的证明。",
        ],
    }


# 定义代表模型节点生成函数，其中 mid_z 只影响上块接触面四个边中间节点。
def _representative_nodes(mid_z: float) -> list[tuple[int, float, float, float]]:
    # 返回 CalculiX 官方 contact4 几何，并把四个目标中间节点的 z 参数化。
    return [
        # 下块角点 1 位于 x=1、y=1、z=0。
        (1, 1.0, 1.0, 0.0),
        # 下块角点 2 位于 x=1、y=0、z=0。
        (2, 1.0, 0.0, 0.0),
        # 下块角点 3 位于 x=1、y=0、z=1。
        (3, 1.0, 0.0, 1.0),
        # 下块角点 4 位于 x=1、y=1、z=1。
        (4, 1.0, 1.0, 1.0),
        # 下块角点 5 位于 x=0、y=1、z=0。
        (5, 0.0, 1.0, 0.0),
        # 下块角点 6 位于坐标原点。
        (6, 0.0, 0.0, 0.0),
        # 下块角点 7 位于 x=0、y=0、z=1。
        (7, 0.0, 0.0, 1.0),
        # 下块角点 8 位于 x=0、y=1、z=1。
        (8, 0.0, 1.0, 1.0),
        # 下块边中间节点 9。
        (9, 1.0, 0.5, 0.0),
        # 下块边中间节点 10。
        (10, 1.0, 0.0, 0.5),
        # 下块接触面边中间节点 11。
        (11, 1.0, 0.5, 1.0),
        # 下块边中间节点 12。
        (12, 1.0, 1.0, 0.5),
        # 下块边中间节点 13。
        (13, 0.5, 1.0, 0.0),
        # 下块边中间节点 14。
        (14, 0.5, 0.0, 0.0),
        # 下块接触面边中间节点 15。
        (15, 0.5, 0.0, 1.0),
        # 下块接触面边中间节点 16。
        (16, 0.5, 1.0, 1.0),
        # 下块边中间节点 17。
        (17, 0.0, 0.5, 0.0),
        # 下块边中间节点 18。
        (18, 0.0, 0.0, 0.5),
        # 下块接触面边中间节点 19。
        (19, 0.0, 0.5, 1.0),
        # 下块边中间节点 20。
        (20, 0.0, 1.0, 0.5),
        # 上块接触面角点 21 保持在主面高度。
        (21, 0.75, 0.75, 1.0),
        # 上块接触面角点 22 保持在主面高度。
        (22, 0.75, 0.25, 1.0),
        # 上块顶面角点 23。
        (23, 0.75, 0.25, 1.5),
        # 上块顶面角点 24。
        (24, 0.75, 0.75, 1.5),
        # 上块接触面角点 25 保持在主面高度。
        (25, 0.25, 0.75, 1.0),
        # 上块接触面角点 26 保持在主面高度。
        (26, 0.25, 0.25, 1.0),
        # 上块顶面角点 27。
        (27, 0.25, 0.25, 1.5),
        # 上块顶面角点 28。
        (28, 0.25, 0.75, 1.5),
        # 上块接触边中间节点 29 使用实验参数 mid_z。
        (29, 0.75, 0.5, mid_z),
        # 上块竖边中间节点 30。
        (30, 0.75, 0.25, 1.25),
        # 上块顶面边中间节点 31。
        (31, 0.75, 0.5, 1.5),
        # 上块竖边中间节点 32。
        (32, 0.75, 0.75, 1.25),
        # 上块接触边中间节点 33 使用实验参数 mid_z。
        (33, 0.5, 0.75, mid_z),
        # 上块接触边中间节点 34 使用实验参数 mid_z。
        (34, 0.5, 0.25, mid_z),
        # 上块顶面边中间节点 35。
        (35, 0.5, 0.25, 1.5),
        # 上块顶面边中间节点 36。
        (36, 0.5, 0.75, 1.5),
        # 上块接触边中间节点 37 使用实验参数 mid_z。
        (37, 0.25, 0.5, mid_z),
        # 上块竖边中间节点 38。
        (38, 0.25, 0.25, 1.25),
        # 上块顶面边中间节点 39。
        (39, 0.25, 0.5, 1.5),
        # 上块竖边中间节点 40。
        (40, 0.25, 0.75, 1.25),
    ]


# 定义代表模型 deck 生成函数，使工具能够真实改变一个登记因素。
def _build_representative_deck(mid_z: float, initial_increment: float, penalty_stiffness: float | None, contact_type: str) -> str:
    # 初始化包含模型来源和用途边界的 CalculiX 注释行。
    lines = [
        # 标记该 deck 来自官方 contact4 的缩减机制实验。
        "** Representative model derived from CalculiX test/contact4.inp.",
        # 标记该 deck 不复刻原始论坛圆柱孔几何。
        "** This is a controlled mechanism experiment, not the original forum geometry.",
        # 开始节点块并建立全节点集合。
        "*NODE, NSET=Nall",
    ]
    # 遍历固定拓扑下的四十个参数化节点。
    for node_id, x_value, y_value, z_value in _representative_nodes(mid_z):
        # 使用科学计数法写入可复现节点坐标。
        lines.append(f"{node_id:8d},{x_value:.12e},{y_value:.12e},{z_value:.12e}")
    # 追加两个 C3D20R 单元、约束、接触、材料、步骤和输出请求。
    lines.extend([
        # 使用与原帖同族的二十节点减缩积分实体单元。
        "*ELEMENT, TYPE=C3D20R, ELSET=Eall",
        # 写入下部实体第一行连接。
        "1,1,2,3,4,5,6,7,8,9,10,",
        # 写入下部实体第二行连接。
        "11,12,17,18,19,20,13,14,15,16",
        # 写入上部实体第一行连接。
        "2,21,22,23,24,25,26,27,28,29,30,",
        # 写入上部实体第二行连接。
        "31,32,37,38,39,40,33,34,35,36",
        # 定义下块底面固定节点集。
        "*NSET,NSET=Nfix",
        # 列出下块底面固定节点。
        "1,2,5,6,9,13,14,17",
        # 固定下块全部三个平移自由度。
        "*BOUNDARY",
        # 对 Nfix 的自由度 1 至 3 施加零位移。
        "Nfix,1,3",
        # 定义上块顶面横向约束节点集。
        "*NSET,NSET=Nfix2",
        # 列出上块顶面和顶边中间节点。
        "23,24,27,28,31,35,36,39",
        # 开始上块横向边界条件。
        "*BOUNDARY",
        # 只固定 x 和 y，使上块可沿接触法向移动。
        "Nfix2,1,2",
        # 定义上块接触面从属节点集。
        "*NSET,NSET=Nslav",
        # 列出四角点和四个边中间节点。
        "21,22,25,26,29,33,34,37",
        # 定义上块底面为从属接触面。
        "*SURFACE,NAME=Sslav",
        # 使用上部单元的 S3 面。
        "2,S3",
        # 定义下块顶面为主接触面。
        "*SURFACE,NAME=Smast",
        # 使用下部单元的 S5 面。
        "1,S5",
        # 定义参数化接触对，接触形式由工具登记字段决定。
        f"*CONTACT PAIR,INTERACTION=SI1,TYPE={contact_type}",
        # 指定从属面和主面顺序。
        "Sslav,Smast",
        # 定义接触相互作用名称。
        "*SURFACE INTERACTION,NAME=SI1",
        # 使用硬压力-过闭合语义；CalculiX 会转为线性 penalty 实现。
        "*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=HARD",
    ])
    # 只有工具明确选择显式 penalty 时才写入数值行。
    if penalty_stiffness is not None:
        # 写入 penalty 刚度和单位大间隙拉力参数。
        lines.append(f"{penalty_stiffness:.12g},1.")
    # 追加材料、截面、非线性静力步骤、载荷和输出请求。
    lines.extend([
        # 定义唯一线弹性材料。
        "*MATERIAL,NAME=EL",
        # 开始弹性常数。
        "*ELASTIC",
        # 使用官方 contact4 的杨氏模量和零泊松比。
        "210000.,0.0",
        # 把材料分配给两个实体。
        "*SOLID SECTION,ELSET=Eall,MATERIAL=EL",
        # 开始考虑几何非线性的静力步骤。
        "*STEP,NLGEOM",
        # 开始静力增量配置。
        "*STATIC",
        # 写入工具选择的初始增量和单位总步长。
        f"{initial_increment:.12g},1.",
        # 对上块顶面施加法向压力。
        "*DLOAD",
        # 使用官方 contact4 的 P5=100 载荷。
        "2,P5,100.0",
        # 请求最终增量的全部节点位移和反力。
        "*NODE PRINT,NSET=Nall,FREQUENCY=10",
        # 同时输出位移 U 和反力 RF 以检查物理响应。
        "U,RF",
        # 结束静力步骤。
        "*END STEP",
    ])
    # 用 LF 和末尾换行生成稳定 deck 文本。
    return "\n".join(lines) + "\n"


# 定义浮点日志字段提取函数。
def _float_values(pattern: str, text: str) -> list[float]:
    # 初始化成功解析的浮点值列表。
    values: list[float] = []
    # 遍历正则匹配到的第一个捕获组。
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        # 尝试把 Fortran 或普通科学计数法转换为 Python 浮点数。
        try:
            # 将 D 指数统一替换为 E 指数后追加。
            values.append(float(match.group(1).replace("D", "E").replace("d", "e")))
        # 捕获单个异常字段而不丢弃其余求解证据。
        except ValueError:
            # 跳过无法解释的字段。
            continue
    # 返回全部有效值。
    return values


# 定义 DAT 位移与反力摘要提取函数。
def _parse_dat_summary(dat_text: str) -> dict[str, Any]:
    # 定义节点三分量行解析辅助函数。
    def parse_rows(section_text: str) -> dict[int, tuple[float, float, float]]:
        # 初始化按节点编号索引的结果。
        parsed: dict[int, tuple[float, float, float]] = {}
        # 提取当前结果区的节点三分量行。
        rows = re.findall(r"^\s*(\d+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s*$", section_text, flags=re.MULTILINE)
        # 遍历每个原始结果行。
        for node_id, first_value, second_value, third_value in rows:
            # 尝试解析当前节点的三个分量。
            try:
                # 保存统一为 E 指数的三分量浮点值。
                parsed[int(node_id)] = (float(first_value.replace("D", "E").replace("d", "e")), float(second_value.replace("D", "E").replace("d", "e")), float(third_value.replace("D", "E").replace("d", "e")))
            # 单行异常时保留其余节点证据。
            except ValueError:
                # 跳过当前异常行。
                continue
        # 返回当前结果区。
        return parsed
    # 定位最终位移区，并在 forces 标题前结束。
    displacement_match = re.search(r"displacements\s*\([^)]*\).*?\n(?P<body>.*?)(?=\n\s*forces\s*\(|\Z)", dat_text, flags=re.IGNORECASE | re.DOTALL)
    # 定位最终力区。
    force_match = re.search(r"forces\s*\([^)]*\).*?\n(?P<body>.*?)(?=\n\s*[A-Za-z].*?\(|\Z)", dat_text, flags=re.IGNORECASE | re.DOTALL)
    # 解析位移节点行。
    displacements = parse_rows(displacement_match.group("body")) if displacement_match else {}
    # 解析力节点行。
    forces = parse_rows(force_match.group("body")) if force_match else {}
    # 收集全部位移分量绝对值。
    displacement_components = [abs(component) for values in displacements.values() for component in values]
    # 收集全部力分量绝对值。
    force_components = [abs(component) for values in forces.values() for component in values]
    # 定义上块接触面四角点和四个边中间节点。
    upper_contact_node_ids = (21, 22, 25, 26, 29, 33, 34, 37)
    # 提取存在的上块接触面法向位移。
    upper_contact_uz = [displacements[node_id][2] for node_id in upper_contact_node_ids if node_id in displacements]
    # 定义下块底面固定节点，与 deck 中 Nfix 完全一致。
    fixed_bottom_node_ids = (1, 2, 5, 6, 9, 13, 14, 17)
    # 汇总下块底面法向反力。
    bottom_reaction_z_sum = sum(forces[node_id][2] for node_id in fixed_bottom_node_ids if node_id in forces)
    # 汇总所有节点法向力，用于检查整体平衡残差。
    all_force_z_sum = sum(values[2] for values in forces.values())
    # 返回分离后的位移、反力和平衡摘要。
    return {"displacement_row_count": len(displacements), "force_row_count": len(forces), "max_absolute_displacement": max(displacement_components) if displacement_components else None, "max_absolute_force_component": max(force_components) if force_components else None, "mean_upper_contact_uz": (sum(upper_contact_uz) / len(upper_contact_uz)) if upper_contact_uz else None, "bottom_reaction_z_sum": bottom_reaction_z_sum if forces else None, "all_force_z_sum": all_force_z_sum if forces else None, "contains_displacements": bool(displacements), "contains_forces": bool(forces)}


# 定义 CalculiX 版本探测函数，记录 stdout、stderr、路径和二进制哈希。
def _solver_version(ccx_executable: str) -> dict[str, Any]:
    # 执行 ccx -v；部分发行版即使成功打印版本也返回非零码。
    result = subprocess.run([ccx_executable, "-v"], capture_output=True, text=True, timeout=10, check=False)
    # 合并标准输出和错误输出以适配不同二进制。
    combined = (result.stdout or "") + (result.stderr or "")
    # 解析首个 CalculiX 版本号。
    match = re.search(r"(?:Version|version)\s+([0-9.]+)", combined)
    # 解析可执行文件的绝对路径。
    resolved_path = Path(ccx_executable).resolve()
    # 返回版本探测记录。
    return {"command": [ccx_executable, "-v"], "return_code": result.returncode, "version": match.group(1) if match else None, "stdout": combined.strip(), "executable_path": str(resolved_path), "executable_sha256": _sha256_file(resolved_path) if resolved_path.is_file() else None}


# 定义真实 CalculiX 变体执行函数，并在同一运行内复用相同配置。
def _run_solver_variant(ccx_executable: str, output_dir: Path, cache: dict[str, dict[str, Any]], label: str, mid_z: float, initial_increment: float, penalty_stiffness: float | None, contact_type: str) -> dict[str, Any]:
    # 构造唯一决定 deck 内容的配置对象。
    configuration = {"mid_z": mid_z, "initial_increment": initial_increment, "penalty_stiffness": penalty_stiffness, "contact_type": contact_type}
    # 计算稳定配置键，防止同一变体被重复求解。
    cache_key = hashlib.sha256(_canonical_json(configuration).encode("utf-8")).hexdigest()
    # 已运行相同配置时返回带 reused 标记的副本。
    if cache_key in cache:
        # 复制已有摘要，避免修改缓存原对象。
        reused = dict(cache[cache_key])
        # 标记本次工具请求没有再次启动求解器。
        reused["reused"] = True
        # 记录当前请求标签，便于追踪为什么复用了结果。
        reused["requested_label"] = label
        # 返回复用摘要。
        return reused
    # 以标签和配置哈希建立唯一求解目录。
    run_dir = output_dir / "solver_runs" / f"{label}-{cache_key[:10]}"
    # 创建求解目录。
    run_dir.mkdir(parents=True, exist_ok=True)
    # 使用安全的固定 job 名称，避免 label 中的字符影响 ccx。
    job_name = "contact_case"
    # 定位输入 deck 文件。
    deck_path = run_dir / f"{job_name}.inp"
    # 生成当前配置的完整 deck 文本。
    deck_text = _build_representative_deck(mid_z, initial_increment, penalty_stiffness, contact_type)
    # 写入生成 deck。
    deck_path.write_text(deck_text, encoding="utf-8", newline="\n")
    # 记录求解开始墙钟时间。
    started = time.monotonic()
    # 初始化超时状态。
    timed_out = False
    # 尝试执行真实 CalculiX。
    try:
        # 在独立目录中运行 job，并捕获公开控制台输出。
        process = subprocess.run([ccx_executable, job_name], cwd=run_dir, capture_output=True, text=True, timeout=SOLVER_TIMEOUT_SECONDS, check=False)
        # 保存求解器返回码。
        return_code = process.returncode
        # 合并标准输出与错误输出。
        console_text = (process.stdout or "") + (process.stderr or "")
    # 捕获超过固定三十秒的微型求解。
    except subprocess.TimeoutExpired as exc:
        # 标记当前求解已超时。
        timed_out = True
        # 使用 124 表示本地超时终止。
        return_code = 124
        # 保存超时前已有的标准输出和错误输出。
        console_text = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + ((exc.stderr or "") if isinstance(exc.stderr, str) else "")
    # 计算本次求解墙钟时间。
    wall_seconds = time.monotonic() - started
    # 写入完整公开控制台日志。
    console_path = run_dir / "console.log"
    # 保存控制台文本并确保末尾换行。
    console_path.write_text(console_text + ("" if console_text.endswith("\n") else "\n"), encoding="utf-8")
    # 定位 CalculiX 可能生成的主要结果文件。
    result_paths = {suffix: run_dir / f"{job_name}.{suffix}" for suffix in ("sta", "cvg", "dat", "frd")}
    # 读取存在的 STA 内容。
    sta_text = result_paths["sta"].read_text(encoding="utf-8", errors="replace") if result_paths["sta"].is_file() else ""
    # 读取存在的 DAT 内容。
    dat_text = result_paths["dat"].read_text(encoding="utf-8", errors="replace") if result_paths["dat"].is_file() else ""
    # 提取所有最大残差力。
    residual_values = _float_values(r"largest residual force=\s*([-+0-9.EeDd]+)", console_text)
    # 提取所有最大位移修正。
    correction_values = _float_values(r"largest correction to disp=\s*([-+0-9.EeDd]+)", console_text)
    # 提取各迭代接触弹簧数量。
    contact_spring_values = [int(value) for value in re.findall(r"Number of contact spring elements=(\d+)", console_text)]
    # 收集存在结果文件的大小和哈希。
    artifact_files = {suffix: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)} for suffix, path in result_paths.items() if path.is_file()}
    # 依据求解器明确终止信息判断完成，而不是只看返回码或 FRD 存在。
    job_finished = "Job finished" in console_text
    # 识别 too many cutbacks 明确失败。
    too_many_cutbacks = "too many cutbacks" in console_text.lower()
    # 构造本次求解摘要。
    summary = {
        # 保存稳定求解标识。
        "solve_id": f"SOLVE-{cache_key[:12].upper()}",
        # 保存首次请求标签。
        "label": label,
        # 标记不是复用结果。
        "reused": False,
        # 保存唯一 deck 配置。
        "configuration": configuration,
        # 保存实际命令。
        "command": [ccx_executable, job_name],
        # 保存求解目录。
        "run_dir": str(run_dir),
        # 保存 deck 路径、大小和哈希。
        "deck": {"path": str(deck_path), "size_bytes": deck_path.stat().st_size, "sha256": _sha256_file(deck_path)},
        # 保存进程层结果。
        "return_code": return_code,
        # 保存是否超时。
        "timed_out": timed_out,
        # 保存墙钟时间。
        "wall_seconds": wall_seconds,
        # 保存求解器明确完成标志。
        "job_finished": job_finished,
        # 保存明确 cutback 失败标志。
        "too_many_cutbacks": too_many_cutbacks,
        # 保存组合成功判据。
        "completed": bool(job_finished and not too_many_cutbacks and not timed_out),
        # 保存迭代中 no convergence 文本出现次数。
        "no_convergence_count": len(re.findall(r"\bno convergence\b", console_text, flags=re.IGNORECASE)),
        # 保存 STA 中增量尝试行数。
        "increment_attempt_count": len(re.findall(r"^\s*\d+\s+\d+\s+\d+[A-Z]?\s+\d+\s+", sta_text, flags=re.MULTILINE)),
        # 保存最大残差力。
        "max_residual_force": max(residual_values) if residual_values else None,
        # 保存最大绝对位移修正。
        "max_absolute_displacement_correction": max((abs(value) for value in correction_values), default=None),
        # 保存最小正位移修正，用于观察准零修正但不把它当结论。
        "min_positive_displacement_correction": min((abs(value) for value in correction_values if abs(value) > 0.0), default=None),
        # 保存接触活动集范围。
        "contact_spring_count_min": min(contact_spring_values) if contact_spring_values else None,
        # 保存接触活动集最大值。
        "contact_spring_count_max": max(contact_spring_values) if contact_spring_values else None,
        # 保存 DAT 中位移和反力的紧凑摘要。
        "dat_summary": _parse_dat_summary(dat_text),
        # 保存原始结果工件清单。
        "artifacts": artifact_files,
        # 保存控制台日志路径和哈希。
        "console": {"path": str(console_path), "size_bytes": console_path.stat().st_size, "sha256": _sha256_file(console_path)},
    }
    # 写入单次求解机器摘要。
    _write_json(run_dir / "solver_summary.json", summary)
    # 把摘要存入本次运行缓存。
    cache[cache_key] = summary
    # 返回新执行的求解摘要。
    return summary


# 定义发送给模型的紧凑求解视图，避免重复大日志浪费缓存和费用。
def _public_solver_summary(summary: dict[str, Any]) -> dict[str, Any]:
    # 返回可判别但不含冗长路径和原始日志的字段。
    return {
        # 引用唯一求解 ID。
        "solve_id": summary.get("solve_id"),
        # 公开实际配置。
        "configuration": summary.get("configuration"),
        # 公开是否复用。
        "reused": summary.get("reused"),
        # 公开求解器进程返回码。
        "return_code": summary.get("return_code"),
        # 公开求解器明确完成状态。
        "job_finished": summary.get("job_finished"),
        # 公开 cutback 失败状态。
        "too_many_cutbacks": summary.get("too_many_cutbacks"),
        # 公开组合完成状态。
        "completed": summary.get("completed"),
        # 公开 no convergence 次数。
        "no_convergence_count": summary.get("no_convergence_count"),
        # 公开增量尝试数量。
        "increment_attempt_count": summary.get("increment_attempt_count"),
        # 公开最大残差力。
        "max_residual_force": summary.get("max_residual_force"),
        # 公开最大位移修正。
        "max_absolute_displacement_correction": summary.get("max_absolute_displacement_correction"),
        # 公开接触活动集范围。
        "contact_spring_count_range": [summary.get("contact_spring_count_min"), summary.get("contact_spring_count_max")],
        # 公开最终 DAT 数值摘要。
        "dat_summary": summary.get("dat_summary"),
        # 公开 deck 哈希以支持复核。
        "deck_sha256": summary.get("deck", {}).get("sha256"),
    }


# 定义代表模型的初始成功/失败对照执行函数。
def _run_initial_evidence(ccx_executable: str, output_dir: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # 运行四个边中间节点与角点共面的健康对照。
    aligned = _run_solver_variant(ccx_executable, output_dir, cache, "initial-aligned-control", MASTER_SURFACE_Z, DEFAULT_INITIAL_INCREMENT, None, DEFAULT_CONTACT_TYPE)
    # 运行只把四个边中间节点压入主面的目标变体。
    penetrating = _run_solver_variant(ccx_executable, output_dir, cache, "initial-penetrating-target", SEARCH_MIDSIDE_Z, DEFAULT_INITIAL_INCREMENT, None, DEFAULT_CONTACT_TYPE)
    # 返回中性实验定义和实际结果，不附加预设根因标签。
    return {"single_changed_field": "upper_contact_face_edge_midside_nodes.z", "control": _public_solver_summary(aligned), "target": _public_solver_summary(penetrating), "known_geometric_delta": {"corner_z": MASTER_SURFACE_Z, "control_midside_z": MASTER_SURFACE_Z, "target_midside_z": SEARCH_MIDSIDE_Z, "target_midside_overclosure": MASTER_SURFACE_Z - SEARCH_MIDSIDE_Z}}


# 定义全部中性工具目录，目录描述能力而不透露实际运行结果。
def _tool_catalog() -> dict[str, str]:
    # 返回固定工具 ID 和公开用途。
    return {
        # 允许检查两个原始附件的真实语义差异。
        "inspect_original_deck_semantics": "读取冻结的原 deck 规模、接触卡、网格和步骤差异，判断现有粗/细附件是否可作为单因素证据。",
        # 允许检查代表模型角点和边中间节点的初始几何关系。
        "inspect_representative_contact_geometry": "报告代表模型角点与边中间节点相对主面的初始间隙/过闭合，不运行新求解。",
        # 允许按需读取官方接触算法事实。
        "lookup_official_contact_semantics": "返回 CalculiX 官方手册中 face-to-face 匹配、二次单元和 pressure-overclosure 的固定出处与摘要。",
        # 允许只改变初始增量并在搜索和留出幅度上求解。
        "run_smaller_initial_increment_pair": "保持几何、接触和 penalty 不变，只把初始增量改为模型给定的正数；同时运行搜索幅度和留出幅度。",
        # 允许只显式声明线性 penalty 并在搜索和留出幅度上求解。
        "run_explicit_penalty_pair": "保持几何、接触和增量不变，只写入模型给定的正 penalty stiffness；同时运行搜索幅度和留出幅度。",
        # 允许只切换接触离散形式。
        "run_surface_to_surface_pair": "保持其余字段不变，只把代表模型从 node-to-surface 切换为 surface-to-surface，并与现有目标结果比较。",
        # 允许模型在证据足够或资源不足时主动结束。
        "finish": "停止工具调用并形成当前证据所允许的最终或保留结论。",
    }


# 定义工具结果构造函数，统一记录输入哈希、changed_fields 和限制。
def _tool_result(tool_id: str, arguments: dict[str, Any], changed_fields: list[str], observations: dict[str, Any], limitations: list[str]) -> dict[str, Any]:
    # 构造可追溯的中性工具事件。
    return {"tool_event_id": f"TOOL-{hashlib.sha256(_canonical_json({'tool_id': tool_id, 'arguments': arguments, 'observations': observations}).encode('utf-8')).hexdigest()[:12].upper()}", "tool_id": tool_id, "input_hash": hashlib.sha256(_canonical_json(arguments).encode("utf-8")).hexdigest().upper(), "changed_fields": changed_fields, "observations": observations, "limitations": limitations}


# 定义模型所选动作执行函数，不按预设答案修改其选择。
def _execute_tool(action: dict[str, Any], source_audit: dict[str, Any], ccx_executable: str, output_dir: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # 读取模型选择的工具 ID。
    tool_id = str(action.get("tool_id") or "").strip()
    # 读取参数对象；非对象参数转为空对象并留在结果限制中。
    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    # 处理原始 deck 语义检查。
    if tool_id == "inspect_original_deck_semantics":
        # 返回 source audit 中全部中性事实和混杂限制。
        return _tool_result(tool_id, arguments, [], {"entries": source_audit.get("entries"), "shared_deck_facts": source_audit.get("shared_deck_facts"), "author_observations": source_audit.get("author_observations"), "pair_differences": {"mesh_and_element_count_changed": True, "pin2_contact_face_count_changed": True, "contact_activation_history_changed": True}}, list(source_audit.get("audit_limitations", [])))
    # 处理代表模型几何检查。
    if tool_id == "inspect_representative_contact_geometry":
        # 返回角点、边中间节点和局部尺度的确定性几何报告。
        observations = {"master_surface_z": MASTER_SURFACE_Z, "upper_corner_z_values": [MASTER_SURFACE_Z] * 4, "search_edge_midside_z_values": [SEARCH_MIDSIDE_Z] * 4, "search_edge_midside_overclosure_values": [MASTER_SURFACE_Z - SEARCH_MIDSIDE_Z] * 4, "holdout_edge_midside_overclosure_values": [MASTER_SURFACE_Z - HOLDOUT_MIDSIDE_Z] * 4, "upper_element_height": 0.5, "search_overclosure_to_element_height_ratio": (MASTER_SURFACE_Z - SEARCH_MIDSIDE_Z) / 0.5}
        # 返回几何观察并强调面弯曲与物理干涉尚未分离。
        return _tool_result(tool_id, arguments, [], observations, ["参数化中间节点既改变初始过闭合，也改变二次面的曲率；代表模型不能区分原模型中的设计过盈与离散曲率误差。"])
    # 处理官方手册查阅。
    if tool_id == "lookup_official_contact_semantics":
        # 返回已冻结的官方出处和不带案例答案的算法事实。
        observations = {"sources": [{"title": "CalculiX contact internals", "url": "https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node307.html"}, {"title": "CalculiX *SURFACE INTERACTION", "url": "https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node249.html"}, {"title": "CalculiX official contact4 test", "url": "https://github.com/Dhondtguido/CalculiX/blob/master/test/contact4.inp"}], "facts": ["face-to-face 接触会基于重叠区域建立积分点，单个面可能产生多个接触积分位置。", "接触面匹配和活动状态与增量路径有关，二次面边中间节点参与表面几何。", "pressure-overclosure 数据控制线性 penalty 实现；未给出严格正刚度时 CalculiX 会选择默认值并发出警告。"]}
        # 返回手册事实且不把旧版本手册当作当前二进制实现证明。
        return _tool_result(tool_id, arguments, [], observations, ["手册页面版本较旧；本次具体行为仍以实际 ccx 运行日志为准。"])
    # 处理初始增量单因素对照。
    if tool_id == "run_smaller_initial_increment_pair":
        # 尝试读取模型给出的初始增量，缺失时使用 0.01 作为工具缺省实验值。
        try:
            # 把数值或数值字符串转换为浮点数。
            requested_increment = float(arguments.get("initial_increment", 0.01))
        # 非数值参数作为工具观察返回，不触发模型修复请求。
        except (TypeError, ValueError):
            # 返回可供下一轮读取的参数错误。
            return _tool_result(tool_id, arguments, [], {"status": "invalid_arguments", "message": "initial_increment must be numeric"}, ["没有启动 CalculiX。"])
        # 拒绝非正或大于总步长的数值，但把错误作为工具结果而不是模型门禁。
        if not 0.0 < requested_increment <= 1.0:
            # 返回可供下一轮读取的参数错误。
            return _tool_result(tool_id, arguments, [], {"status": "invalid_arguments", "message": "initial_increment must be within (0, 1]"}, ["没有启动 CalculiX。"])
        # 复用或运行搜索幅度默认目标。
        search_control = _run_solver_variant(ccx_executable, output_dir, cache, "increment-search-control", SEARCH_MIDSIDE_Z, DEFAULT_INITIAL_INCREMENT, None, DEFAULT_CONTACT_TYPE)
        # 只改变搜索幅度的初始增量。
        search_variant = _run_solver_variant(ccx_executable, output_dir, cache, "increment-search-variant", SEARCH_MIDSIDE_Z, requested_increment, None, DEFAULT_CONTACT_TYPE)
        # 运行留出幅度默认目标。
        holdout_control = _run_solver_variant(ccx_executable, output_dir, cache, "increment-holdout-control", HOLDOUT_MIDSIDE_Z, DEFAULT_INITIAL_INCREMENT, None, DEFAULT_CONTACT_TYPE)
        # 只改变留出幅度的初始增量。
        holdout_variant = _run_solver_variant(ccx_executable, output_dir, cache, "increment-holdout-variant", HOLDOUT_MIDSIDE_Z, requested_increment, None, DEFAULT_CONTACT_TYPE)
        # 返回两组单因素求解证据。
        return _tool_result(tool_id, arguments, ["step.initial_increment"], {"search": {"control": _public_solver_summary(search_control), "variant": _public_solver_summary(search_variant)}, "holdout": {"control": _public_solver_summary(holdout_control), "variant": _public_solver_summary(holdout_variant)}}, ["留出幅度是代表模型泛化检查，不等于原始圆柱孔模型验证。"])
    # 处理显式 penalty 单因素对照。
    if tool_id == "run_explicit_penalty_pair":
        # 尝试读取模型给出的正 penalty stiffness，缺失时使用登记的 1e5 实验值。
        try:
            # 把数值或数值字符串转换为浮点数。
            requested_penalty = float(arguments.get("penalty_stiffness", DEFAULT_EXPLICIT_PENALTY))
        # 非数值参数作为工具观察返回，不触发模型修复请求。
        except (TypeError, ValueError):
            # 返回可供下一轮读取的参数错误。
            return _tool_result(tool_id, arguments, [], {"status": "invalid_arguments", "message": "penalty_stiffness must be numeric"}, ["没有启动 CalculiX。"])
        # 限制微型实验的正数范围，防止无意义或溢出输入。
        if not 1.0 <= requested_penalty <= 1.0e9:
            # 返回参数错误且不启动求解。
            return _tool_result(tool_id, arguments, [], {"status": "invalid_arguments", "message": "penalty_stiffness must be within [1, 1e9]"}, ["没有启动 CalculiX。"])
        # 复用或运行搜索幅度的默认 penalty 目标。
        search_control = _run_solver_variant(ccx_executable, output_dir, cache, "penalty-search-control", SEARCH_MIDSIDE_Z, DEFAULT_INITIAL_INCREMENT, None, DEFAULT_CONTACT_TYPE)
        # 只写入搜索幅度的显式 penalty 数值。
        search_variant = _run_solver_variant(ccx_executable, output_dir, cache, "penalty-search-variant", SEARCH_MIDSIDE_Z, DEFAULT_INITIAL_INCREMENT, requested_penalty, DEFAULT_CONTACT_TYPE)
        # 运行留出幅度的默认 penalty 目标。
        holdout_control = _run_solver_variant(ccx_executable, output_dir, cache, "penalty-holdout-control", HOLDOUT_MIDSIDE_Z, DEFAULT_INITIAL_INCREMENT, None, DEFAULT_CONTACT_TYPE)
        # 只写入留出幅度的显式 penalty 数值。
        holdout_variant = _run_solver_variant(ccx_executable, output_dir, cache, "penalty-holdout-variant", HOLDOUT_MIDSIDE_Z, DEFAULT_INITIAL_INCREMENT, requested_penalty, DEFAULT_CONTACT_TYPE)
        # 返回两种几何幅度的单因素求解证据。
        return _tool_result(tool_id, arguments, ["surface_behavior.linear_penalty_stiffness"], {"search": {"control": _public_solver_summary(search_control), "variant": _public_solver_summary(search_variant)}, "holdout": {"control": _public_solver_summary(holdout_control), "variant": _public_solver_summary(holdout_variant)}}, ["较低 penalty 可能通过允许更大过闭合形成数值 workaround；必须检查位移、反力和原模型已有 penalty=69000 的事实。"])
    # 处理接触形式单因素对照。
    if tool_id == "run_surface_to_surface_pair":
        # 复用 node-to-surface 目标。
        node_to_surface = _run_solver_variant(ccx_executable, output_dir, cache, "contact-form-search-control", SEARCH_MIDSIDE_Z, DEFAULT_INITIAL_INCREMENT, None, DEFAULT_CONTACT_TYPE)
        # 只把接触形式切换为 surface-to-surface。
        surface_to_surface = _run_solver_variant(ccx_executable, output_dir, cache, "contact-form-search-variant", SEARCH_MIDSIDE_Z, DEFAULT_INITIAL_INCREMENT, None, "SURFACE TO SURFACE")
        # 返回单因素接触形式对照。
        return _tool_result(tool_id, arguments, ["contact_pair.type"], {"node_to_surface": _public_solver_summary(node_to_surface), "surface_to_surface": _public_solver_summary(surface_to_surface)}, ["代表模型的 surface-to-surface 结果可能受当前 CalculiX 版本和 SPOOLES 后端影响；不能外推为原帖 Pardiso 结果。"])
    # 处理模型主动结束。
    if tool_id == "finish":
        # 返回没有求解副作用的结束记录。
        return _tool_result(tool_id, arguments, [], {"status": "finished_by_model"}, [])
    # 未知工具也作为可观察错误返回，不发修复调用。
    return _tool_result(tool_id or "missing_tool_id", arguments, [], {"status": "unknown_tool", "available_tools": sorted(_tool_catalog())}, ["没有启动 CalculiX；模型选择未被脚本替换。"])


# 定义 DeepSeek 公共系统提示构造函数，注入通用 Skill 但不注入预期答案。
def _system_prompt(skill: dict[str, Any]) -> str:
    # 构造一份所有轮次逐字节复用的稳定系统提示。
    return (
        # 定义模型职责。
        "你是负责结构有限元故障诊断的工程代理。"
        # 要求只输出公开可审计判断。
        "你只输出公开、简洁、可核验的工程判断，不输出隐藏思维链。"
        # 要求根据当前事实而非关键词或常见套话行动。
        "你必须根据当前原始 deck 事实、CalculiX 求解观察和此前工具反馈行动，不能因为出现“接触”“裂纹”或“网格”等词就套用固定答案。"
        # 说明应用级 Skill 的角色。
        "下面的 engineering_skill 是通用经验流程，不是本案例答案；你可以遵循其中的证据纪律，但必须自己提出机制、选择动作和更新判断。\n"
        # 注入稳定 Skill JSON。
        + _canonical_json(skill)
    )


# 定义统一模型输出 schema 描述，允许调查中和最终状态共用。
def _decision_schema() -> dict[str, Any]:
    # 返回发送给模型的公开字段合同。
    return {
        # 要求明确当前结论状态。
        "state": "investigating | confirmed_fix | probable_workaround | narrowed_unresolved",
        # 要求保留多个竞争假设及可核验证据。
        "hypotheses": [{"id": "H1", "claim": "可证伪机制", "confidence": "0到1", "evidence_for": ["证据引用"], "evidence_against": ["反证引用"], "missing_discriminator": "仍缺什么", "predicted_signature": "该机制成立时应观察到什么"}],
        # 要求简述当前证据如何改变判断。
        "evidence_assessment": "公开工程判断",
        # 调查中选择一个工具；结束时使用 null。
        "next_action": {"tool_id": "工具目录中的一个 ID", "arguments": {}, "why_discriminative": "区分哪些假设", "prediction_if_positive": "执行前的正结果预测", "prediction_if_negative": "执行前的负结果预测"},
        # 允许模型主动结束。
        "stop": "布尔值",
        # 要求始终给用户一个当前可说的直接答复。
        "user_answer": "面向论坛用户的直接答复",
        # 最终状态应列出实际得到证据支持的动作。
        "implemented_actions": ["实际执行且有证据的动作"],
        # 最终状态应说明结论可用于什么。
        "can_use": ["有证据支持的用途"],
        # 最终状态应说明结论不可用于什么。
        "cannot_use": ["证据不足或禁止外推的用途"],
        # 要求引用 trace 中可定位的求解或工具 ID。
        "evidence_refs": ["SOLVE-... 或 TOOL-..."],
    }


# 定义初始模型消息包。
def _initial_packet(source_audit: dict[str, Any], manifest: dict[str, Any], solver_version: dict[str, Any], initial_evidence: dict[str, Any]) -> dict[str, Any]:
    # 返回中性任务、事实、预算、工具和输出合同。
    return {
        # 定义当前自然语言工程任务。
        "task": "诊断论坛用户的过盈接触网格细化失败，选择一项最有判别力的下一步，并最终给出具体但不过度外推的答复。",
        # 提供论坛用户观察。
        "forum_user_observations": source_audit.get("author_observations"),
        # 提供原附件共享卡片和关键差异。
        "original_deck_evidence": {"entries": source_audit.get("entries"), "shared_deck_facts": source_audit.get("shared_deck_facts"), "limitations": source_audit.get("audit_limitations")},
        # 提供来源明确的问题清单。
        "problem_manifest": manifest,
        # 提供真实求解器环境。
        "solver_environment": solver_version,
        # 提供首轮前已执行的成功/失败对照。
        "initial_solver_evidence": initial_evidence,
        # 提供不带答案的工具能力目录。
        "tool_catalog": _tool_catalog(),
        # 提供统一 JSON 输出合同。
        "output_schema": _decision_schema(),
        # 说明费用和执行边界。
        "budget": {"max_http_requests": MAX_HTTP_REQUESTS, "remaining_http_requests_including_this_one": MAX_HTTP_REQUESTS, "one_action_per_round": True, "sdk_retries": 0},
        # 要求首轮显式提出竞争机制和预先预测。
        "round_instruction": "这是第1轮。提出至少三个性质不同的可证伪假设，引用给定事实，选择一个最有判别力的动作，并在看到结果前写明正、负预测。不要假定代表模型等于原模型。只输出一个合法 JSON 对象。",
    }


# 定义工具反馈后的下一轮消息包。
def _next_round_packet(round_index: int, tool_result: dict[str, Any], remaining_requests: int, final_round: bool) -> dict[str, Any]:
    # 根据是否最后一轮生成明确但不指定答案的指令。
    instruction = "这是最后一次模型调用。根据全部已有事实和真实工具结果形成最终状态与用户答复；不要再选择工具，next_action 必须为 null，并诚实区分确认修复、可能 workaround 或仍未解决。" if final_round else "根据新工具结果更新所有竞争假设。若证据足够可以主动结束；否则只选择一个互补动作，并在执行前写明正、负预测。"
    # 返回工具结果和剩余预算。
    return {"tool_result": tool_result, "tool_catalog": _tool_catalog(), "output_schema": _decision_schema(), "budget": {"remaining_http_requests_including_this_one": remaining_requests, "sdk_retries": 0}, "round_instruction": f"这是第{round_index}轮。{instruction} 只输出一个合法 JSON 对象。"}


# 定义 usage 字段归一化函数，兼容 DeepSeek 和 OpenAI 风格缓存统计。
def _usage_summary(usage: dict[str, Any]) -> dict[str, int]:
    # 读取可选的嵌套 prompt token 明细。
    details = usage.get("prompt_tokens_details", {}) if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    # 优先读取 DeepSeek 顶层缓存命中字段，其次读取兼容字段。
    cache_hit = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0))
    # 读取缓存未命中 token。
    cache_miss = usage.get("prompt_cache_miss_tokens", 0)
    # 返回统一整数摘要。
    return {"prompt_tokens": int(usage.get("prompt_tokens", 0) or 0), "completion_tokens": int(usage.get("completion_tokens", 0) or 0), "total_tokens": int(usage.get("total_tokens", 0) or 0), "prompt_cache_hit_tokens": int(cache_hit or 0), "prompt_cache_miss_tokens": int(cache_miss or 0)}


# 定义一次且仅一次的 DeepSeek 调用函数。
def _call_model(client: Any, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    # 计算调用前完整消息历史哈希，便于核查增长前缀。
    history_sha256 = hashlib.sha256(_canonical_json(messages).encode("utf-8")).hexdigest().upper()
    # 发起一次真实 DeepSeek 请求，SDK 层重试在 client 构造时已关闭。
    response = client.chat.completions.create(model=model, messages=messages, response_format={"type": "json_object"}, max_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="medium", extra_body={"thinking": {"type": "enabled"}}, stream=False)
    # 读取唯一公开回复文本。
    raw_text = response.choices[0].message.content or ""
    # 保存服务返回的完整 usage 字段。
    raw_usage = response.usage.model_dump() if response.usage is not None else {}
    # 初始化 JSON 解析错误。
    parse_error: str | None = None
    # 尝试解析模型公开 JSON。
    try:
        # 解析公开回复文本。
        decision = json.loads(raw_text)
        # 顶层不是对象时视为不可执行输出。
        if not isinstance(decision, dict):
            # 设置结构错误。
            parse_error = "response JSON top level is not an object"
            # 使用空对象避免后续访问异常。
            decision = {}
    # 捕获非法 JSON，不发修复请求。
    except json.JSONDecodeError as exc:
        # 保存精确解析错误。
        parse_error = str(exc)
        # 使用空对象保存本轮失败。
        decision = {}
    # 返回原始文本、解析对象和服务元数据。
    return {"raw_text": raw_text, "decision": decision, "parse_error": parse_error, "metadata": {"requested_model": model, "response_model": response.model, "finish_reason": response.choices[0].finish_reason, "usage": raw_usage, "usage_summary": _usage_summary(raw_usage), "history_sha256": history_sha256}}


# 定义模型动作提取函数，只做结构读取，不按期望答案修正。
def _selected_action(decision: dict[str, Any]) -> dict[str, Any] | None:
    # 读取 next_action 字段。
    action = decision.get("next_action")
    # 对象动作原样返回。
    if isinstance(action, dict):
        # 返回模型提供的动作。
        return action
    # 其他类型表示没有可执行动作。
    return None


# 定义最终决策归一化函数，只映射字段名称而不补造工程结论。
def _final_decision(decision: dict[str, Any]) -> dict[str, Any]:
    # 返回论文生成器所需的稳定字段。
    return {
        # 保存模型给出的状态。
        "state": decision.get("state"),
        # 保存模型直接用户答复。
        "answer": str(decision.get("user_answer") or "").strip(),
        # 保存模型最终假设。
        "hypotheses": decision.get("hypotheses") if isinstance(decision.get("hypotheses"), list) else [],
        # 保存模型证据判断。
        "evidence_assessment": str(decision.get("evidence_assessment") or "").strip(),
        # 保存模型列出的实际动作。
        "implemented_actions": decision.get("implemented_actions") if isinstance(decision.get("implemented_actions"), list) else [],
        # 保存模型允许用途。
        "can_use": decision.get("can_use") if isinstance(decision.get("can_use"), list) else [],
        # 保存模型禁止外推用途。
        "cannot_use": decision.get("cannot_use") if isinstance(decision.get("cannot_use"), list) else [],
        # 保存模型引用的证据 ID。
        "evidence_refs": decision.get("evidence_refs") if isinstance(decision.get("evidence_refs"), list) else [],
    }


# 定义 Markdown 列表生成函数。
def _markdown_list(items: Any, empty_text: str) -> list[str]:
    # 只有非空列表才逐项输出。
    if isinstance(items, list) and items:
        # 返回每项一行的 Markdown 无序列表。
        return [f"- {str(item)}" for item in items]
    # 返回明确缺失说明，避免把空字段伪装成结论。
    return [f"- {empty_text}"]


# 定义完整论文 Markdown 生成函数。
def _render_markdown(trace: dict[str, Any], source_audit: dict[str, Any], skill_sha256: str, trace_sha256: str) -> str:
    # 读取最终 DeepSeek 决策。
    final = trace.get("final_decision", {})
    # 读取所有模型决策事件。
    model_events = [event for event in trace.get("events", []) if event.get("type") == "model_decision"]
    # 读取所有工具事件。
    tool_events = [event for event in trace.get("events", []) if event.get("type") == "tool_call"]
    # 读取全部实际求解摘要。
    solver_runs = trace.get("solver_runs", [])
    # 初始化论文行。
    lines = [
        # 写入论文主标题。
        "# 从论坛原始输入到真实求解反馈：DeepSeek 对 CalculiX 过盈接触不收敛的有限调用诊断",
        # 写入副标题。
        "",
        "## 摘要",
        # 概括研究对象和方法边界。
        "",
        "本文研究 DeepSeek 能否在不训练 DQN、不执行网格优化、不使用预设答案门禁的条件下，诊断一个真实 CalculiX 论坛过盈接触问题。实验取得并冻结原帖两个大型 C3D20R 输入文件，但因约一千一百六十万自由度、Pardiso 后端和标准 runner 资源限制，不冒充已执行原始大型模型。系统改用可快速运行的二次减缩积分接触代表模型，保留“边中间节点进入初始干涉后 cutback 失败”的机制，并让 DeepSeek在最多三次、同一增长消息历史中提出竞争假设、选择单因素实验、读取真实 ccx 反馈和形成最终答复。论文结论直接来自同一 trace 的 DeepSeek 最终决策，论文生成没有额外模型调用。",
        # 写入关键词。
        "",
        "**关键词：** CalculiX；C3D20R；非线性接触；过盈；DeepSeek；反事实求解；证据追踪",
        # 开始问题来源。
        "",
        "## 1. 原始工程问题与附件证据",
        "",
        f"原帖为 [{source_audit.get('forum', {}).get('title')}]({source_audit.get('forum', {}).get('url')})。截至证据冻结时，帖子没有作者确认的最终修复。公开附件归档 SHA-256 为 `{source_audit.get('download', {}).get('archive_sha256')}`。",
        "",
        "两个原始附件的关键事实如下：",
        "",
        "| 输入 | 节点 | C3D20R 单元 | PIN2 接触面 | 步骤 | REMOVE/ADD PIN2 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    # 逐个写入附件规模表。
    for entry in source_audit.get("entries", []):
        # 追加当前附件行。
        lines.append(f"| `{entry.get('name')}` | {entry.get('node_count')} | {entry.get('c3d20r_element_count')} | {entry.get('pin2_contact_face_count')} | {entry.get('step_count')} | {entry.get('removes_then_readds_pin2_contact')} |")
    # 说明原附件的因果混杂。
    lines.extend([
        # 增加段落间隔。
        "",
        "这两个附件同时改变网格规模、PIN2 接触面数量和接触激活历史，不能直接作为“只因网格细化而失败”的因果对照。",
        # 开始方法。
        "",
        "## 2. 方法：应用级 Skill、真实求解和有限调用 loop",
        "",
        f"运行时使用 `nonlinear-contact-diagnosis` 应用级 Skill（SHA-256 `{skill_sha256}`）。Skill 只提供证据纪律：区分事实与解释、保留竞争假设、实验前声明正反预测、一次只改一个字段、把失败求解作为证据、保留适用边界。它不包含本案例的参数、工具顺序或答案。",
        "",
        f"整个运行只创建一个 DeepSeek client，SDK 重试为 0，模型 HTTP 请求上限为 {MAX_HTTP_REQUESTS}。每轮把模型原始 JSON 和工具结果追加到同一消息历史，既保留真实多轮上下文，也让稳定前缀具备缓存条件。非法 JSON 不触发付费修复请求。",
        # 开始代表模型。
        "",
        "## 3. 代表模型和初始可复现现象",
        "",
        "代表模型来自 CalculiX 官方 `test/contact4.inp` 的两个二次实体接触块，统一为 C3D20R。正式对照使用 node-to-surface 以建立健康基线；这是机制实验替代，不是对原帖 surface-to-surface 的预设修复。对照与目标只改变上块接触面四个边中间节点的 z 坐标：1.00 对 0.95，四个角点均保持 z=1.00。",
        "",
        "| SOLVE-ID | mid-z | 初始增量 | penalty | 接触 | 完成 | no convergence | 增量尝试 | 最大修正 |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|",
    ])
    # 逐个写入真实求解表。
    for run in solver_runs:
        # 读取当前配置。
        config = run.get("configuration", {})
        # 格式化可选 penalty。
        penalty_text = "default" if config.get("penalty_stiffness") is None else str(config.get("penalty_stiffness"))
        # 追加求解摘要行。
        lines.append(f"| `{run.get('solve_id')}` | {config.get('mid_z')} | {config.get('initial_increment')} | {penalty_text} | {config.get('contact_type')} | {run.get('completed')} | {run.get('no_convergence_count')} | {run.get('increment_attempt_count')} | {run.get('max_absolute_displacement_correction')} |")
    # 开始逐轮决策链。
    lines.extend([
        # 增加段落间隔。
        "",
        "## 4. DeepSeek 决策链",
    ])
    # 逐轮输出模型公开判断。
    for event in model_events:
        # 读取当前决策。
        decision = event.get("decision", {})
        # 写入轮次标题。
        lines.extend(["", f"### 第 {event.get('iteration')} 轮", "", f"- 状态：`{decision.get('state')}`", f"- 证据判断：{decision.get('evidence_assessment')}", f"- 选择动作：`{(decision.get('next_action') or {}).get('tool_id') if isinstance(decision.get('next_action'), dict) else None}`", f"- 正结果预测：{(decision.get('next_action') or {}).get('prediction_if_positive') if isinstance(decision.get('next_action'), dict) else None}", f"- 负结果预测：{(decision.get('next_action') or {}).get('prediction_if_negative') if isinstance(decision.get('next_action'), dict) else None}", f"- 当前用户答复：{decision.get('user_answer')}"])
        # 写入当前竞争假设。
        for hypothesis in decision.get("hypotheses", []) if isinstance(decision.get("hypotheses"), list) else []:
            # 追加假设 ID、置信度和主张。
            lines.append(f"- 假设 `{hypothesis.get('id')}`，置信度 `{hypothesis.get('confidence')}`：{hypothesis.get('claim')}")
    # 写入工具执行链。
    lines.extend(["", "### 工具反馈", ""])
    # 没有工具时写明模型直接结束。
    if not tool_events:
        # 追加无工具说明。
        lines.append("- 模型未选择执行工具。")
    # 逐个写入工具事件。
    for event in tool_events:
        # 读取工具结果。
        result = event.get("result", {})
        # 追加工具、changed fields 和事件 ID。
        lines.append(f"- 第 {event.get('iteration')} 轮执行 `{event.get('request', {}).get('tool_id')}`，事件 `{result.get('tool_event_id')}`，changed_fields={result.get('changed_fields')}。")
    # 写入最终决策。
    lines.extend([
        # 开始最终答复。
        "",
        "## 5. DeepSeek 最终工程答复",
        "",
        f"**最终状态：** `{final.get('state')}`",
        "",
        "**以下答复为 trace 中 DeepSeek 最终原文，而非论文模板预写结论：**",
        "",
        str(final.get("answer") or "模型未形成非空最终答复。"),
        "",
        "### 已实施动作",
        "",
    ])
    # 写入模型列出的已实施动作。
    lines.extend(_markdown_list(final.get("implemented_actions"), "模型未记录已实施动作。"))
    # 写入允许用途。
    lines.extend(["", "### 可使用范围", ""])
    # 追加可用范围列表。
    lines.extend(_markdown_list(final.get("can_use"), "模型未记录可使用范围。"))
    # 写入禁止外推用途。
    lines.extend(["", "### 不可使用范围", ""])
    # 追加不可用范围列表。
    lines.extend(_markdown_list(final.get("cannot_use"), "模型未记录不可使用范围。"))
    # 写入模型证据引用。
    lines.extend(["", "### 模型引用的证据", ""])
    # 追加证据引用列表。
    lines.extend(_markdown_list(final.get("evidence_refs"), "模型未记录证据引用。"))
    # 写入讨论和限制。
    lines.extend([
        # 开始讨论。
        "",
        "## 6. 讨论与适用边界",
        "",
        "本实验能评价 DeepSeek 是否会根据细微的 C3D20R 求解差异改判，不能证明原始大型圆柱孔模型已经修复。尤其需要同时考虑：原输入已经显式使用约 69000 的线性 penalty；代表模型的 penalty 结果可能只是数值 regularization；原附件还混入接触重新激活和接触面离散变化；当前 runner 后端不是原 deck 指定的 Pardiso。",
        "",
        "只有在原几何、原 contact pair、批准后端和完整边界上复验位移、反力、接触压力及穿透后，候选措施才可升级为原问题的确认修复。",
        # 写入调用和缓存审计。
        "",
        "## 7. 调用、缓存与来源审计",
        "",
        f"- 实际 DeepSeek HTTP 请求：{trace.get('http_requests_attempted')} / {MAX_HTTP_REQUESTS}",
        f"- API client 数量：{trace.get('api_client_count')}",
        f"- SDK 自动重试：{trace.get('sdk_max_retries')}",
        f"- 缓存命中 token：{trace.get('cache_audit', {}).get('prompt_cache_hit_tokens')}",
        f"- 缓存未命中 token：{trace.get('cache_audit', {}).get('prompt_cache_miss_tokens')}",
        f"- 论文额外 DeepSeek 调用：0",
        f"- agent trace SHA-256：`{trace_sha256}`",
        # 写入参考文献。
        "",
        "## 参考资料",
        "",
        f"1. CalculiX Discourse, [{source_audit.get('forum', {}).get('title')}]({source_audit.get('forum', {}).get('url')}).",
        "2. CalculiX official repository, [test/contact4.inp](https://github.com/Dhondtguido/CalculiX/blob/master/test/contact4.inp).",
        "3. CalculiX documentation, [Contact internals](https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node307.html).",
        "4. CalculiX documentation, [*SURFACE INTERACTION](https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node249.html).",
    ])
    # 返回末尾带换行的完整 Markdown。
    return "\n".join(lines) + "\n"


# 定义 PDF 字体注册函数，优先使用支持中文的系统字体。
def _register_pdf_font() -> str:
    # 延迟导入 ReportLab 字体注册器，使离线求解仍可在缺少 PDF 依赖时记录错误。
    from reportlab.pdfbase import pdfmetrics
    # 延迟导入 TrueType 字体类型。
    from reportlab.pdfbase.ttfonts import TTFont
    # 读取可选的显式字体路径。
    configured = os.environ.get("PAPER_CJK_FONT")
    # 建立 Windows 和 Ubuntu 常见 CJK 字体候选。
    candidates = [
        # 使用用户显式配置的 CJK 字体。
        Path(configured) if configured else None,
        # 使用 GitHub Actions 安装的 Noto CJK 字体。
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        # 使用 Ubuntu 文泉驿字体作为第二候选。
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        # 使用 Windows 微软雅黑集合。
        Path("C:/Windows/Fonts/msyh.ttc"),
        # 使用 Windows 宋体集合。
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    # 遍历存在的候选字体。
    for candidate in candidates:
        # 跳过空候选和不存在的文件。
        if candidate is None or not candidate.is_file():
            # 继续检查下一个候选。
            continue
        # 尝试以集合第一个子字体注册。
        try:
            # 注册统一论文 CJK 字体名。
            pdfmetrics.registerFont(TTFont("PaperCJK", str(candidate), subfontIndex=0))
            # 返回成功注册的字体名。
            return "PaperCJK"
        # 当前字体无法注册时继续下一个候选。
        except Exception:
            # 跳过不兼容字体。
            continue
    # 没有 CJK 字体时退回 Helvetica；论文来源收据会记录字体限制。
    return "Helvetica"


# 定义 PDF 页眉页脚绘制函数。
def _draw_page_number(canvas: Any, document: Any, font_name: str) -> None:
    # 保存当前绘图状态。
    canvas.saveState()
    # 使用小号论文正文字体。
    canvas.setFont(font_name, 8)
    # 在左下角写入短标题。
    canvas.drawString(document.leftMargin, 18, "DeepSeek - CalculiX contact diagnosis")
    # 在右下角写入页码。
    canvas.drawRightString(document.pagesize[0] - document.rightMargin, 18, f"Page {document.page}")
    # 恢复绘图状态。
    canvas.restoreState()


# 定义论文 PDF 生成函数，内容直接来自同一 trace 和 source audit。
def _render_pdf(path: Path, trace: dict[str, Any], source_audit: dict[str, Any]) -> dict[str, Any]:
    # 延迟导入 ReportLab 颜色模块。
    from reportlab.lib import colors
    # 延迟导入 A4 页面尺寸。
    from reportlab.lib.pagesizes import A4
    # 延迟导入样式和段落样式类型。
    from reportlab.lib.styles import ParagraphStyle
    # 延迟导入默认样式表。
    from reportlab.lib.styles import getSampleStyleSheet
    # 延迟导入毫米单位。
    from reportlab.lib.units import mm
    # 延迟导入文档、段落、分页、间距和表格组件。
    from reportlab.platypus import PageBreak
    # 导入段落组件。
    from reportlab.platypus import Paragraph
    # 导入可分页文档组件。
    from reportlab.platypus import SimpleDocTemplate
    # 导入垂直间距组件。
    from reportlab.platypus import Spacer
    # 导入表格组件。
    from reportlab.platypus import Table
    # 导入表格样式组件。
    from reportlab.platypus import TableStyle
    # 创建 PDF 父目录。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 注册可用的 CJK 字体。
    font_name = _register_pdf_font()
    # 获取 ReportLab 默认样式。
    styles = getSampleStyleSheet()
    # 定义主标题样式。
    title_style = ParagraphStyle("PaperTitle", parent=styles["Title"], fontName=font_name, fontSize=18, leading=24, textColor=colors.HexColor("#183B56"), spaceAfter=12)
    # 定义一级标题样式。
    heading_style = ParagraphStyle("PaperHeading", parent=styles["Heading1"], fontName=font_name, fontSize=13, leading=18, textColor=colors.HexColor("#0F5C78"), spaceBefore=10, spaceAfter=6)
    # 定义二级标题样式。
    subheading_style = ParagraphStyle("PaperSubheading", parent=styles["Heading2"], fontName=font_name, fontSize=10.5, leading=15, textColor=colors.HexColor("#2F6F89"), spaceBefore=7, spaceAfter=4)
    # 定义正文字体和行距。
    body_style = ParagraphStyle("PaperBody", parent=styles["BodyText"], fontName=font_name, fontSize=9.2, leading=14, textColor=colors.HexColor("#263238"), spaceAfter=6)
    # 定义小号来源文字。
    small_style = ParagraphStyle("PaperSmall", parent=body_style, fontSize=7.8, leading=11, textColor=colors.HexColor("#546E7A"))
    # 创建 A4 文档和稳定页边距。
    document = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=16 * mm, title="DeepSeek CalculiX Interference Contact Diagnosis", author="Decision-derived engineering experiment")
    # 初始化 PDF flowable 列表。
    story: list[Any] = []
    # 定义安全段落辅助函数。
    def add_paragraph(text_value: Any, style: Any = body_style) -> None:
        # 转义模型文本中的 XML 特殊字符并保留换行。
        safe_text = html.escape(str(text_value if text_value is not None else "")).replace("\n", "<br/>")
        # 追加安全段落。
        story.append(Paragraph(safe_text, style))
    # 写入论文标题。
    add_paragraph("从论坛原始输入到真实求解反馈：DeepSeek 对 CalculiX 过盈接触不收敛的有限调用诊断", title_style)
    # 写入来源副标题。
    add_paragraph("Decision-derived paper | 原帖附件审计 + C3D20R 代表模型 + 最多三次 DeepSeek 调用", small_style)
    # 写入摘要标题。
    add_paragraph("摘要", heading_style)
    # 写入摘要正文。
    add_paragraph("本文检验 DeepSeek 是否能在不训练 DQN、不做网格优化、不使用预设答案门禁的条件下，根据真实 CalculiX 反馈诊断一个论坛过盈接触问题。原始大型附件已取得并冻结哈希，但未在资源不足且后端不同的标准 runner 上冒充精确复现。系统使用二次减缩积分接触代表模型形成可执行反事实，通过一条增长消息历史让 DeepSeek提出竞争假设、选择单因素实验、读取求解反馈并形成有边界的答复。论文内容来自同一 agent trace，生成论文没有额外模型调用。")
    # 写入原始问题章节。
    add_paragraph("1. 原始问题与证据边界", heading_style)
    # 写入原帖和附件事实。
    add_paragraph(f"论坛问题：{source_audit.get('forum', {}).get('title')}。公开链接：{source_audit.get('forum', {}).get('url')}。归档 SHA-256：{source_audit.get('download', {}).get('archive_sha256')}。两个附件合计约一千一百六十万自由度，使用 C3D20R、surface-to-surface、线性 pressure-overclosure=69000 并请求 Pardiso。")
    # 写入原附件混杂。
    add_paragraph("原附件并非单因素对照：细模型同时改变 PIN2 接触面离散数量和 MODEL CHANGE 激活历史。论坛冻结时没有作者确认的最终答案。")
    # 写入方法章节。
    add_paragraph("2. 方法与有限调用协议", heading_style)
    # 写入方法概述。
    add_paragraph(f"应用级 Skill 只提供证据纪律，不提供答案。运行只创建一个 API client，max_retries=0，最多 {MAX_HTTP_REQUESTS} 次请求；每轮把模型 JSON 与工具结果追加到同一消息历史。模型可主动停止，工具失败和求解失败也进入 trace。")
    # 写入代表模型章节。
    add_paragraph("3. C3D20R 代表模型与真实求解", heading_style)
    # 写入代表模型定义。
    add_paragraph("模型来自 CalculiX 官方 contact4 的两个二次实体接触块，并使用 C3D20R。对照和目标只改变上块接触面四个边中间节点高度：z=1.00 对 z=0.95；角点保持 z=1.00。正式代表模型为获得健康对照使用 node-to-surface，因此结果只用于机制判别。")
    # 建立求解表头。
    table_data: list[list[Any]] = [["SOLVE-ID", "mid-z", "inc", "penalty", "完成", "no-conv", "尝试", "最大修正"]]
    # 遍历所有实际求解。
    for run in trace.get("solver_runs", []):
        # 读取配置。
        config = run.get("configuration", {})
        # 追加紧凑求解行。
        table_data.append([str(run.get("solve_id")), str(config.get("mid_z")), str(config.get("initial_increment")), "default" if config.get("penalty_stiffness") is None else f"{config.get('penalty_stiffness'):.3g}", str(run.get("completed")), str(run.get("no_convergence_count")), str(run.get("increment_attempt_count")), f"{run.get('max_absolute_displacement_correction'):.3g}" if isinstance(run.get("max_absolute_displacement_correction"), (int, float)) else "-"])
    # 创建按页面宽度分配的求解表格。
    solver_table = Table(table_data, colWidths=[32 * mm, 14 * mm, 13 * mm, 20 * mm, 14 * mm, 16 * mm, 13 * mm, 22 * mm], repeatRows=1)
    # 设置清晰表头、网格和小字号。
    solver_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EEF5")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#183B56")), ("FONTNAME", (0, 0), (-1, -1), font_name), ("FONTSIZE", (0, 0), (-1, -1), 6.5), ("LEADING", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#90A4AE")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")])]))
    # 追加求解表格。
    story.append(solver_table)
    # 追加表格后间距。
    story.append(Spacer(1, 5 * mm))
    # 强制后续决策链从新页开始，避免表格和小节拥挤。
    story.append(PageBreak())
    # 写入决策链章节。
    add_paragraph("4. DeepSeek 决策链", heading_style)
    # 遍历公开模型决策事件。
    for event in [item for item in trace.get("events", []) if item.get("type") == "model_decision"]:
        # 读取当前决策。
        decision = event.get("decision", {})
        # 写入轮次标题。
        add_paragraph(f"第 {event.get('iteration')} 轮", subheading_style)
        # 写入状态和证据判断。
        add_paragraph(f"状态：{decision.get('state')}。证据判断：{decision.get('evidence_assessment')}")
        # 读取可选动作。
        action = decision.get("next_action") if isinstance(decision.get("next_action"), dict) else {}
        # 写入动作及预先预测。
        add_paragraph(f"动作：{action.get('tool_id')}。正结果预测：{action.get('prediction_if_positive')}。负结果预测：{action.get('prediction_if_negative')}")
        # 遍历当前假设。
        for hypothesis in decision.get("hypotheses", []) if isinstance(decision.get("hypotheses"), list) else []:
            # 写入每个假设。
            add_paragraph(f"{hypothesis.get('id')} | confidence={hypothesis.get('confidence')} | {hypothesis.get('claim')}", small_style)
    # 写入最终答复章节。
    add_paragraph("5. DeepSeek 最终工程答复", heading_style)
    # 读取最终决策。
    final = trace.get("final_decision", {})
    # 写入最终状态。
    add_paragraph(f"最终状态：{final.get('state')}", subheading_style)
    # 写入最终原文。
    add_paragraph(final.get("answer") or "模型未形成非空最终答复。")
    # 写入已实施动作。
    add_paragraph("已实施动作", subheading_style)
    # 逐项写入已实施动作。
    for item in final.get("implemented_actions", []) or ["模型未记录已实施动作。"]:
        # 写入项目符号。
        add_paragraph(f"- {item}")
    # 写入可用范围。
    add_paragraph("可使用范围", subheading_style)
    # 逐项写入可用范围。
    for item in final.get("can_use", []) or ["模型未记录可使用范围。"]:
        # 写入项目符号。
        add_paragraph(f"- {item}")
    # 写入不可用范围。
    add_paragraph("不可使用范围", subheading_style)
    # 逐项写入不可用范围。
    for item in final.get("cannot_use", []) or ["模型未记录不可使用范围。"]:
        # 写入项目符号。
        add_paragraph(f"- {item}")
    # 写入限制章节。
    add_paragraph("6. 限制与来源审计", heading_style)
    # 写入不可外推边界。
    add_paragraph("代表模型不是原始圆柱孔几何，也没有使用原 deck 请求的 Pardiso。代表模型上的 penalty 或增量结果只能作为机制证据或候选 workaround。原模型必须再次验证位移、反力、接触压力和穿透后，才能称为确认修复。")
    # 写入调用和缓存统计。
    add_paragraph(f"DeepSeek 请求 {trace.get('http_requests_attempted')} 次；API client {trace.get('api_client_count')} 个；SDK retries={trace.get('sdk_max_retries')}；cache hit tokens={trace.get('cache_audit', {}).get('prompt_cache_hit_tokens')}；cache miss tokens={trace.get('cache_audit', {}).get('prompt_cache_miss_tokens')}；论文额外模型调用=0。", small_style)
    # 写入参考资料。
    add_paragraph("参考资料", heading_style)
    # 写入原帖链接。
    add_paragraph(f"1. {source_audit.get('forum', {}).get('url')}", small_style)
    # 写入官方 contact4 链接。
    add_paragraph("2. https://github.com/Dhondtguido/CalculiX/blob/master/test/contact4.inp", small_style)
    # 写入官方接触文档。
    add_paragraph("3. https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node307.html", small_style)
    # 构造页脚回调并生成 PDF。
    document.build(story, onFirstPage=lambda canvas, doc: _draw_page_number(canvas, doc, font_name), onLaterPages=lambda canvas, doc: _draw_page_number(canvas, doc, font_name))
    # 返回 PDF 渲染元数据。
    return {"path": str(path), "font_name": font_name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}


# 定义论文和来源收据最终生成函数。
def _finalize_outputs(output_dir: Path, trace: dict[str, Any], source_audit: dict[str, Any], skill_sha256: str) -> dict[str, Any]:
    # 定位最终 trace 文件。
    trace_path = output_dir / "agent_trace.json"
    # 先写入完整 trace。
    _write_json(trace_path, trace)
    # 计算固定 trace 哈希。
    trace_sha256 = _sha256_file(trace_path)
    # 生成决策来源 Markdown。
    markdown_text = _render_markdown(trace, source_audit, skill_sha256, trace_sha256)
    # 定位 Markdown 论文。
    markdown_path = output_dir / "PAPER.md"
    # 写入 Markdown 论文。
    markdown_path.write_text(markdown_text, encoding="utf-8", newline="\n")
    # 定位稳定 PDF 输出路径。
    pdf_path = output_dir / "output" / "pdf" / "deepseek_calculix_interference_diagnosis.pdf"
    # 初始化 PDF 渲染错误。
    pdf_error: str | None = None
    # 尝试生成 PDF。
    try:
        # 生成并记录 PDF 元数据。
        pdf_metadata = _render_pdf(pdf_path, trace, source_audit)
    # 捕获 PDF 依赖或字体错误，但保留 Markdown 和 trace。
    except Exception as exc:
        # 保存明确渲染错误。
        pdf_error = f"{type(exc).__name__}: {exc}"
        # 使用空 PDF 元数据。
        pdf_metadata = {}
    # 构造论文来源收据。
    provenance = {
        # 写入收据格式版本。
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        # 标明论文没有额外 DeepSeek 调用。
        "additional_deepseek_calls_for_paper": 0,
        # 保存 trace 路径和哈希。
        "agent_trace": {"path": str(trace_path), "sha256": trace_sha256},
        # 保存 source audit 路径和哈希。
        "source_audit": {"path": str(SOURCE_AUDIT_PATH), "sha256": _sha256_file(SOURCE_AUDIT_PATH)},
        # 保存应用 Skill 路径和哈希。
        "engineering_skill": {"path": str(SKILL_PATH), "sha256": skill_sha256},
        # 保存 Markdown 论文来源。
        "paper_markdown": {"path": str(markdown_path), "sha256": _sha256_file(markdown_path), "size_bytes": markdown_path.stat().st_size},
        # 保存 PDF 元数据或空对象。
        "paper_pdf": pdf_metadata,
        # 保存 PDF 渲染错误。
        "pdf_error": pdf_error,
        # 保存最终答复确实来自 trace final_decision。
        "decision_source": "agent_trace.json#/final_decision",
        # 保存实际模型决策数量。
        "model_decision_count": len([event for event in trace.get("events", []) if event.get("type") == "model_decision"]),
    }
    # 写入来源收据。
    _write_json(output_dir / "paper_provenance.json", provenance)
    # 返回来源收据。
    return provenance


# 定义主程序，顺序执行真实求解、有限 DeepSeek loop 和论文生成。
def main() -> int:
    # 创建命令行解析器。
    parser = argparse.ArgumentParser(description="Run a bounded DeepSeek + CalculiX contact diagnosis loop.")
    # 添加输出目录参数。
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/deepseek_calculix_interference"), help="Directory for trace, solver runs and paper artifacts.")
    # 添加 DeepSeek 模型参数，并优先读取 ds Environment。
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL), help="Explicit DeepSeek model name.")
    # 添加 ccx 可执行文件参数。
    parser.add_argument("--ccx", default=os.environ.get("CALCULIX_CCX"), help="Path or command name for CalculiX ccx.")
    # 添加离线证据模式，允许本地只验证真实求解和论文渲染而不付费调用。
    parser.add_argument("--offline-evidence-only", action="store_true", help="Run solver evidence and render an unresolved paper without calling DeepSeek.")
    # 解析命令行参数。
    args = parser.parse_args()
    # 解析绝对输出目录。
    output_dir = args.output_dir.resolve()
    # 创建输出目录。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 读取冻结 source audit。
    source_audit = _read_json(SOURCE_AUDIT_PATH)
    # 读取应用级工程 Skill。
    skill = _read_json(SKILL_PATH)
    # 计算 Skill 哈希。
    skill_sha256 = _sha256_file(SKILL_PATH)
    # 依次查找命令行、通用命令、Ubuntu 版本化命令和本机长期化 CalculiX。
    ccx_executable = args.ccx or shutil.which("ccx") or shutil.which("ccx_2.22") or shutil.which("ccx_2.21") or shutil.which("ccx_2.20") or shutil.which("ccx_2.19") or (str(Path("C:/Users/asus/.local/bin/ccx.exe")) if Path("C:/Users/asus/.local/bin/ccx.exe").is_file() else None)
    # 无求解器时写入明确诊断并停止付费调用。
    if not ccx_executable:
        # 构造无求解器 trace。
        trace = {"schema_version": TRACE_SCHEMA_VERSION, "status": "solver_unavailable", "started_at": datetime.now(timezone.utc).isoformat(), "http_requests_attempted": 0, "api_client_count": 0, "sdk_max_retries": 0, "events": [], "solver_runs": [], "final_decision": {"state": "narrowed_unresolved", "answer": "当前运行环境没有找到 ccx，因此没有执行求解，也没有调用 DeepSeek。", "implemented_actions": [], "can_use": [], "cannot_use": ["不能据此诊断原问题。"], "evidence_refs": []}, "cache_audit": {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}}
        # 生成仍可读的失败论文。
        _finalize_outputs(output_dir, trace, source_audit, skill_sha256)
        # 返回求解器不可用状态。
        return 2
    # 探测求解器版本和二进制来源。
    solver_version = _solver_version(ccx_executable)
    # 初始化同运行求解缓存。
    solver_cache: dict[str, dict[str, Any]] = {}
    # 在第一次模型调用前执行真实初始对照。
    initial_evidence = _run_initial_evidence(ccx_executable, output_dir, solver_cache)
    # 构造问题清单。
    manifest = _build_problem_manifest(source_audit, solver_version, initial_evidence, args.model)
    # 写入独立问题清单 artifact。
    _write_json(output_dir / "problem_manifest.json", manifest)
    # 初始化完整 trace。
    trace: dict[str, Any] = {
        # 写入 trace 版本。
        "schema_version": TRACE_SCHEMA_VERSION,
        # 标记运行中状态。
        "status": "running",
        # 写入 UTC 开始时间。
        "started_at": datetime.now(timezone.utc).isoformat(),
        # 保存原帖 source audit 哈希。
        "source_audit_sha256": _sha256_file(SOURCE_AUDIT_PATH),
        # 保存 Skill 哈希。
        "engineering_skill_sha256": skill_sha256,
        # 保存求解器环境。
        "solver_environment": solver_version,
        # 保存初始求解证据。
        "initial_solver_evidence": initial_evidence,
        # 初始化事件列表。
        "events": [],
        # 初始化 HTTP 请求计数。
        "http_requests_attempted": 0,
        # 离线前客户端数量为零。
        "api_client_count": 0,
        # 明确 SDK 自动重试为零。
        "sdk_max_retries": 0,
        # 保存最多请求数。
        "max_http_requests": MAX_HTTP_REQUESTS,
        # 保存模型名称。
        "requested_model": args.model,
    }
    # 离线模式不调用 DeepSeek，并生成明确未解决论文。
    if args.offline_evidence_only:
        # 标记离线证据完成。
        trace["status"] = "offline_evidence_complete"
        # 写入不冒充模型结论的占位最终决策。
        trace["final_decision"] = {"state": "narrowed_unresolved", "answer": "离线证据模式只执行了 CalculiX 初始对照，没有调用 DeepSeek，因此没有模型诊断结论。", "hypotheses": [], "evidence_assessment": "仅确认代表模型能形成成功/失败对照。", "implemented_actions": ["执行 C3D20R 对齐与中间节点初始过闭合对照。"], "can_use": ["检查求解环境和论文渲染链。"], "cannot_use": ["不能作为 DeepSeek 决策实验结果。", "不能声称解决原帖。"], "evidence_refs": [initial_evidence["control"]["solve_id"], initial_evidence["target"]["solve_id"]]}
        # 保存当前所有唯一求解摘要。
        trace["solver_runs"] = list(solver_cache.values())
        # 写入零缓存统计。
        trace["cache_audit"] = {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0, "hit_ratio": None}
        # 写入结束时间。
        trace["finished_at"] = datetime.now(timezone.utc).isoformat()
        # 生成离线 trace、Markdown 和 PDF。
        provenance = _finalize_outputs(output_dir, trace, source_audit, skill_sha256)
        # 输出不含敏感信息的简短摘要。
        print(json.dumps({"status": trace["status"], "solver_runs": len(trace["solver_runs"]), "http_requests_attempted": 0, "pdf_error": provenance.get("pdf_error")}, ensure_ascii=False))
        # 返回成功的离线准备状态。
        return 0
    # 读取 ds Environment 中的 DeepSeek API key。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    # 凭据缺失时生成论文并避免无证据的付费流程。
    if not api_key:
        # 标记凭据缺失。
        trace["status"] = "credential_unavailable"
        # 写入明确未调用模型的最终记录。
        trace["final_decision"] = {"state": "narrowed_unresolved", "answer": "ds Environment 没有向本进程提供 DEEPSEEK_API_KEY；真实 CalculiX 初始证据已保存，但没有调用模型。", "hypotheses": [], "evidence_assessment": "DeepSeek 未运行。", "implemented_actions": ["执行初始 CalculiX 对照。"], "can_use": ["检查求解器初始证据。"], "cannot_use": ["不能作为 DeepSeek 诊断结果。"], "evidence_refs": [initial_evidence["control"]["solve_id"], initial_evidence["target"]["solve_id"]]}
        # 保存唯一求解摘要。
        trace["solver_runs"] = list(solver_cache.values())
        # 写入零缓存统计。
        trace["cache_audit"] = {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0, "hit_ratio": None}
        # 写入结束时间。
        trace["finished_at"] = datetime.now(timezone.utc).isoformat()
        # 生成可审计论文。
        _finalize_outputs(output_dir, trace, source_audit, skill_sha256)
        # 返回凭据不可用状态。
        return 3
    # 延迟导入 OpenAI 兼容客户端，确保离线模式不要求 API 包。
    from openai import OpenAI
    # 创建整个运行唯一一个 DeepSeek client，并关闭 SDK 隐式重试。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", max_retries=0, timeout=API_TIMEOUT_SECONDS)
    # 记录唯一客户端数量。
    trace["api_client_count"] = 1
    # 构造所有轮次共享的稳定 system prompt。
    system_prompt = _system_prompt(skill)
    # 构造初始请求包。
    initial_packet = _initial_packet(source_audit, manifest, solver_version, initial_evidence)
    # 初始化一条不断增长的消息历史。
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": _canonical_json(initial_packet)}]
    # 初始化最后一个有效模型决策。
    latest_decision: dict[str, Any] = {}
    # 初始化传输错误。
    transport_error: str | None = None
    # 最多执行三轮模型调用。
    for iteration in range(1, MAX_HTTP_REQUESTS + 1):
        # 调用前增加实际 HTTP 请求计数。
        trace["http_requests_attempted"] += 1
        # 尝试发起本轮唯一请求。
        try:
            # 执行真实模型调用。
            model_result = _call_model(client, args.model, messages)
        # 捕获传输或 SDK 错误，禁止自动重试。
        except Exception as exc:
            # 保存错误类型和公开文本。
            transport_error = f"{type(exc).__name__}: {exc}"
            # 追加传输错误事件。
            trace["events"].append({"type": "model_transport_error", "iteration": iteration, "error": transport_error})
            # 立即冻结当前 trace。
            _write_json(output_dir / "agent_trace.partial.json", trace)
            # 结束模型循环。
            break
        # 把模型原始公开回复追加到同一历史。
        messages.append({"role": "assistant", "content": model_result["raw_text"]})
        # 追加可审计模型事件。
        trace["events"].append({"type": "model_decision", "iteration": iteration, "decision": model_result["decision"], "raw_response": model_result["raw_text"], "parse_error": model_result["parse_error"], "provider_metadata": {"provider": "deepseek", **model_result["metadata"]}})
        # 每轮后冻结当前 trace，防止后续失败丢失已付费回答。
        _write_json(output_dir / "agent_trace.partial.json", trace)
        # 非法 JSON 不发修复请求。
        if model_result["parse_error"]:
            # 记录运行状态。
            trace["status"] = "invalid_model_json"
            # 结束模型循环。
            break
        # 保存当前有效决策。
        latest_decision = model_result["decision"]
        # 提取模型所选动作。
        action = _selected_action(latest_decision)
        # 模型主动停止或选择 finish 时直接结束。
        if bool(latest_decision.get("stop")) or (action is not None and action.get("tool_id") == "finish"):
            # 标记模型主动完成。
            trace["status"] = "finished_by_model"
            # 结束模型循环。
            break
        # 第三轮是硬预算末轮，不再执行没有机会反馈的新工具。
        if iteration == MAX_HTTP_REQUESTS:
            # 标记预算用尽。
            trace["status"] = "model_budget_exhausted"
            # 结束模型循环。
            break
        # 没有可执行动作时记录错误并在预算内结束，不代替模型选择工具。
        if action is None:
            # 追加缺少动作事件。
            trace["events"].append({"type": "model_action_error", "iteration": iteration, "error": "next_action is missing or not an object"})
            # 标记动作错误。
            trace["status"] = "missing_model_action"
            # 结束模型循环。
            break
        # 尝试执行模型实际选择的唯一工具。
        try:
            # 执行模型选择，不替换为脚本偏好的动作。
            tool_result = _execute_tool(action, source_audit, ccx_executable, output_dir, solver_cache)
        # 工具异常进入下一轮证据，不阻断 trace 和论文。
        except Exception as exc:
            # 构造可审计工具执行错误。
            tool_result = _tool_result(str(action.get("tool_id") or "missing_tool_id"), action.get("arguments") if isinstance(action.get("arguments"), dict) else {}, [], {"status": "tool_execution_error", "error": f"{type(exc).__name__}: {exc}"}, ["工具异常已冻结；没有自动改选其它工具，也没有自动重试。"])
        # 追加工具事件。
        trace["events"].append({"type": "tool_call", "iteration": iteration, "request": action, "result": tool_result})
        # 计算下一轮编号。
        next_iteration = iteration + 1
        # 判断下一轮是否硬预算末轮。
        final_round = next_iteration == MAX_HTTP_REQUESTS
        # 构造包含真实工具反馈的下一轮包。
        next_packet = _next_round_packet(next_iteration, tool_result, MAX_HTTP_REQUESTS - iteration, final_round)
        # 把工具结果和下一轮指令作为新 user 消息追加，而不重建会话。
        messages.append({"role": "user", "content": _canonical_json(next_packet)})
    # 没有显式状态时按是否有有效决策记录结束状态。
    if trace.get("status") == "running":
        # 有有效决策时标记循环结束，否则标记无模型决策。
        trace["status"] = "loop_complete" if latest_decision else "no_valid_model_decision"
    # 从最后一个有效模型 JSON 映射论文最终决策。
    trace["final_decision"] = _final_decision(latest_decision)
    # 完全没有有效模型回答时写入明确传输/格式失败说明，而不伪造 DeepSeek 结论。
    if not latest_decision:
        # 写入非模型诊断记录。
        trace["final_decision"] = {"state": "narrowed_unresolved", "answer": f"本次运行没有获得可解析的 DeepSeek 决策。错误：{transport_error or trace.get('status')}。真实 CalculiX 初始证据已保存。", "hypotheses": [], "evidence_assessment": "没有有效模型输出。", "implemented_actions": ["执行初始 CalculiX 对照。"], "can_use": ["审计求解和 API 失败。"], "cannot_use": ["不能作为 DeepSeek 工程结论。"], "evidence_refs": [initial_evidence["control"]["solve_id"], initial_evidence["target"]["solve_id"]]}
    # 保存所有唯一求解摘要。
    trace["solver_runs"] = list(solver_cache.values())
    # 保存最终公共消息历史，确认没有新建分支会话。
    trace["messages"] = messages
    # 汇总所有模型调用 usage。
    usage_rows = [event.get("provider_metadata", {}).get("usage_summary", {}) for event in trace["events"] if event.get("type") == "model_decision"]
    # 计算缓存命中 token。
    cache_hit_tokens = sum(int(row.get("prompt_cache_hit_tokens", 0)) for row in usage_rows)
    # 计算缓存未命中 token。
    cache_miss_tokens = sum(int(row.get("prompt_cache_miss_tokens", 0)) for row in usage_rows)
    # 计算有效缓存分母。
    cache_denominator = cache_hit_tokens + cache_miss_tokens
    # 保存缓存审计。
    trace["cache_audit"] = {"prompt_cache_hit_tokens": cache_hit_tokens, "prompt_cache_miss_tokens": cache_miss_tokens, "hit_ratio": (cache_hit_tokens / cache_denominator) if cache_denominator else None}
    # 写入结束时间。
    trace["finished_at"] = datetime.now(timezone.utc).isoformat()
    # 生成完整 trace、Markdown、PDF 和来源收据。
    provenance = _finalize_outputs(output_dir, trace, source_audit, skill_sha256)
    # 输出不含凭据和隐藏推理的简短运行摘要。
    print(json.dumps({"status": trace["status"], "http_requests_attempted": trace["http_requests_attempted"], "solver_runs": len(trace["solver_runs"]), "final_state": trace["final_decision"].get("state"), "cache_audit": trace["cache_audit"], "pdf_error": provenance.get("pdf_error")}, ensure_ascii=False))
    # 传输错误时返回非零以便 workflow 在上传后反映真实状态。
    if transport_error:
        # 返回 API 传输失败状态。
        return 4
    # 其他工程结论状态不作为进程门禁。
    return 0


# 仅在直接执行脚本时进入主程序。
if __name__ == "__main__":
    # 把主程序状态原样返回操作系统。
    raise SystemExit(main())
