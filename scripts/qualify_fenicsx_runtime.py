#!/usr/bin/env python3  # 使用官方 FEniCSx 容器验证 DOLFINx 运行时身份和可导入性。
"""记录 FEniCSx/DOLFINx 运行时资格证据，但不创建网格、不组装弱式也不求解。"""  # 限定本脚本只做环境资格测试。
from __future__ import annotations  # 启用现代类型注解行为，兼容当前官方容器 Python。

import argparse  # 解析工作流显式传入的 DOLFINx 版本和 receipt 路径。
import importlib  # 在可捕获异常的阶段动态导入 FEniCSx 运行时组件。
import json  # 将环境版本、检查和零执行边界写成机器可读 JSON。
import os  # 读取 GitHub Actions 与固定容器镜像的环境元数据。
import platform  # 记录 Python、操作系统和硬件架构信息。
import sys  # 返回真实成功或失败退出状态，并记录 Python 版本。
from datetime import datetime, timezone  # 记录资格测试开始和结束的 UTC 时间。
from pathlib import Path  # 用跨平台路径对象创建 artifact receipt。

REQUIRED_MODULE_NAMES = (  # 冻结官方 FEniCSx Python 运行时必须能够导入的组件。
    "dolfinx",  # DOLFINx 是 FEniCSx 的有限元问题求解环境。
    "dolfinx.common",  # DOLFINx common 提供官方构建提交哈希接口。
    "dolfinx.fem",  # DOLFINx fem 是后续函数空间与变分离散入口，本测试只导入。
    "dolfinx.mesh",  # DOLFINx mesh 是后续网格入口，本测试禁止创建网格。
    "dolfinx.io",  # DOLFINx io 是后续外部网格读取入口，本测试禁止读取案例文件。
    "basix",  # Basix 提供有限元基函数定义。
    "ufl",  # UFL 提供变分形式语言。
    "ffcx",  # FFCx 提供 FEniCSx 形式编译器，本测试只验证可导入性。
    "mpi4py",  # mpi4py 提供 Python MPI 运行时接口。
    "mpi4py.MPI",  # mpi4py.MPI 提供实际通信器对象。
    "petsc4py",  # petsc4py 提供 Python PETSc 包元数据。
    "petsc4py.PETSc",  # petsc4py.PETSc 提供实际线性代数和求解运行时。
)  # 结束必需模块名称元组。


def utc_now() -> str:  # 返回带时区的 ISO 8601 UTC 时间文本。
    return datetime.now(timezone.utc).isoformat()  # 使用 Actions runner 真实系统时钟生成证据时间。


def write_json(path: Path, payload: dict[str, object]) -> None:  # 将资格 receipt 以稳定 UTF-8 JSON 写入 artifact。
    path.parent.mkdir(parents=True, exist_ok=True)  # 在成功和失败路径都创建 receipt 父目录。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保留中文并使用两空格缩进。


def module_version(module: object) -> str:  # 从已导入模块读取公开版本并返回文本。
    return str(getattr(module, "__version__", "")).strip()  # 缺少公开版本时返回空字符串并由检查明确失败。


def add_check(checks: list[dict[str, object]], name: str, passed: bool, details: object) -> None:  # 追加一个命名运行时检查和可审计细节。
    checks.append({"name": name, "passed": bool(passed), "details": details})  # 使用统一结构供 Actions 和前端读取。


def failure_receipt(args: argparse.Namespace, started_at: str, error: BaseException) -> dict[str, object]:  # 在模块导入或元数据读取失败时构造可上传的 receipt。
    return {  # 返回不冒充 FEniCSx 可用的失败对象。
        "schema_version": "fenicsx-runtime-qualification/1.0",  # 冻结环境资格 receipt 合同版本。
        "status": "failure",  # 明确运行时资格测试失败。
        "execution_family": "FEniCS/FEniCSx",  # 保持用户确认的 FEN 求解器族映射。
        "expected_dolfinx_version_prefix": args.expected_dolfinx_version_prefix,  # 记录工作流要求的显式版本。
        "runtime_image": os.environ.get("FENICSX_RUNTIME_IMAGE", ""),  # 记录工作流固定的官方容器镜像引用。
        "started_at_utc": started_at,  # 记录资格测试开始时间。
        "finished_at_utc": utc_now(),  # 记录失败 receipt 完成时间。
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),  # 记录 GitHub 仓库标识。
        "ref": os.environ.get("GITHUB_REF", ""),  # 记录 GitHub 分支完整引用。
        "head_sha": os.environ.get("GITHUB_SHA", ""),  # 记录不可变提交 SHA。
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),  # 记录 Actions 运行编号。
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),  # 记录 Actions 重跑编号。
        "error": type(error).__name__ + ": " + str(error),  # 保存真实异常类型和文本。
        "checks": [],  # 导入失败时不存在可宣称通过的组件检查。
        "execution_counts": {  # 明确环境失败也没有执行工程计算。
            "mesh_generation_calls": 0,  # 没有创建 DOLFINx Mesh。
            "form_assembly_calls": 0,  # 没有组装 UFL 变分形式。
            "linear_or_nonlinear_solve_calls": 0,  # 没有调用 PETSc 求解。
            "engineering_result_extractions": 0,  # 没有提取任何工程 QoI。
        },  # 结束零执行计数对象。
        "engineering_results_generated": False,  # 明确没有生成物理或工程结果。
    }  # 结束失败 receipt。


def parse_args() -> argparse.Namespace:  # 定义工作流必须显式提供的版本和输出路径参数。
    parser = argparse.ArgumentParser(description="Qualify the pinned FEniCSx/DOLFINx runtime without solving a case.")  # 创建只描述环境资格用途的解析器。
    parser.add_argument("--expected-dolfinx-version-prefix", required=True, help="DOLFINx version prefix required by the pinned official image.")  # 要求显式传入预期 DOLFINx 版本前缀。
    parser.add_argument("--receipt", required=True, help="Path for the always-written runtime qualification receipt.")  # 要求显式传入 artifact receipt 路径。
    return parser.parse_args()  # 解析调用方提供的参数。


def main() -> int:  # 动态导入官方运行时、记录版本、验证映射并返回真实状态。
    args = parse_args()  # 读取 Actions 明确冻结的 DOLFINx 版本和 receipt 路径。
    receipt_path = Path(args.receipt).resolve()  # 将输出位置解析为容器内挂载仓库的绝对路径。
    started_at = utc_now()  # 记录资格测试真实开始时间。
    try:  # 捕获任一 FEniCSx 组件导入失败并始终留下 receipt。
        modules = {name: importlib.import_module(name) for name in REQUIRED_MODULE_NAMES}  # 导入 DOLFINx、Basix、UFL、MPI 和 PETSc 组件。
        dolfinx_module = modules["dolfinx"]  # 取得 DOLFINx 顶层模块供版本核对。
        mpi_module = modules["mpi4py.MPI"]  # 取得 MPI 子模块供通信器检查。
        petsc_module = modules["petsc4py.PETSc"]  # 取得 PETSc 子模块供运行时版本检查。
        dolfinx_common_module = modules["dolfinx.common"]  # 取得 DOLFINx common 模块供构建提交追溯。
        dolfinx_git_commit = str(dolfinx_common_module.git_commit_hash()).strip()  # 读取官方镜像内嵌的 DOLFINx 构建提交哈希。
        versions = {  # 收集全部实际导入组件的公开版本。
            "dolfinx": module_version(dolfinx_module),  # 记录核心 DOLFINx 版本。
            "basix": module_version(modules["basix"]),  # 记录有限元基函数库版本。
            "ufl": module_version(modules["ufl"]),  # 记录变分形式语言版本。
            "ffcx": module_version(modules["ffcx"]),  # 记录 FEniCSx 形式编译器版本。
            "mpi4py": module_version(modules["mpi4py"]),  # 记录 Python MPI 包版本。
            "petsc4py": module_version(modules["petsc4py"]),  # 记录 Python PETSc 包版本。
            "petsc": ".".join(str(item) for item in petsc_module.Sys.getVersion()),  # 记录 PETSc C 运行时主次补丁版本。
        }  # 结束运行时版本对象。
        mpi_size = int(mpi_module.COMM_WORLD.Get_size())  # 读取当前 Actions 容器通信器进程数。
        mpi_rank = int(mpi_module.COMM_WORLD.Get_rank())  # 读取当前进程在通信器中的秩。
    except BaseException as error:  # 捕获导入、动态库或运行时元数据异常。
        receipt = failure_receipt(args, started_at, error)  # 将真实异常转换为可上传失败证据。
        write_json(receipt_path, receipt)  # 在退出前始终写出失败 receipt。
        print(json.dumps({"status": "failure", "receipt": str(receipt_path), "error": receipt["error"]}, ensure_ascii=False))  # 在 Actions 日志输出精简失败位置。
        return 1  # 使 GitHub Actions 正确标记环境资格失败。
    checks: list[dict[str, object]] = []  # 创建成功导入后的命名检查列表。
    add_check(checks, "dolfinx_version_prefix", versions["dolfinx"].startswith(args.expected_dolfinx_version_prefix), {"expected_prefix": args.expected_dolfinx_version_prefix, "actual": versions["dolfinx"]})  # 固定镜像摘要并要求版本属于明确的零点十一版本线。
    named_versions = {name: versions[name] for name in ("dolfinx", "basix", "ufl", "ffcx", "mpi4py", "petsc4py", "petsc")}  # 选择必须非空的七项版本文本。
    add_check(checks, "required_versions_present", all(bool(value) for value in named_versions.values()), named_versions)  # 防止组件可导入但版本身份不可审计。
    add_check(checks, "mpi_runtime_available", mpi_size >= 1 and 0 <= mpi_rank < mpi_size, {"size": mpi_size, "rank": mpi_rank})  # 要求存在合法 MPI 通信器。
    runtime_image = os.environ.get("FENICSX_RUNTIME_IMAGE", "")  # 读取 workflow 传入的完整镜像标签和摘要。
    add_check(checks, "pinned_official_image_recorded", runtime_image.startswith("ghcr.io/fenics/dolfinx/dolfinx:v0.11.0@sha256:"), {"runtime_image": runtime_image})  # 要求 receipt 保留固定的官方 FEniCSx GHCR 镜像身份。
    valid = bool(checks) and all(bool(row["passed"]) for row in checks)  # 仅当全部版本和运行时检查通过时标记资格成功。
    receipt = {  # 构造成功导入后的机器可读环境资格 receipt。
        "schema_version": "fenicsx-runtime-qualification/1.0",  # 冻结环境资格机器合同版本。
        "status": "success" if valid else "failure",  # 根据全部命名检查记录真实状态。
        "execution_family": "FEniCS/FEniCSx",  # 明确本环境只服务 FEN-003 和 FEN-014。
        "expected_dolfinx_version_prefix": args.expected_dolfinx_version_prefix,  # 记录工作流显式要求的 DOLFINx 版本。
        "runtime_image": runtime_image,  # 记录官方容器标签和不可变摘要。
        "versions": versions,  # 保存实际导入的 FEniCSx 组件版本。
        "dolfinx_git_commit": dolfinx_git_commit,  # 保存官方镜像内嵌的 DOLFINx 构建提交哈希。
        "python": {  # 保存容器 Python 和平台身份。
            "version": sys.version,  # 记录完整 Python 构建版本文本。
            "implementation": platform.python_implementation(),  # 记录 CPython 或其他实现名称。
            "platform": platform.platform(),  # 记录容器操作系统与架构文本。
        },  # 结束 Python 平台对象。
        "mpi": {"size": mpi_size, "rank": mpi_rank},  # 保存当前单进程或多进程通信器状态。
        "checks": checks,  # 保存版本、组件、MPI 和镜像检查。
        "started_at_utc": started_at,  # 记录资格测试开始时间。
        "finished_at_utc": utc_now(),  # 记录 receipt 写出前的完成时间。
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),  # 记录 GitHub 仓库标识。
        "ref": os.environ.get("GITHUB_REF", ""),  # 记录权威分支完整引用。
        "head_sha": os.environ.get("GITHUB_SHA", ""),  # 记录本次验证的提交 SHA。
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),  # 记录 Actions 运行编号。
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),  # 记录 Actions 重跑编号。
        "execution_counts": {  # 明确本资格测试没有执行案例物理计算。
            "mesh_generation_calls": 0,  # 没有创建 DOLFINx Mesh。
            "form_assembly_calls": 0,  # 没有组装任何 UFL 形式。
            "linear_or_nonlinear_solve_calls": 0,  # 没有调用 PETSc 线性或非线性求解。
            "engineering_result_extractions": 0,  # 没有提取工程 QoI 或场结果。
        },  # 结束零执行计数对象。
        "engineering_results_generated": False,  # 环境可用不能解释为案例已经求解。
        "allowed_use": "证明当前提交能在固定官方 FEniCSx 容器中导入 DOLFINx 运行时。",  # 限定成功 receipt 的允许用途。
        "disallowed_use": "不能证明 FEN-003 或 FEN-014 的模型、网格、弱式、求解或工程结论成立。",  # 防止环境成功被误读为案例成功。
    }  # 结束环境资格 receipt。
    write_json(receipt_path, receipt)  # 在返回状态前写出完整机器证据。
    print(json.dumps({"status": receipt["status"], "dolfinx": versions["dolfinx"], "receipt": str(receipt_path)}, ensure_ascii=False))  # 在 Actions 日志输出精简成功状态。
    return 0 if valid else 1  # 全部环境检查通过时返回零，否则使 workflow 失败。


if __name__ == "__main__":  # 仅在脚本由 Actions 直接执行时运行资格入口。
    raise SystemExit(main())  # 将真实环境资格状态传递给 GitHub Actions。
