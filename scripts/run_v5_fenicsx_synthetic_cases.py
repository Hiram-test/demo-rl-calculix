#!/usr/bin/env python3
# 上一行选择当前环境的 Python 3 解释器；本脚本在固定官方 DOLFINx 容器中运行两个真实数值合同测试。
"""运行 FEN-003 网格收敛与 FEN-014 XDMF 标签往返合成基准，并写出可审计机器回执。"""  # 明确本脚本生成合成数值证据而非原研究案例结论。
from __future__ import annotations  # 启用现代类型注解语义，兼容官方 DOLFINx 0.11 Python 环境。

import argparse  # 解析工作流显式传入的 artifact 输出目录。
import hashlib  # 计算脚本、XDMF 和 HDF5 文件的 SHA-256 来源哈希。
import json  # 写出稳定且机器可读的案例与活动回执。
import math  # 提供圆周率、平方根和观测收敛阶计算。
import os  # 读取 GitHub Actions 与固定容器镜像元数据。
import platform  # 记录 Python 容器平台身份供复现。
import sys  # 返回真实测试退出码并记录 Python 版本。
import traceback  # 在异常路径保留完整 Python 调用栈。
from datetime import datetime, timezone  # 记录带时区的 UTC 测试起止时间。
from pathlib import Path  # 使用跨平台路径创建案例 artifact。
from typing import Any  # 为嵌套 JSON 回执提供通用值类型。

from mpi4py import MPI  # 必须先初始化 MPI，再导入依赖 MPI 的 DOLFINx 组件。
from petsc4py import PETSc  # 提供 PETSc 标量类型、求解器状态和运行时版本。

import basix  # 记录 FEniCSx 有限元基函数库的实际版本。
import ffcx  # 记录实际编译 UFL 形式的 FFCx 版本。
import numpy as np  # 构造标签数组、执行有限性检查并计算节点误差。
import ufl  # 定义 Poisson 与 Laplace 变分形式和解析解。
from dolfinx import __version__ as dolfinx_version  # 记录当前真实 DOLFINx Python 版本。
from dolfinx import fem, io, mesh  # 创建网格、函数空间、形式、标签与 XDMF 输入输出。
from dolfinx.fem.petsc import LinearProblem  # 组装并真实调用 PETSc 求解线性有限元问题。

COMM = MPI.COMM_WORLD  # 使用当前容器的全局 MPI 通信器汇总所有积分与计数。
ROOT_RANK = 0  # 仅让零号进程写 JSON 和打印活动摘要，避免并行文件竞争。
FEN003_LEVELS = (8, 16, 32, 64)  # 使用四个逐次二分尺寸层级验证一阶单元的理论收敛阶。
FEN014_DIVISIONS = 8  # 使用八乘八方格生成八十一节点和一百二十八三角形的轻量往返网格。
QUADRATURE_DEGREE = 8  # 用八阶积分降低制造解正弦函数的数值积分误差对收敛判断的影响。
FEN003_FINE_L2_LIMIT = 1.0e-3  # 要求最细层 L2 误差不超过千分之一的预冻结基准门。
FEN003_FINE_H1_LIMIT = 1.0e-1  # 要求最细层 H1 半范误差不超过零点一的预冻结基准门。
FEN003_FINE_QOI_RELATIVE_LIMIT = 2.0e-3  # 要求最细层全域积分 QoI 相对误差不超过千分之二。
FEN003_ENERGY_BALANCE_LIMIT = 1.0e-10  # 要求离散能量与载荷功相对不平衡不超过一乘十负十次方。
FEN014_EXACTNESS_LIMIT = 1.0e-10  # 线性补丁解与参考或重读结果必须在一乘十负十次方内一致。
FEN014_MEASURE_LIMIT = 1.0e-12  # 面积和边长的绝对误差必须在一乘十负十二次方内。
FEN014_MESH_NAME = "fen014_mesh"  # 固定 XDMF 网格名称以验证按名称读取而非依赖默认顺序。
FEN014_CELL_TAG_NAME = "fen014_cell_tags"  # 固定左右子域 cell MeshTags 名称供写出和重读。
FEN014_FACET_TAG_NAME = "fen014_facet_tags"  # 固定四边 facet MeshTags 名称供边界映射。
FEN014_CELL_IDS = (1, 2)  # 用一和二分别标记单位方形左半区与右半区。
FEN014_FACET_IDS = (11, 12, 13, 14)  # 用十一至十四分别标记左、右、下、上四条边。


def utc_now() -> str:  # 返回带显式 UTC 时区的 ISO 8601 时间文本。
    return datetime.now(timezone.utc).isoformat()  # 使用 runner 真实系统时钟生成可追溯时间。


def parse_args() -> argparse.Namespace:  # 定义数值测试唯一允许的命令行输入。
    parser = argparse.ArgumentParser(description="Run two synthetic DOLFINx numerical contract tests.")  # 创建不暗示原研究案例已执行的解析器。
    parser.add_argument("--output-dir", required=True, help="Artifact directory for receipts, meshes, fields, and logs.")  # 要求工作流显式指定输出目录。
    return parser.parse_args()  # 返回完成校验的命令行参数。


def write_json(path: Path, payload: dict[str, Any]) -> None:  # 由零号进程将回执稳定写为 UTF-8 JSON。
    if COMM.rank != ROOT_RANK:  # 非零进程不参与共享 JSON 文件写入。
        return  # 直接返回以避免多个 MPI rank 覆盖同一文件。
    path.parent.mkdir(parents=True, exist_ok=True)  # 在成功和失败路径都创建目标父目录。
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"  # 固定排序并拒绝 NaN 或 Infinity，保证严格 JSON 有效。
    path.write_text(serialized, encoding="utf-8")  # 使用 UTF-8 保留中文边界说明且不依赖系统编码。


def sha256_file(path: Path) -> str:  # 计算任意证据文件的完整 SHA-256 十六进制摘要。
    digest = hashlib.sha256()  # 创建新的 SHA-256 状态避免跨文件混合。
    with path.open("rb") as source:  # 以二进制只读方式打开目标文件保持字节不变。
        for chunk in iter(lambda: source.read(1024 * 1024), b""):  # 按一 MiB 数据块读取以限制内存占用。
            digest.update(chunk)  # 将当前数据块追加到摘要状态。
    return digest.hexdigest()  # 返回六十四字符摘要供 receipt 保存。


def file_record(path: Path) -> dict[str, Any]:  # 记录证据文件的存在性、字节大小和完整内容摘要。
    return {"path": str(path), "exists": path.is_file(), "size_bytes": int(path.stat().st_size) if path.is_file() else 0, "sha256": sha256_file(path) if path.is_file() else ""}  # 对不存在文件返回零大小和空摘要以供显式失败。


def artifact_file_record(path: Path, output_root: Path) -> dict[str, Any]:  # 记录供独立验证器复核的 artifact 相对路径、正字节数与 SHA-256。
    return {"path": str(path.relative_to(output_root)), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}  # 文件缺失或越界时直接抛错并由案例失败回执留证。


def source_identity() -> dict[str, Any]:  # 记录当前 runner 与 GitHub Actions 调用来源。
    script_path = Path(__file__).resolve()  # 解析正在实际执行的脚本绝对路径。
    return {  # 返回不会把合成 fixture 误标为用户输入的来源对象。
        "runner_path": "scripts/run_v5_fenicsx_synthetic_cases.py",  # 保存仓库相对入口路径。
        "runner_sha256": sha256_file(script_path),  # 保存实际执行脚本内容摘要。
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),  # 保存 GitHub 仓库标识。
        "ref": os.environ.get("GITHUB_REF", ""),  # 保存权威分支完整引用。
        "head_sha": os.environ.get("GITHUB_SHA", ""),  # 保存不可变提交 SHA。
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),  # 保存 GitHub Actions 运行编号。
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),  # 保存工作流重跑编号。
        "input_provenance": "generated_in_ci",  # 明确全部几何、网格、标签和载荷由本测试生成。
        "uses_original_research_inputs": False,  # 明确没有使用 FEN-003 或 FEN-014 原始研究输入。
    }  # 结束来源对象。


def runtime_identity() -> dict[str, Any]:  # 记录真实执行数值测试的 FEniCSx 与底层运行时身份。
    return {  # 返回版本和容器身份而不读取秘密环境变量。
        "execution_family": "FEniCS/FEniCSx",  # 冻结用户确认的 FEN 执行技术族。
        "runtime_image": os.environ.get("FENICSX_RUNTIME_IMAGE", ""),  # 保存固定官方镜像完整标签与摘要。
        "dolfinx": str(dolfinx_version),  # 保存 DOLFINx 实际版本。
        "basix": str(getattr(basix, "__version__", "")),  # 保存 Basix 实际版本。
        "ufl": str(getattr(ufl, "__version__", "")),  # 保存 UFL 实际版本。
        "ffcx": str(getattr(ffcx, "__version__", "")),  # 保存 FFCx 实际版本。
        "petsc": ".".join(str(value) for value in PETSc.Sys.getVersion()),  # 保存 PETSc 主次补丁版本。
        "python": sys.version,  # 保存完整 Python 构建版本。
        "platform": platform.platform(),  # 保存容器操作系统和架构文本。
        "mpi_size": int(COMM.size),  # 保存实际 MPI 通信器进程数。
        "mpi_rank": int(COMM.rank),  # 保存当前进程秩；最终 receipt 由零号进程写出。
    }  # 结束运行时身份对象。


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:  # 追加统一结构的数值验收检查。
    checks.append({"name": name, "passed": bool(passed), "details": details})  # 强制布尔状态并保留可审计细节。


def checks_pass(checks: list[dict[str, Any]]) -> bool:  # 判断当前案例全部命名检查是否通过。
    return bool(checks) and all(bool(check["passed"]) for check in checks)  # 拒绝空检查列表和任意单项失败。


def global_real(local_value: Any) -> float:  # 将当前 rank 的标量贡献求和并规范化为 JSON 可写浮点数。
    global_value = COMM.allreduce(local_value, op=MPI.SUM)  # 对所有 MPI rank 的局部装配量执行全局求和。
    return float(np.real(global_value))  # 去除可能的零虚部并转换为 Python 浮点数。


def global_integer(local_value: int) -> int:  # 将当前 rank 的整数计数汇总为全局计数。
    return int(COMM.allreduce(int(local_value), op=MPI.SUM))  # 用 MPI 求和并返回 Python 整数。


def assemble_real(expression: Any) -> float:  # 编译并装配一个 UFL 标量形式后执行 MPI 全局求和。
    compiled_form = fem.form(expression)  # 使用当前 FFCx 运行时真实编译传入 UFL 形式。
    local_value = fem.assemble_scalar(compiled_form)  # 装配当前 MPI rank 拥有单元或边界的局部贡献。
    return global_real(local_value)  # 返回跨 rank 汇总后的真实标量。


def on_unit_square_boundary(points: np.ndarray) -> np.ndarray:  # 标记单位方形全部四条外边界上的几何点。
    left = np.isclose(points[0], 0.0)  # 标记横坐标为零的左边界点。
    right = np.isclose(points[0], 1.0)  # 标记横坐标为一的右边界点。
    bottom = np.isclose(points[1], 0.0)  # 标记纵坐标为零的下边界点。
    top = np.isclose(points[1], 1.0)  # 标记纵坐标为一的上边界点。
    return left | right | bottom | top  # 返回四条边界的布尔并集。


def observed_rates(values: list[float]) -> list[float | None]:  # 计算相邻二分网格误差的二进制对数观测阶。
    rates: list[float | None] = []  # 创建允许用空值表示无定义观测阶的列表。
    for index in range(1, len(values)):  # 从第二个误差开始与前一层配对。
        coarse_value = float(values[index - 1])  # 读取较粗层误差并规范化为 Python 浮点数。
        fine_value = float(values[index])  # 读取较细层误差并规范化为 Python 浮点数。
        if not math.isfinite(coarse_value) or not math.isfinite(fine_value) or coarse_value <= 0.0 or fine_value <= 0.0:  # 识别零值、负值、NaN 或无穷值异常。
            rates.append(None)  # 使用 JSON 合法空值让后续命名检查失败且仍可写出 receipt。
            continue  # 跳过当前无定义的对数计算并处理下一对层级。
        rates.append(math.log(coarse_value / fine_value, 2.0))  # 对当前正有限误差对计算 log2 比值。
    return rates  # 返回可由后续有限性和范围门审计的观测阶列表。


def strictly_decreasing(values: list[float]) -> bool:  # 判断正误差序列是否随网格加密严格下降。
    return all(values[index] < values[index - 1] for index in range(1, len(values)))  # 检查每个相邻层级均有改善。


def run_fen003(output_root: Path) -> dict[str, Any]:  # 运行制造解 Poisson 四层网格收敛真实数值测试。
    case_root = output_root / "fen-003"  # 为 FEN-003 合成基准使用独立 artifact 子目录。
    case_root.mkdir(parents=True, exist_ok=True)  # 在求解前创建目录以便失败时仍可留证。
    exact_qoi = 4.0 / (math.pi * math.pi)  # 解析计算正弦制造解在单位方形上的全域积分。
    level_results: list[dict[str, Any]] = []  # 收集四个受控网格层级的数值结果。
    for divisions in FEN003_LEVELS:  # 每次只改变两个方向的等分数以隔离离散误差。
        domain = mesh.create_unit_square(COMM, divisions, divisions, cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.right)  # 生成确定性右对角三角网格。
        topological_dimension = domain.topology.dim  # 读取二维网格拓扑维数。
        facet_dimension = topological_dimension - 1  # 将边界实体维数设为一。
        boundary_facets = mesh.locate_entities_boundary(domain, facet_dimension, on_unit_square_boundary)  # 几何定位四条 Dirichlet 边界。
        function_space = fem.functionspace(domain, ("Lagrange", 1))  # 创建连续一阶 Lagrange 标量函数空间。
        boundary_dofs = fem.locate_dofs_topological(function_space, facet_dimension, boundary_facets)  # 从边界 facets 映射受约束自由度。
        boundary_condition = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, function_space)  # 施加解析解对应的全边界零值条件。
        trial = ufl.TrialFunction(function_space)  # 创建未知离散解的试函数。
        test = ufl.TestFunction(function_space)  # 创建弱式测试函数。
        coordinates = ufl.SpatialCoordinate(domain)  # 创建 UFL 空间坐标供解析解与源项使用。
        exact_solution = ufl.sin(math.pi * coordinates[0]) * ufl.sin(math.pi * coordinates[1])  # 定义 u*=sin(pi*x)sin(pi*y)。
        source_term = 2.0 * math.pi * math.pi * exact_solution  # 根据负 Laplace 算子解析得到 f=2*pi^2*u*。
        integration_measure = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": QUADRATURE_DEGREE})  # 固定八阶体积分规则。
        bilinear_form = ufl.inner(ufl.grad(trial), ufl.grad(test)) * integration_measure  # 定义 Poisson 刚度双线性形式。
        linear_form = ufl.inner(source_term, test) * integration_measure  # 定义制造源项载荷线性形式。
        problem = LinearProblem(bilinear_form, linear_form, bcs=[boundary_condition], petsc_options_prefix=f"fen003_n{divisions}_", petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True})  # 使用唯一前缀和 LU 真实求解。
        solution = problem.solve()  # 触发 FFCx 编译、矩阵向量组装和 PETSc 线性求解。
        solution.x.scatter_forward()  # 在提取 QoI 前同步可能存在的幽灵自由度值。
        converged_reason = int(problem.solver.getConvergedReason())  # 记录 PETSc 正数成功或负数失败原因码。
        iterations = int(problem.solver.getIterationNumber())  # 记录求解器实际迭代次数但不把 LU 次数当 golden 值。
        difference = solution - exact_solution  # 构造离散解与解析解的 UFL 差值。
        l2_error = math.sqrt(max(assemble_real(ufl.inner(difference, difference) * integration_measure), 0.0))  # 装配并开方得到全局 L2 误差。
        h1_error = math.sqrt(max(assemble_real(ufl.inner(ufl.grad(difference), ufl.grad(difference)) * integration_measure), 0.0))  # 装配 H1 半范误差。
        qoi_value = assemble_real(solution * integration_measure)  # 提取全域积分作为固定主 QoI。
        qoi_absolute_error = abs(qoi_value - exact_qoi)  # 计算主 QoI 的绝对误差。
        qoi_relative_error = qoi_absolute_error / abs(exact_qoi)  # 用非零解析值归一化得到相对误差。
        discrete_energy = assemble_real(ufl.inner(ufl.grad(solution), ufl.grad(solution)) * integration_measure)  # 计算离散应变能型一致性量。
        load_work = assemble_real(source_term * solution * integration_measure)  # 计算同一离散解上的外载功。
        energy_denominator = max(abs(discrete_energy), abs(load_work), 1.0e-30)  # 用极小正数防止异常零值除法。
        energy_balance_relative = abs(discrete_energy - load_work) / energy_denominator  # 计算能量与载荷功相对不平衡。
        global_cells = int(domain.topology.index_map(topological_dimension).size_global)  # 读取 DOLFINx 拓扑提供的全局单元数。
        global_dofs = int(function_space.dofmap.index_map.size_global * function_space.dofmap.index_map_bs)  # 读取函数空间全局自由度数。
        theoretical_h_max = math.sqrt(2.0) / float(divisions)  # 根据右对角单位方格解析计算最长三角形边长。
        owned_cells = np.arange(domain.topology.index_map(topological_dimension).size_local, dtype=np.int32)  # 构造当前 rank 拥有的单元索引供几何尺寸实测。
        local_h_values = domain.h(topological_dimension, owned_cells)  # 使用 DOLFINx 几何接口计算每个本地单元的尺寸量。
        local_h_max = float(np.max(local_h_values)) if local_h_values.size else 0.0  # 取得当前 rank 最大单元尺寸并处理空分区。
        actual_h_max = float(COMM.allreduce(local_h_max, op=MPI.MAX))  # 用 MPI 最大归约得到全局实测 hmax。
        level_result = {  # 组装当前网格层级的机器可读数值记录。
            "divisions_per_axis": divisions,  # 记录每个坐标方向的等分数。
            "theoretical_h_max": theoretical_h_max,  # 记录本结构化网格的解析最大边长。
            "actual_h_max": actual_h_max,  # 记录 DOLFINx 几何接口实际测得的最大单元尺寸。
            "global_cells": global_cells,  # 记录全局三角形数量。
            "global_dofs": global_dofs,  # 记录全局 CG1 自由度数量。
            "boundary_dofs": global_integer(int(boundary_dofs.size)),  # 汇总当前边界自由度局部计数。
            "ksp_converged_reason": converged_reason,  # 记录 PETSc 收敛原因码。
            "ksp_iterations": iterations,  # 记录 PETSc 实际迭代次数。
            "qoi_value": qoi_value,  # 记录全域积分 QoI 数值。
            "qoi_absolute_error": qoi_absolute_error,  # 记录 QoI 绝对误差。
            "qoi_relative_error": qoi_relative_error,  # 记录 QoI 相对误差。
            "l2_error": l2_error,  # 记录解的 L2 范数误差。
            "h1_seminorm_error": h1_error,  # 记录解梯度的 H1 半范误差。
            "discrete_energy": discrete_energy,  # 记录离散能量型标量。
            "load_work": load_work,  # 记录载荷功标量。
            "energy_balance_relative": energy_balance_relative,  # 记录独立平衡相对误差。
        }  # 结束当前层级结果对象。
        level_results.append(level_result)  # 按粗到细顺序保存当前层级。
        if divisions == FEN003_LEVELS[-1]:  # 只为最细层保存场结果以保持 artifact 轻量。
            solution.name = "fen003_mms_solution"  # 设置 XDMF 中可识别的数值场名称。
            solution_path = case_root / "fen003_finest_solution.xdmf"  # 定义最细层 XDMF 输出路径。
            with io.XDMFFile(COMM, solution_path, "w", encoding=io.XDMFFile.Encoding.HDF5) as solution_file:  # 显式创建 HDF5 后端 XDMF 场文件，避免依赖默认编码。
                solution_file.write_mesh(domain)  # 写出与最细层解对应的网格。
                solution_file.write_function(solution)  # 写出真实求解得到的有限元函数。
    l2_values = [float(result["l2_error"]) for result in level_results]  # 提取 L2 误差序列供单调性与阶次判断。
    h1_values = [float(result["h1_seminorm_error"]) for result in level_results]  # 提取 H1 半范误差序列。
    qoi_values = [float(result["qoi_absolute_error"]) for result in level_results]  # 提取 QoI 绝对误差序列。
    l2_rates = observed_rates(l2_values)  # 计算三个相邻层级的 L2 观测阶。
    h1_rates = observed_rates(h1_values)  # 计算三个相邻层级的 H1 观测阶。
    qoi_rates = observed_rates(qoi_values)  # 仅记录 QoI 观测阶而不设置可能误伤超收敛的上限。
    checks: list[dict[str, Any]] = []  # 创建预冻结的 FEN-003 数值验收列表。
    add_check(checks, "four_real_mesh_levels", len(level_results) == len(FEN003_LEVELS), {"expected": len(FEN003_LEVELS), "actual": len(level_results)})  # 要求四层均实际完成。
    add_check(checks, "mesh_cell_and_dof_counts", all(int(result["global_cells"]) == 2 * int(result["divisions_per_axis"]) ** 2 and int(result["global_dofs"]) == (int(result["divisions_per_axis"]) + 1) ** 2 for result in level_results), [{"n": result["divisions_per_axis"], "cells": result["global_cells"], "dofs": result["global_dofs"]} for result in level_results])  # 核对网格拓扑与空间规模。
    add_check(checks, "measured_hmax_matches_contract", all(abs(float(result["actual_h_max"]) - float(result["theoretical_h_max"])) <= 1.0e-12 for result in level_results), [{"n": result["divisions_per_axis"], "expected": result["theoretical_h_max"], "actual": result["actual_h_max"]} for result in level_results])  # 要求实测网格尺寸匹配右对角解析值。
    add_check(checks, "all_ksp_solves_converged", all(int(result["ksp_converged_reason"]) > 0 for result in level_results), [result["ksp_converged_reason"] for result in level_results])  # 要求四次真实 PETSc 求解全部收敛。
    add_check(checks, "all_metrics_finite", all(math.isfinite(float(value)) for result in level_results for key, value in result.items() if key not in {"divisions_per_axis", "global_cells", "global_dofs", "boundary_dofs", "ksp_converged_reason", "ksp_iterations"}), "全部浮点指标必须为有限数。")  # 拒绝 NaN 与无穷值。
    add_check(checks, "l2_error_strictly_decreases", strictly_decreasing(l2_values), l2_values)  # 要求 L2 误差随加密严格下降。
    add_check(checks, "h1_error_strictly_decreases", strictly_decreasing(h1_values), h1_values)  # 要求 H1 半范误差随加密严格下降。
    add_check(checks, "qoi_error_strictly_decreases", strictly_decreasing(qoi_values), qoi_values)  # 要求固定 QoI 误差随加密严格下降。
    add_check(checks, "late_l2_rates_match_p1", all(rate is not None and 1.7 <= rate <= 2.3 for rate in l2_rates[-2:]), l2_rates)  # 要求后两段 L2 阶存在且接近理论二阶。
    add_check(checks, "late_h1_rates_match_p1", all(rate is not None and 0.8 <= rate <= 1.2 for rate in h1_rates[-2:]), h1_rates)  # 要求后两段 H1 阶存在且接近理论一阶。
    add_check(checks, "qoi_rates_positive", all(rate is not None and rate > 0.0 for rate in qoi_rates), qoi_rates)  # 只要求每段 QoI 观测阶存在且为正。
    add_check(checks, "finest_l2_within_limit", l2_values[-1] <= FEN003_FINE_L2_LIMIT, {"limit": FEN003_FINE_L2_LIMIT, "actual": l2_values[-1]})  # 检查最细层 L2 精度。
    add_check(checks, "finest_h1_within_limit", h1_values[-1] <= FEN003_FINE_H1_LIMIT, {"limit": FEN003_FINE_H1_LIMIT, "actual": h1_values[-1]})  # 检查最细层 H1 精度。
    add_check(checks, "finest_qoi_within_limit", float(level_results[-1]["qoi_relative_error"]) <= FEN003_FINE_QOI_RELATIVE_LIMIT, {"limit": FEN003_FINE_QOI_RELATIVE_LIMIT, "actual": level_results[-1]["qoi_relative_error"]})  # 检查最细层 QoI 精度。
    add_check(checks, "energy_balance_within_limit", all(float(result["energy_balance_relative"]) <= FEN003_ENERGY_BALANCE_LIMIT for result in level_results), {"limit": FEN003_ENERGY_BALANCE_LIMIT, "actual": [result["energy_balance_relative"] for result in level_results]})  # 检查四层能量一致性。
    case_success = checks_pass(checks)  # 只有全部预冻结检查通过才接受合成基准。
    solution_xdmf = case_root / "fen003_finest_solution.xdmf"  # 定位最细层 XDMF 文件供来源哈希记录。
    solution_h5 = case_root / "fen003_finest_solution.h5"  # 定位 XDMF 自动生成的 HDF5 数据文件。
    receipt = {  # 构造 FEN-003 合成数值合同主回执。
        "schema_version": "v5-fenicsx-synthetic-case/1.0",  # 冻结单案例回执版本。
        "case_id": "FEN-003",  # 记录所覆盖的能力参考案例。
        "benchmark_id": "SYN-FEN-003-MMS-POISSON-P1",  # 使用 SYN 前缀阻止与原研究案例混淆。
        "artifact_kind": "synthetic_numerical_contract_evidence",  # 明确证据类别为合成数值合同。
        "execution_family": "FEniCS/FEniCSx",  # 记录真实执行技术族。
        "contract_test_outcome": "pass" if case_success else "fail",  # 记录数值合同通过或失败。
        "research_case_execution_status": "not_executed_missing_current_evidence",  # 保留原研究案例仍未执行的状态。
        "scientific_claim_allowed": False,  # 禁止把合成基准解释为工程或论文结论。
        "synthetic_numeric_results_generated": True,  # 明确本次确实生成了数值网格、解和指标。
        "problem_contract": {"domain": "unit_square", "pde": "minus_laplacian_u_equals_f", "exact_solution": "sin(pi*x)*sin(pi*y)", "source": "2*pi^2*sin(pi*x)*sin(pi*y)", "boundary": "homogeneous_dirichlet_all_sides", "element": "continuous_lagrange_degree_1", "diagonal": "right", "quadrature_degree": QUADRATURE_DEGREE},  # 保存制造解问题的全部固定要素。
        "qoi_contract": {"quantity": "domain_integral_of_scalar_solution", "exact_value": exact_qoi, "unit": "dimensionless", "spatial_selection": "whole_unit_square", "coordinate_frame": "cartesian_xy", "analysis_step": "steady", "reduction": "MPI_SUM", "singularity_exposure": "none_by_design"},  # 保存固定主 QoI 定义。
        "mesh_levels": list(FEN003_LEVELS),  # 保存四个受控等分层级。
        "level_results": level_results,  # 保存逐层真实数值结果。
        "convergence": {"l2_rates": l2_rates, "h1_seminorm_rates": h1_rates, "qoi_rates": qoi_rates},  # 保存三类观测收敛阶。
        "checks": checks,  # 保存全部预冻结验收结果。
        "runtime": runtime_identity(),  # 保存实际 FEniCSx 和 PETSc 运行时。
        "provenance": source_identity(),  # 保存当前提交与生成输入来源。
        "files": {"solution_xdmf": artifact_file_record(solution_xdmf, output_root), "solution_h5": artifact_file_record(solution_h5, output_root)},  # 保存最细层场文件的相对路径、字节数和哈希。
        "execution_counts": {"mesh_generation_calls": len(FEN003_LEVELS), "mesh_import_calls": 0, "linear_solve_calls": len(FEN003_LEVELS), "qoi_extraction_calls": len(FEN003_LEVELS), "calculix_calls": 0, "model_calls": 0},  # 记录真实调用边界。
        "allowed_use": ["验证固定 DOLFINx 环境能完成网格收敛、组装、求解和 QoI 提取链。", "作为 FEN-003 后续真实输入测试的回归基线。"],  # 限定允许用途。
        "disallowed_use": ["不得宣称原 FEN-003 研究输入已执行或通过。", "不得删除或满足 fen-003.json 中五项 missing_facts。", "不得据此形成工程设计或论文数值结论。"],  # 限定禁止用途。
    }  # 结束 FEN-003 回执。
    write_json(case_root / "numerical_contract_receipt.json", receipt)  # 写出单案例机器回执。
    return receipt  # 返回给整体活动汇总。


def build_fen014_tags(domain: mesh.Mesh) -> tuple[mesh.MeshTags, mesh.MeshTags, dict[str, Any]]:  # 为单位方形创建左右 cell tags 和四边 facet tags。
    topological_dimension = domain.topology.dim  # 读取二维拓扑维数。
    facet_dimension = topological_dimension - 1  # 将边界 facet 维数设为一。
    domain.topology.create_connectivity(facet_dimension, topological_dimension)  # 建立 facet 到 cell 邻接供标签写出与读取检查。
    domain.topology.create_connectivity(facet_dimension, 0)  # 建立 facet 到顶点邻接供 XDMF 序列化。
    local_cell_count = domain.topology.index_map(topological_dimension).size_local  # 读取当前 rank 拥有的非幽灵单元数。
    local_cells = np.arange(local_cell_count, dtype=np.int32)  # 构造按本地实体编号排序的 cell 索引。
    cell_midpoints = mesh.compute_midpoints(domain, topological_dimension, local_cells)  # 计算每个单元几何中点供左右分区。
    cell_values = np.where(cell_midpoints[:, 0] < 0.5, FEN014_CELL_IDS[0], FEN014_CELL_IDS[1]).astype(np.int32)  # 按横坐标将单元标记为左一或右二。
    cell_tags = mesh.meshtags(domain, topological_dimension, local_cells, cell_values)  # 创建覆盖全部本地单元的 MeshTags。
    cell_tags.name = FEN014_CELL_TAG_NAME  # 设置固定 XDMF cell tag 名称。
    left_facets = mesh.locate_entities_boundary(domain, facet_dimension, lambda points: np.isclose(points[0], 0.0))  # 定位左边界 facets。
    right_facets = mesh.locate_entities_boundary(domain, facet_dimension, lambda points: np.isclose(points[0], 1.0))  # 定位右边界 facets。
    bottom_facets = mesh.locate_entities_boundary(domain, facet_dimension, lambda points: np.isclose(points[1], 0.0))  # 定位下边界 facets。
    top_facets = mesh.locate_entities_boundary(domain, facet_dimension, lambda points: np.isclose(points[1], 1.0))  # 定位上边界 facets。
    facet_groups = (left_facets, right_facets, bottom_facets, top_facets)  # 按十一至十四的约定冻结四组边界实体。
    facet_indices = np.concatenate(facet_groups).astype(np.int32)  # 合并四条边界的本地 facet 索引。
    facet_values = np.concatenate([np.full(group.size, tag_id, dtype=np.int32) for group, tag_id in zip(facet_groups, FEN014_FACET_IDS)])  # 为每组 facet 生成对应标签值。
    facet_order = np.argsort(facet_indices)  # 计算 MeshTags 要求的实体升序排列。
    sorted_facet_indices = facet_indices[facet_order]  # 按实体编号排序 facet 索引。
    sorted_facet_values = facet_values[facet_order]  # 用同一排列保持标签与实体配对。
    facet_tags = mesh.meshtags(domain, facet_dimension, sorted_facet_indices, sorted_facet_values)  # 创建四边 facet MeshTags。
    facet_tags.name = FEN014_FACET_TAG_NAME  # 设置固定 XDMF facet tag 名称。
    geometric_boundary = mesh.locate_entities_boundary(domain, facet_dimension, on_unit_square_boundary)  # 独立几何定位全部外边界 facets。
    boundary_sets_match = np.array_equal(np.sort(np.unique(sorted_facet_indices)), np.sort(np.unique(geometric_boundary)))  # 检查标签并集恰好覆盖几何外边界。
    diagnostics = {"tagged_boundary_matches_geometry": bool(boundary_sets_match), "local_duplicate_tagged_facets": int(sorted_facet_indices.size - np.unique(sorted_facet_indices).size)}  # 保存映射预检结果。
    return cell_tags, facet_tags, diagnostics  # 返回两类标签和独立覆盖诊断。


def write_fen014_import_package(path: Path, domain: mesh.Mesh, cell_tags: mesh.MeshTags, facet_tags: mesh.MeshTags) -> None:  # 将网格与两类命名标签写入 HDF5 XDMF 包。
    domain.name = FEN014_MESH_NAME  # 设置固定网格名称供重读时显式选择。
    with io.XDMFFile(COMM, path, "w", encoding=io.XDMFFile.Encoding.HDF5) as xdmf_file:  # 显式创建 HDF5 编码的 XDMF 与配套数据文件。
        xdmf_file.write_mesh(domain)  # 首先写入网格拓扑和几何。
        xdmf_file.write_meshtags(cell_tags, domain.geometry)  # 写入左右 cell tags 并关联同一几何。
        xdmf_file.write_meshtags(facet_tags, domain.geometry)  # 写入四边 facet tags 并关联同一几何。


def read_fen014_import_package(path: Path) -> tuple[mesh.Mesh, mesh.MeshTags, mesh.MeshTags]:  # 从关闭后的 XDMF 包创建全新的网格和标签对象。
    with io.XDMFFile(COMM, path, "r", encoding=io.XDMFFile.Encoding.HDF5) as xdmf_file:  # 以显式 HDF5 编码只读方式重新打开已落盘文件。
        imported_domain = xdmf_file.read_mesh(name=FEN014_MESH_NAME)  # 按固定名称读取全新 Mesh 对象。
        topological_dimension = imported_domain.topology.dim  # 读取重建网格的拓扑维数。
        facet_dimension = topological_dimension - 1  # 计算重建网格的 facet 维数。
        imported_domain.topology.create_connectivity(facet_dimension, topological_dimension)  # 在读 facet tags 前建立 facet 到 cell 邻接。
        imported_domain.topology.create_connectivity(facet_dimension, 0)  # 建立 facet 到顶点邻接供标签读取。
        imported_cell_tags = xdmf_file.read_meshtags(imported_domain, name=FEN014_CELL_TAG_NAME)  # 按名称读取左右 cell tags。
        imported_facet_tags = xdmf_file.read_meshtags(imported_domain, name=FEN014_FACET_TAG_NAME)  # 按名称读取四边 facet tags。
    return imported_domain, imported_cell_tags, imported_facet_tags  # 返回与参考对象无共享状态的重读对象。


def tag_geometry_report(domain: mesh.Mesh, tags: mesh.MeshTags, tag_ids: tuple[int, ...]) -> dict[str, Any]:  # 用实体中点构造与本地实体编号无关的标签几何报告。
    local_records: list[tuple[int, float, float]] = []  # 收集当前 rank 的标签值与二维中点坐标记录。
    midpoint_means: dict[str, list[float]] = {}  # 保存每个标签实体中点的全局平均坐标。
    for tag_id in tag_ids:  # 按冻结标签顺序逐一处理实体集合。
        entity_indices = tags.find(tag_id).astype(np.int32)  # 从当前 MeshTags 取得本地实体索引。
        midpoints = mesh.compute_midpoints(domain, tags.dim, entity_indices)  # 计算标签实体的几何中点而不使用实体编号比较。
        for midpoint in midpoints:  # 将每个实体转换为可稳定排序的几何记录。
            local_records.append((int(tag_id), round(float(midpoint[0]), 14), round(float(midpoint[1]), 14)))  # 将二维坐标舍入到十四位以消除无意义序列化噪声。
        global_count = global_integer(int(midpoints.shape[0]))  # 汇总当前标签的全局实体数量。
        global_x_sum = global_real(float(np.sum(midpoints[:, 0])) if midpoints.size else 0.0)  # 汇总全部实体中点横坐标。
        global_y_sum = global_real(float(np.sum(midpoints[:, 1])) if midpoints.size else 0.0)  # 汇总全部实体中点纵坐标。
        midpoint_means[str(tag_id)] = [global_x_sum / float(global_count), global_y_sum / float(global_count)] if global_count > 0 else [None, None]  # 计算非空标签的全局平均中点坐标；空标签使用严格 JSON 合法空值。
    gathered_records = COMM.gather(local_records, root=ROOT_RANK)  # 将各 rank 几何记录汇集到零号进程。
    if COMM.rank == ROOT_RANK:  # 仅由零号进程规范化并哈希全局记录。
        global_records = sorted(record for rank_records in gathered_records for record in rank_records)  # 展平并按标签与坐标稳定排序。
        encoded_records = json.dumps(global_records, ensure_ascii=True, separators=(",", ":")).encode("ascii")  # 使用无空格 ASCII JSON 固定字节表达。
        signature = hashlib.sha256(encoded_records).hexdigest()  # 计算标签值与实体几何绑定关系摘要。
    else:  # 非零进程没有全局记录集合。
        signature = None  # 使用空占位等待 MPI 广播零号进程结果。
    shared_signature = COMM.bcast(signature, root=ROOT_RANK)  # 将统一摘要广播给全部 rank 保持返回结构一致。
    return {"signature_sha256": str(shared_signature), "midpoint_means": midpoint_means}  # 返回几何绑定摘要与可解释坐标均值。


def midpoint_means_match(report: dict[str, Any], expected_means: dict[str, list[float]]) -> bool:  # 安全比较标签中点均值并让空标签形成命名检查失败。
    for tag_id, expected_coordinates in expected_means.items():  # 按预冻结标签语义逐项读取期望二维坐标。
        actual_coordinates = report.get("midpoint_means", {}).get(tag_id)  # 从报告读取可能缺失或为空的实际坐标。
        if not isinstance(actual_coordinates, list) or len(actual_coordinates) != 2:  # 拒绝缺失、非列表或非二维坐标结构。
            return False  # 将结构异常转换为可审计检查失败而不是抛出异常。
        for axis in (0, 1):  # 依次比较横纵两个笛卡尔坐标分量。
            actual_value = actual_coordinates[axis]  # 读取当前实际坐标分量。
            if actual_value is None:  # 空标签使用 None 表示没有可计算均值。
                return False  # 空标签不能满足几何语义合同。
            numeric_value = float(actual_value)  # 将 JSON 标量规范化为 Python 浮点数。
            if not math.isfinite(numeric_value) or abs(numeric_value - float(expected_coordinates[axis])) > FEN014_MEASURE_LIMIT:  # 检查有限性和预冻结绝对容差。
                return False  # 任一分量非有限或超差时立即返回失败。
    return True  # 只有全部标签的两个分量都存在、有限且在容差内时通过。


def preflight_fen014(domain: mesh.Mesh, cell_tags: mesh.MeshTags, facet_tags: mesh.MeshTags) -> dict[str, Any]:  # 在任何 LinearProblem 之前验证拓扑、尺度、标签与边界映射。
    topological_dimension = int(domain.topology.dim)  # 读取当前网格拓扑维数。
    geometric_dimension = int(domain.geometry.dim)  # 读取当前网格几何坐标维数。
    facet_dimension = topological_dimension - 1  # 计算边界 facet 维数。
    domain.topology.create_connectivity(facet_dimension, topological_dimension)  # 确保 facet 到 cell 邻接存在。
    domain.topology.create_connectivity(facet_dimension, 0)  # 确保 facet 到顶点邻接存在。
    global_cells = int(domain.topology.index_map(topological_dimension).size_global)  # 读取全局单元数量。
    global_vertices = int(domain.topology.index_map(0).size_global)  # 读取全局顶点数量。
    global_facets = int(domain.topology.index_map(facet_dimension).size_global)  # 读取全局 facet 数量。
    coordinate_array = np.asarray(domain.geometry.x[:, :geometric_dimension], dtype=np.float64)  # 读取当前网格使用的几何坐标分量。
    coordinate_finite = bool(np.all(np.isfinite(coordinate_array)))  # 检查所有坐标均为有限数。
    bounding_box_min = np.min(coordinate_array, axis=0).tolist() if coordinate_array.size else []  # 计算当前坐标包围盒最小角。
    bounding_box_max = np.max(coordinate_array, axis=0).tolist() if coordinate_array.size else []  # 计算当前坐标包围盒最大角。
    cell_value_set = sorted(int(value) for value in np.unique(cell_tags.values))  # 读取 cell tags 的精确值集合。
    facet_value_set = sorted(int(value) for value in np.unique(facet_tags.values))  # 读取 facet tags 的精确值集合。
    cell_tag_counts = {str(tag_id): int(cell_tags.find(tag_id).size) for tag_id in FEN014_CELL_IDS}  # 串行统计左右 cell 标签数量。
    facet_tag_counts = {str(tag_id): int(facet_tags.find(tag_id).size) for tag_id in FEN014_FACET_IDS}  # 串行统计四边 facet 标签数量。
    cell_geometry = tag_geometry_report(domain, cell_tags, FEN014_CELL_IDS)  # 构造 cell 标签与实体几何绑定报告。
    facet_geometry = tag_geometry_report(domain, facet_tags, FEN014_FACET_IDS)  # 构造 facet 标签与实体几何绑定报告。
    tagged_facets = np.sort(np.unique(np.concatenate([facet_tags.find(tag_id) for tag_id in FEN014_FACET_IDS]))).astype(np.int32)  # 构造规范化标签边界 facet 并集。
    geometric_facets = np.sort(np.unique(mesh.locate_entities_boundary(domain, facet_dimension, on_unit_square_boundary))).astype(np.int32)  # 独立几何定位规范化外边界 facet 集合。
    preflight_space = fem.functionspace(domain, ("Lagrange", 1))  # 创建只用于边界映射预检的 CG1 空间。
    tagged_dofs = np.sort(np.unique(fem.locate_dofs_topological(preflight_space, facet_dimension, tagged_facets)))  # 从标签边界得到规范化自由度集合。
    geometric_dofs = np.sort(np.unique(fem.locate_dofs_topological(preflight_space, facet_dimension, geometric_facets)))  # 从几何边界得到独立自由度集合。
    local_cell_count = int(domain.topology.index_map(topological_dimension).size_local)  # 读取当前串行进程拥有的单元数。
    geometry_dofmap = domain.geometry.dofmaps[0]  # 使用 DOLFINx 0.11 的首个几何 dofmap 取得单元顶点映射。
    cell_areas: list[float] = []  # 收集每个三角形的绝对面积供退化和尺度检查。
    cell_keys: list[tuple[tuple[float, float], ...]] = []  # 收集与顶点编号无关的单元几何键供重复检测。
    orientation_histogram = {"positive": 0, "negative": 0, "zero": 0}  # 记录三角形有向面积符号分布而不假定输入编号一致。
    cell_vertex_uniqueness = True  # 初始化每个三角形拥有三个不同几何点的检查状态。
    for cell_index in range(local_cell_count):  # 逐一检查当前串行网格拥有的三角形。
        geometry_indices = np.asarray(geometry_dofmap[cell_index], dtype=np.int32)  # 读取当前单元三个几何节点索引。
        points = coordinate_array[geometry_indices, :2]  # 取得当前单元二维顶点坐标。
        normalized_vertices = tuple(sorted((round(float(point[0]), 14), round(float(point[1]), 14)) for point in points))  # 生成与节点次序无关的规范顶点元组。
        cell_keys.append(normalized_vertices)  # 保存当前单元几何键供全局重复检测。
        cell_vertex_uniqueness = cell_vertex_uniqueness and len(set(normalized_vertices)) == 3  # 要求当前三角形包含三个不同点。
        first_edge = points[1] - points[0]  # 构造从首顶点指向第二顶点的二维边向量。
        second_edge = points[2] - points[0]  # 构造从首顶点指向第三顶点的二维边向量。
        signed_twice_area = float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])  # 用二维行列式计算两倍有向面积，避免弃用的二维 cross 接口。
        cell_areas.append(0.5 * abs(signed_twice_area))  # 保存与方向无关的正面积。
        orientation_key = "positive" if signed_twice_area > 0.0 else "negative" if signed_twice_area < 0.0 else "zero"  # 将有向面积归入正、负或零类别。
        orientation_histogram[orientation_key] += 1  # 累加当前方向类别数量。
    minimum_cell_area = min(cell_areas) if cell_areas else 0.0  # 读取最小绝对单元面积供退化检查。
    maximum_cell_area = max(cell_areas) if cell_areas else 0.0  # 读取最大绝对单元面积供均匀尺度检查。
    expected_cell_midpoints = {"1": [0.25, 0.5], "2": [0.75, 0.5]}  # 冻结左右半区中点平均坐标语义。
    expected_facet_midpoints = {"11": [0.0, 0.5], "12": [1.0, 0.5], "13": [0.5, 0.0], "14": [0.5, 1.0]}  # 冻结四边标签中点平均坐标语义。
    checks: list[dict[str, Any]] = []  # 创建求解前必须全部通过的命名检查列表。
    add_check(checks, "serial_execution_scope", COMM.size == 1, {"expected_mpi_size": 1, "actual_mpi_size": int(COMM.size)})  # 限定当前计数与实体集合合同只用于单进程。
    add_check(checks, "dimensions_exact", topological_dimension == 2 and geometric_dimension == 2 and cell_tags.dim == 2 and facet_tags.dim == 1, {"tdim": topological_dimension, "gdim": geometric_dimension, "cell_tag_dim": int(cell_tags.dim), "facet_tag_dim": int(facet_tags.dim)})  # 检查二维拓扑、几何和标签维数。
    add_check(checks, "topology_counts_exact", (global_vertices, global_cells, global_facets) == (81, 128, 208), {"vertices": global_vertices, "cells": global_cells, "facets": global_facets})  # 检查八乘八三角网格拓扑规模。
    add_check(checks, "coordinates_finite_and_unit_scale", coordinate_finite and np.allclose(bounding_box_min, [0.0, 0.0], atol=1.0e-14, rtol=0.0) and np.allclose(bounding_box_max, [1.0, 1.0], atol=1.0e-14, rtol=0.0), {"finite": coordinate_finite, "bbox_min": bounding_box_min, "bbox_max": bounding_box_max})  # 检查单位方形尺度与有限坐标。
    add_check(checks, "tag_names_dims_and_values", cell_tags.name == FEN014_CELL_TAG_NAME and facet_tags.name == FEN014_FACET_TAG_NAME and cell_value_set == list(FEN014_CELL_IDS) and facet_value_set == list(FEN014_FACET_IDS), {"cell_name": cell_tags.name, "facet_name": facet_tags.name, "cell_values": cell_value_set, "facet_values": facet_value_set})  # 检查命名、维数和精确标签值集合。
    add_check(checks, "tag_counts_exact", cell_tag_counts == {"1": 64, "2": 64} and facet_tag_counts == {"11": 8, "12": 8, "13": 8, "14": 8}, {"cell_counts": cell_tag_counts, "facet_counts": facet_tag_counts})  # 检查全部预期实体均被标记一次。
    add_check(checks, "cell_tags_cover_all_cells_once", len(np.unique(cell_tags.indices)) == local_cell_count and len(cell_tags.indices) == local_cell_count, {"tagged": int(len(cell_tags.indices)), "unique": int(len(np.unique(cell_tags.indices))), "owned_cells": local_cell_count})  # 检查 cell tags 无遗漏无重复覆盖全部单元。
    add_check(checks, "facet_tags_cover_exact_boundary", np.array_equal(tagged_facets, geometric_facets) and len(facet_tags.indices) == len(np.unique(facet_tags.indices)), {"tagged_facets": int(tagged_facets.size), "geometric_facets": int(geometric_facets.size), "raw_tag_entries": int(len(facet_tags.indices))})  # 检查 facet tags 恰好覆盖外边界且无重复。
    add_check(checks, "tagged_dofs_match_geometry", np.array_equal(tagged_dofs, geometric_dofs) and tagged_dofs.size == 32, {"tagged_dofs": int(tagged_dofs.size), "geometric_dofs": int(geometric_dofs.size)})  # 检查标签派生边界自由度与几何定位完全一致。
    add_check(checks, "cell_tag_semantics_exact", midpoint_means_match(cell_geometry, expected_cell_midpoints), {"expected": expected_cell_midpoints, "actual": cell_geometry["midpoint_means"]})  # 检查 cell 标签一为左、二为右，并把空标签保留为命名失败。
    add_check(checks, "facet_tag_semantics_exact", midpoint_means_match(facet_geometry, expected_facet_midpoints), {"expected": expected_facet_midpoints, "actual": facet_geometry["midpoint_means"]})  # 检查 facet 标签十一至十四对应左、右、下、上，并拒绝空标签。
    add_check(checks, "cells_unique_nondegenerate_and_scaled", cell_vertex_uniqueness and len(set(cell_keys)) == local_cell_count and minimum_cell_area > 0.0 and abs(minimum_cell_area - 1.0 / 128.0) <= FEN014_MEASURE_LIMIT and abs(maximum_cell_area - 1.0 / 128.0) <= FEN014_MEASURE_LIMIT and orientation_histogram["zero"] == 0, {"unique_cells": len(set(cell_keys)), "owned_cells": local_cell_count, "min_area": minimum_cell_area, "max_area": maximum_cell_area, "orientation_histogram": orientation_histogram})  # 检查无重复、无退化且单元尺度固定。
    return {"passed": checks_pass(checks), "checks": checks, "topology": {"tdim": topological_dimension, "gdim": geometric_dimension, "vertices": global_vertices, "cells": global_cells, "facets": global_facets}, "bbox": {"min": bounding_box_min, "max": bounding_box_max}, "cell_tag_geometry": cell_geometry, "facet_tag_geometry": facet_geometry, "orientation_histogram": orientation_histogram, "parallel_scope": "serial_only"}  # 返回求解门和完整预检证据。


def solve_fen014_patch(domain: mesh.Mesh, cell_tags: mesh.MeshTags, facet_tags: mesh.MeshTags, prefix: str) -> dict[str, Any]:  # 在指定网格和标签上求解 u=x+y 线性补丁问题。
    topological_dimension = domain.topology.dim  # 读取当前网格拓扑维数。
    facet_dimension = topological_dimension - 1  # 计算当前边界 facet 维数。
    function_space = fem.functionspace(domain, ("Lagrange", 1))  # 为当前网格重新创建独立 CG1 函数空间。
    tagged_facets = np.unique(np.concatenate([facet_tags.find(tag_id) for tag_id in FEN014_FACET_IDS])).astype(np.int32)  # 从重读标签构造全部边界 facet 并集。
    tagged_boundary_dofs = fem.locate_dofs_topological(function_space, facet_dimension, tagged_facets)  # 仅使用 facet tags 映射 Dirichlet 自由度。
    geometric_facets = mesh.locate_entities_boundary(domain, facet_dimension, on_unit_square_boundary)  # 独立按几何定位边界 facets。
    geometric_boundary_dofs = fem.locate_dofs_topological(function_space, facet_dimension, geometric_facets)  # 独立映射几何边界自由度。
    boundary_values = fem.Function(function_space)  # 创建承载非齐次解析边界值的离散函数。
    boundary_values.interpolate(lambda points: points[0] + points[1])  # 将线性解析解 u=x+y 插值到全部自由度。
    boundary_condition = fem.dirichletbc(boundary_values, tagged_boundary_dofs)  # 仅在标签派生自由度上施加解析 Dirichlet 值。
    trial = ufl.TrialFunction(function_space)  # 创建当前网格的独立试函数。
    test = ufl.TestFunction(function_space)  # 创建当前网格的独立测试函数。
    cell_measure = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)  # 从当前 cell tags 重建带子域的体积分 Measure。
    facet_measure = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)  # 从当前 facet tags 重建带边界标签的 Measure。
    zero_source = fem.Constant(domain, PETSc.ScalarType(0.0))  # 定义 Laplace 补丁问题的零体源项。
    stiffness_density = ufl.inner(ufl.grad(trial), ufl.grad(test))  # 定义不含 Measure 的 Laplace 刚度密度。
    bilinear_form = stiffness_density * cell_measure(FEN014_CELL_IDS[0]) + stiffness_density * cell_measure(FEN014_CELL_IDS[1])  # 分别在左右 cell tags 上积分并相加，避免依赖 Measure 相加语义。
    linear_form = zero_source * test * cell_measure(FEN014_CELL_IDS[0]) + zero_source * test * cell_measure(FEN014_CELL_IDS[1])  # 用同一两个标签子域组装零载荷。
    problem = LinearProblem(bilinear_form, linear_form, bcs=[boundary_condition], petsc_options_prefix=prefix, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True})  # 创建唯一前缀的真实 PETSc 线性问题。
    solution = problem.solve()  # 触发当前标签网格上的编译、组装与线性求解。
    solution.x.scatter_forward()  # 同步幽灵自由度后再提取结果。
    coordinates = ufl.SpatialCoordinate(domain)  # 创建当前网格的 UFL 坐标。
    exact_solution = coordinates[0] + coordinates[1]  # 定义线性解析补丁解 u=x+y。
    difference = solution - exact_solution  # 构造当前解误差表达式。
    qoi_value = assemble_real(solution * cell_measure(FEN014_CELL_IDS[0]) + solution * cell_measure(FEN014_CELL_IDS[1]))  # 分别装配左右子域并得到全域积分 QoI，解析值为一。
    error_density = ufl.inner(difference, difference)  # 定义不含 Measure 的 L2 误差密度。
    l2_error = math.sqrt(max(assemble_real(error_density * cell_measure(FEN014_CELL_IDS[0]) + error_density * cell_measure(FEN014_CELL_IDS[1])), 0.0))  # 在两个 cell tags 上装配解析解 L2 误差。
    energy_density = ufl.inner(ufl.grad(solution), ufl.grad(solution))  # 定义不含 Measure 的梯度能量密度。
    energy = assemble_real(energy_density * cell_measure(FEN014_CELL_IDS[0]) + energy_density * cell_measure(FEN014_CELL_IDS[1]))  # 在两个 cell tags 上装配梯度能量，解析值为二。
    dof_coordinates = function_space.tabulate_dof_coordinates()  # 获取当前 CG1 自由度几何坐标。
    nodal_exact = dof_coordinates[:, 0] + dof_coordinates[:, 1]  # 在每个自由度坐标计算解析线性值。
    local_max_nodal_error = float(np.max(np.abs(np.real(solution.x.array) - nodal_exact))) if nodal_exact.size else 0.0  # 计算当前 rank 最大节点误差。
    max_nodal_error = float(COMM.allreduce(local_max_nodal_error, op=MPI.MAX))  # 汇总全局最大节点误差。
    cell_areas = {str(tag_id): assemble_real(PETSc.ScalarType(1.0) * cell_measure(tag_id)) for tag_id in FEN014_CELL_IDS}  # 分别装配左右子域面积。
    boundary_lengths = {str(tag_id): assemble_real(PETSc.ScalarType(1.0) * facet_measure(tag_id)) for tag_id in FEN014_FACET_IDS}  # 分别装配四条标签边界长度。
    cell_counts = {str(tag_id): global_integer(int(cell_tags.find(tag_id).size)) for tag_id in FEN014_CELL_IDS}  # 汇总左右 cell tag 实体数量。
    facet_counts = {str(tag_id): global_integer(int(facet_tags.find(tag_id).size)) for tag_id in FEN014_FACET_IDS}  # 汇总四边 facet tag 实体数量。
    cell_geometry = tag_geometry_report(domain, cell_tags, FEN014_CELL_IDS)  # 生成左右 cell tag 与几何绑定的编号无关报告。
    facet_geometry = tag_geometry_report(domain, facet_tags, FEN014_FACET_IDS)  # 生成四边 facet tag 与几何绑定的编号无关报告。
    tagged_dofs_sorted = np.sort(np.unique(tagged_boundary_dofs))  # 规范化标签派生自由度集合。
    geometric_dofs_sorted = np.sort(np.unique(geometric_boundary_dofs))  # 规范化几何派生自由度集合。
    boundary_dof_sets_match = np.array_equal(tagged_dofs_sorted, geometric_dofs_sorted)  # 检查标签边界与独立几何边界映射一致。
    global_cells = int(domain.topology.index_map(topological_dimension).size_global)  # 读取当前网格全局单元数。
    global_vertices = int(domain.topology.index_map(0).size_global)  # 读取当前网格全局顶点数。
    global_facets = int(domain.topology.index_map(facet_dimension).size_global)  # 读取当前网格全局 facet 数。
    return {  # 返回参考与导入路径共用的完整数值诊断。
        "global_cells": global_cells,  # 保存全局单元数量。
        "global_vertices": global_vertices,  # 保存全局顶点数量。
        "global_facets": global_facets,  # 保存全局 facet 数量。
        "cell_tag_counts": cell_counts,  # 保存左右 cell 标签数量。
        "facet_tag_counts": facet_counts,  # 保存四边 facet 标签数量。
        "cell_tag_areas": cell_areas,  # 保存左右子域积分面积。
        "facet_tag_lengths": boundary_lengths,  # 保存四条边界积分长度。
        "cell_tag_geometry": cell_geometry,  # 保存 cell 标签几何摘要与中点均值。
        "facet_tag_geometry": facet_geometry,  # 保存 facet 标签几何摘要与中点均值。
        "tagged_boundary_dofs": global_integer(int(tagged_dofs_sorted.size)),  # 保存标签派生边界自由度数。
        "geometric_boundary_dofs": global_integer(int(geometric_dofs_sorted.size)),  # 保存几何派生边界自由度数。
        "boundary_dof_sets_match": bool(boundary_dof_sets_match),  # 保存两套映射集合一致性。
        "ksp_converged_reason": int(problem.solver.getConvergedReason()),  # 保存 PETSc 收敛原因码。
        "ksp_iterations": int(problem.solver.getIterationNumber()),  # 保存实际迭代次数。
        "qoi_domain_integral": qoi_value,  # 保存解析值为一的全域积分。
        "energy": energy,  # 保存解析值为二的梯度能量。
        "l2_error": l2_error,  # 保存解析解 L2 误差。
        "max_nodal_error": max_nodal_error,  # 保存最大节点值误差。
        "solution": solution,  # 暂存有限元函数供调用方写出，写 JSON 前会移除。
    }  # 结束当前网格数值诊断对象。


def run_fen014(output_root: Path) -> dict[str, Any]:  # 运行带 cell/facet tags 的 XDMF 往返差分数值测试。
    case_root = output_root / "fen-014"  # 为 FEN-014 合成基准使用独立 artifact 子目录。
    case_root.mkdir(parents=True, exist_ok=True)  # 在生成网格前创建目录以保留失败上下文。
    if COMM.size != 1:  # 当前轻量合同的实体计数和集合比较明确限定为单 MPI rank。
        raise RuntimeError("FEN-014 synthetic contract requires exactly one MPI rank.")  # 拒绝把含 ghost 的并行计数误写成全局事实。
    reference_domain = mesh.create_unit_square(COMM, FEN014_DIVISIONS, FEN014_DIVISIONS, cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.right)  # 生成确定性八乘八参考三角网格。
    reference_cell_tags, reference_facet_tags, tag_build_diagnostics = build_fen014_tags(reference_domain)  # 创建左右 cell tags 与四边 facet tags。
    reference_preflight = preflight_fen014(reference_domain, reference_cell_tags, reference_facet_tags)  # 在任何写出或求解前检查参考拓扑、尺度和标签语义。
    write_json(case_root / "reference_preflight.json", reference_preflight)  # 保存参考网格求解前门证据。
    if not bool(reference_preflight["passed"]):  # 参考网格预检失败时禁止继续导出或创建 LinearProblem。
        raise RuntimeError("FEN-014 reference mesh failed preflight before export or solve.")  # 明确失败阶段为参考预检。
    import_path = case_root / "fen014_tagged_import.xdmf"  # 定义模拟外部输入的 XDMF 文件路径。
    import_h5 = case_root / "fen014_tagged_import.h5"  # 定义 XDMF 配套 HDF5 数据文件路径。
    write_fen014_import_package(import_path, reference_domain, reference_cell_tags, reference_facet_tags)  # 将网格和两类标签真实写入 HDF5 XDMF。
    COMM.barrier()  # 等待全部 rank 关闭文件后再开始重读。
    import_files_before_read = {"xdmf": file_record(import_path), "h5": file_record(import_h5)}  # 在重读前冻结 XDMF 与 HDF5 的大小和哈希。
    import_files_exist = all(bool(record["exists"]) and int(record["size_bytes"]) > 0 for record in import_files_before_read.values())  # 要求两个输入文件均存在且非空。
    if not import_files_exist:  # 文件写出不完整时禁止进入网格读取和求解。
        raise RuntimeError("FEN-014 HDF5 XDMF export did not produce both non-empty files.")  # 明确失败阶段为导出证据门。
    imported_domain, imported_cell_tags, imported_facet_tags = read_fen014_import_package(import_path)  # 从磁盘创建全新网格和标签对象。
    import_files_after_read = {"xdmf": file_record(import_path), "h5": file_record(import_h5)}  # 在重读关闭后再次计算输入包大小和哈希。
    imported_preflight = preflight_fen014(imported_domain, imported_cell_tags, imported_facet_tags)  # 在任何重读网格求解前执行独立预检。
    write_json(case_root / "imported_preflight.json", imported_preflight)  # 保存重读网格求解前门证据。
    preflight_checks: list[dict[str, Any]] = []  # 创建参考到重读差分的求解许可门。
    add_check(preflight_checks, "reference_preflight_passed", bool(reference_preflight["passed"]), reference_preflight["checks"])  # 要求参考网格全部预检通过。
    add_check(preflight_checks, "imported_preflight_passed", bool(imported_preflight["passed"]), imported_preflight["checks"])  # 要求重读网格全部预检通过。
    add_check(preflight_checks, "input_files_unchanged_by_read", import_files_before_read == import_files_after_read, {"before": import_files_before_read, "after": import_files_after_read})  # 要求只读导入不改写 XDMF/HDF5 字节。
    add_check(preflight_checks, "cell_tag_geometry_roundtrip_exact", reference_preflight["cell_tag_geometry"]["signature_sha256"] == imported_preflight["cell_tag_geometry"]["signature_sha256"], {"reference": reference_preflight["cell_tag_geometry"]["signature_sha256"], "imported": imported_preflight["cell_tag_geometry"]["signature_sha256"]})  # 要求 cell 标签值与实体几何绑定精确保持。
    add_check(preflight_checks, "facet_tag_geometry_roundtrip_exact", reference_preflight["facet_tag_geometry"]["signature_sha256"] == imported_preflight["facet_tag_geometry"]["signature_sha256"], {"reference": reference_preflight["facet_tag_geometry"]["signature_sha256"], "imported": imported_preflight["facet_tag_geometry"]["signature_sha256"]})  # 要求 facet 标签值与实体几何绑定精确保持。
    add_check(preflight_checks, "orientation_histogram_preserved", reference_preflight["orientation_histogram"] == imported_preflight["orientation_histogram"], {"reference": reference_preflight["orientation_histogram"], "imported": imported_preflight["orientation_histogram"]})  # 比较重读前后有向面积符号分布。
    preflight_receipt = {"passed": checks_pass(preflight_checks), "checks": preflight_checks, "reference": reference_preflight, "imported": imported_preflight, "input_files_before_read": import_files_before_read, "input_files_after_read": import_files_after_read, "solve_calls_before_gate": 0}  # 组装明确零求解的差分预检回执。
    write_json(case_root / "preflight_receipt.json", preflight_receipt)  # 在允许求解前写出完整预检证据。
    if not bool(preflight_receipt["passed"]):  # 任一差分预检失败时保持零求解并停止。
        raise RuntimeError("FEN-014 imported mesh failed preflight; solve was not started.")  # 明确拒绝机械进入求解阶段。
    reference_result = solve_fen014_patch(reference_domain, reference_cell_tags, reference_facet_tags, "fen014_reference_")  # 在内存参考网格上真实组装与求解。
    imported_result = solve_fen014_patch(imported_domain, imported_cell_tags, imported_facet_tags, "fen014_imported_")  # 在重读网格上独立重建空间、Measure 并求解。
    imported_solution = imported_result.pop("solution")  # 从 JSON 诊断中取出不可序列化的有限元函数。
    reference_result.pop("solution")  # 移除参考结果中的不可序列化有限元函数。
    imported_solution.name = "fen014_imported_patch_solution"  # 设置重读网格解的 XDMF 字段名称。
    solution_path = case_root / "fen014_imported_solution.xdmf"  # 定义重读求解场输出路径。
    with io.XDMFFile(COMM, solution_path, "w", encoding=io.XDMFFile.Encoding.HDF5) as solution_file:  # 显式创建 HDF5 编码的重读解文件。
        solution_file.write_mesh(imported_domain)  # 写出重读后的网格对象。
        solution_file.write_function(imported_solution)  # 写出在重读网格上真实求得的解。
    checks: list[dict[str, Any]] = []  # 创建 FEN-014 预冻结验收检查列表。
    expected_topology = {"global_vertices": 81, "global_cells": 128, "global_facets": 208}  # 冻结八乘八右对角网格的拓扑规模。
    add_check(checks, "reference_topology_exact", all(int(reference_result[key]) == value for key, value in expected_topology.items()), {"expected": expected_topology, "actual": {key: reference_result[key] for key in expected_topology}})  # 核对参考网格规模。
    add_check(checks, "imported_topology_exact", all(int(imported_result[key]) == value for key, value in expected_topology.items()), {"expected": expected_topology, "actual": {key: imported_result[key] for key in expected_topology}})  # 核对重读网格规模。
    expected_cell_counts = {"1": 64, "2": 64}  # 冻结左右半区各六十四个三角形。
    expected_facet_counts = {"11": 8, "12": 8, "13": 8, "14": 8}  # 冻结四边各八条边界 facet。
    add_check(checks, "cell_tag_counts_preserved", reference_result["cell_tag_counts"] == expected_cell_counts and imported_result["cell_tag_counts"] == expected_cell_counts, {"expected": expected_cell_counts, "reference": reference_result["cell_tag_counts"], "imported": imported_result["cell_tag_counts"]})  # 检查 cell tags 数量保持。
    add_check(checks, "facet_tag_counts_preserved", reference_result["facet_tag_counts"] == expected_facet_counts and imported_result["facet_tag_counts"] == expected_facet_counts, {"expected": expected_facet_counts, "reference": reference_result["facet_tag_counts"], "imported": imported_result["facet_tag_counts"]})  # 检查 facet tags 数量保持。
    add_check(checks, "cell_tag_geometry_preserved", reference_result["cell_tag_geometry"]["signature_sha256"] == imported_result["cell_tag_geometry"]["signature_sha256"], {"reference": reference_result["cell_tag_geometry"], "imported": imported_result["cell_tag_geometry"]})  # 检查左右标签绑定到相同几何实体而非只保持数量。
    add_check(checks, "facet_tag_geometry_preserved", reference_result["facet_tag_geometry"]["signature_sha256"] == imported_result["facet_tag_geometry"]["signature_sha256"], {"reference": reference_result["facet_tag_geometry"], "imported": imported_result["facet_tag_geometry"]})  # 检查四边标签值与几何位置绑定关系保持。
    expected_cell_midpoints = {"1": [0.25, 0.5], "2": [0.75, 0.5]}  # 冻结左右半区三角形中点的全局平均坐标。
    expected_facet_midpoints = {"11": [0.0, 0.5], "12": [1.0, 0.5], "13": [0.5, 0.0], "14": [0.5, 1.0]}  # 冻结四条边界中点的全局平均坐标。
    add_check(checks, "cell_tag_semantics_exact", all(midpoint_means_match(result["cell_tag_geometry"], expected_cell_midpoints) for result in (reference_result, imported_result)), {"limit": FEN014_MEASURE_LIMIT, "expected": expected_cell_midpoints, "reference": reference_result["cell_tag_geometry"]["midpoint_means"], "imported": imported_result["cell_tag_geometry"]["midpoint_means"]})  # 检查一确为左半区且二确为右半区，并安全拒绝空标签。
    add_check(checks, "facet_tag_semantics_exact", all(midpoint_means_match(result["facet_tag_geometry"], expected_facet_midpoints) for result in (reference_result, imported_result)), {"limit": FEN014_MEASURE_LIMIT, "expected": expected_facet_midpoints, "reference": reference_result["facet_tag_geometry"]["midpoint_means"], "imported": imported_result["facet_tag_geometry"]["midpoint_means"]})  # 检查十一至十四分别对应左、右、下、上，并安全拒绝空标签。
    add_check(checks, "tagged_boundary_covers_geometry", bool(tag_build_diagnostics["tagged_boundary_matches_geometry"]) and int(tag_build_diagnostics["local_duplicate_tagged_facets"]) == 0, tag_build_diagnostics)  # 检查标签恰好覆盖全部外边界且无重复。
    add_check(checks, "boundary_dof_mapping_preserved", bool(reference_result["boundary_dof_sets_match"]) and bool(imported_result["boundary_dof_sets_match"]) and int(reference_result["tagged_boundary_dofs"]) == 32 and int(imported_result["tagged_boundary_dofs"]) == 32, {"reference": reference_result["tagged_boundary_dofs"], "imported": imported_result["tagged_boundary_dofs"]})  # 检查标签到自由度映射。
    add_check(checks, "cell_measures_exact", all(abs(float(result["cell_tag_areas"][tag_id]) - 0.5) <= FEN014_MEASURE_LIMIT for result in (reference_result, imported_result) for tag_id in ("1", "2")), {"limit": FEN014_MEASURE_LIMIT, "reference": reference_result["cell_tag_areas"], "imported": imported_result["cell_tag_areas"]})  # 检查左右面积均为零点五。
    add_check(checks, "facet_measures_exact", all(abs(float(result["facet_tag_lengths"][tag_id]) - 1.0) <= FEN014_MEASURE_LIMIT for result in (reference_result, imported_result) for tag_id in ("11", "12", "13", "14")), {"limit": FEN014_MEASURE_LIMIT, "reference": reference_result["facet_tag_lengths"], "imported": imported_result["facet_tag_lengths"]})  # 检查四边长度均为一。
    add_check(checks, "both_ksp_solves_converged", int(reference_result["ksp_converged_reason"]) > 0 and int(imported_result["ksp_converged_reason"]) > 0, {"reference": reference_result["ksp_converged_reason"], "imported": imported_result["ksp_converged_reason"]})  # 要求参考和重读两次 PETSc 求解均收敛。
    add_check(checks, "patch_solution_exact", all(float(result["l2_error"]) <= FEN014_EXACTNESS_LIMIT and float(result["max_nodal_error"]) <= FEN014_EXACTNESS_LIMIT for result in (reference_result, imported_result)), {"limit": FEN014_EXACTNESS_LIMIT, "reference_l2": reference_result["l2_error"], "imported_l2": imported_result["l2_error"], "reference_node": reference_result["max_nodal_error"], "imported_node": imported_result["max_nodal_error"]})  # 检查线性补丁解达到数值精度。
    add_check(checks, "qoi_and_energy_exact", all(abs(float(result["qoi_domain_integral"]) - 1.0) <= FEN014_EXACTNESS_LIMIT and abs(float(result["energy"]) - 2.0) <= FEN014_EXACTNESS_LIMIT for result in (reference_result, imported_result)), {"limit": FEN014_EXACTNESS_LIMIT, "reference_qoi": reference_result["qoi_domain_integral"], "imported_qoi": imported_result["qoi_domain_integral"], "reference_energy": reference_result["energy"], "imported_energy": imported_result["energy"]})  # 检查解析 QoI 与能量。
    add_check(checks, "reference_imported_differential", abs(float(reference_result["qoi_domain_integral"]) - float(imported_result["qoi_domain_integral"])) <= FEN014_EXACTNESS_LIMIT and abs(float(reference_result["energy"]) - float(imported_result["energy"])) <= FEN014_EXACTNESS_LIMIT, {"limit": FEN014_EXACTNESS_LIMIT, "qoi_delta": abs(float(reference_result["qoi_domain_integral"]) - float(imported_result["qoi_domain_integral"])), "energy_delta": abs(float(reference_result["energy"]) - float(imported_result["energy"]))})  # 检查往返前后聚合结果一致。
    case_success = checks_pass(checks)  # 只有拓扑、标签、Measure、求解和差分检查全部通过才接受。
    solution_h5 = case_root / "fen014_imported_solution.h5"  # 定位重读解的 HDF5 数据文件。
    receipt = {  # 构造 FEN-014 合成 XDMF 数值合同回执。
        "schema_version": "v5-fenicsx-synthetic-case/1.0",  # 冻结单案例回执版本。
        "case_id": "FEN-014",  # 记录所覆盖的能力参考案例。
        "benchmark_id": "SYN-FEN-014-XDMF-MESHTAGS-ROUNDTRIP",  # 使用 SYN 前缀防止冒充真实外部网格。
        "artifact_kind": "synthetic_numerical_contract_evidence",  # 明确证据类别。
        "execution_family": "FEniCS/FEniCSx",  # 记录实际执行技术族。
        "contract_test_outcome": "pass" if case_success else "fail",  # 记录合成数值合同结果。
        "research_case_execution_status": "not_executed_missing_current_evidence",  # 保留原 FEN-014 尚未执行状态。
        "scientific_claim_allowed": False,  # 禁止形成原研究或工程结论。
        "synthetic_numeric_results_generated": True,  # 明确确实执行了网格读写、组装和求解。
        "problem_contract": {"domain": "unit_square", "pde": "minus_laplacian_u_equals_zero", "exact_solution": "x+y", "boundary": "dirichlet_from_imported_facet_tags", "element": "continuous_lagrange_degree_1", "cell_measure": "dx(1)+dx(2)", "cell_tags": {"1": "left_half", "2": "right_half"}, "facet_tags": {"11": "left", "12": "right", "13": "bottom", "14": "top"}},  # 保存标签化补丁问题合同。
        "qoi_contract": {"quantity": "domain_integral_of_scalar_solution", "exact_value": 1.0, "unit": "dimensionless", "spatial_selection": "cell_tags_1_and_2", "coordinate_frame": "cartesian_xy", "analysis_step": "steady", "reduction": "MPI_SUM", "comparison_rule": "reference_and_roundtrip_each_within_1e-10_and_mutually_equal", "singularity_exposure": "none_by_design"},  # 保存参考与重读路径共用的固定 QoI 定义。
        "xdmf_contract": {"encoding": "HDF5", "mesh_name": FEN014_MESH_NAME, "cell_tag_name": FEN014_CELL_TAG_NAME, "facet_tag_name": FEN014_FACET_TAG_NAME, "roundtrip_requires_fresh_mesh_space_measures": True},  # 保存按名称往返合同。
        "preflight": preflight_receipt,  # 保存先预检后求解的完整质量门证据。
        "reference": reference_result,  # 保存内存参考网格诊断。
        "imported": imported_result,  # 保存重读网格诊断。
        "checks": checks,  # 保存全部预冻结检查结果。
        "runtime": runtime_identity(),  # 保存实际 FEniCSx 与 PETSc 运行时。
        "provenance": source_identity(),  # 保存当前提交与合成输入来源。
        "files": {"import_xdmf": artifact_file_record(import_path, output_root), "import_h5": artifact_file_record(import_h5, output_root), "solution_xdmf": artifact_file_record(solution_path, output_root), "solution_h5": artifact_file_record(solution_h5, output_root)},  # 保存输入包与重读解全部文件的相对路径、字节数和哈希。
        "execution_counts": {"mesh_generation_calls": 1, "mesh_export_calls": 1, "mesh_import_calls": 1, "linear_solve_calls": 2, "qoi_extraction_calls": 2, "calculix_calls": 0, "model_calls": 0},  # 记录真实调用边界。
        "parallel_scope": "serial_only",  # 明确当前实体计数和集合门只在单 MPI rank 下获准使用。
        "allowed_use": ["验证固定 DOLFINx 环境能保持 XDMF/HDF5 中命名 MeshTags 并完成重读求解。", "作为 FEN-014 后续真实外部网格诊断的回归基线。"],  # 限定允许用途。
        "disallowed_use": ["不得宣称原 FEN-014 外部网格已读取、修复或通过。", "不得删除或满足 fen-014.json 中七项 missing_facts。", "不得据此形成工程设计或论文数值结论。"],  # 限定禁止用途。
    }  # 结束 FEN-014 回执。
    write_json(case_root / "numerical_contract_receipt.json", receipt)  # 写出单案例机器回执。
    return receipt  # 返回整体活动汇总。


def failure_case(case_id: str, benchmark_id: str, error: BaseException) -> dict[str, Any]:  # 将任一案例异常转换为可上传失败回执。
    return {  # 返回不冒充数值通过的失败对象。
        "schema_version": "v5-fenicsx-synthetic-case/1.0",  # 保持与成功路径相同合同版本。
        "case_id": case_id,  # 记录失败能力参考案例。
        "benchmark_id": benchmark_id,  # 记录失败合成基准标识。
        "artifact_kind": "synthetic_numerical_contract_evidence",  # 保持证据类型明确。
        "execution_family": "FEniCS/FEniCSx",  # 保持 FEN 到 FEniCSx 的正确映射。
        "contract_test_outcome": "fail",  # 明确案例合同失败。
        "research_case_execution_status": "not_executed_missing_current_evidence",  # 原研究案例状态不受异常影响。
        "scientific_claim_allowed": False,  # 失败路径同样禁止科学结论。
        "synthetic_numeric_results_generated": False,  # 异常路径不宣称完整数值结果已生成。
        "error": type(error).__name__ + ": " + str(error),  # 保存异常类型与消息。
        "traceback": traceback.format_exc(),  # 保存完整 Python traceback 供 Actions 调试。
        "runtime": runtime_identity(),  # 保存失败发生时的真实运行时身份。
        "provenance": source_identity(),  # 保存失败对应的不可变提交来源。
    }  # 结束失败回执。


def main() -> int:  # 依次尝试两个真实数值合同并生成整体活动回执。
    args = parse_args()  # 读取工作流显式传入的输出目录。
    output_root = Path(args.output_dir).resolve()  # 将 artifact 根解析为绝对路径。
    output_root.mkdir(parents=True, exist_ok=True)  # 在任何求解前创建活动目录。
    started_at = utc_now()  # 记录整体活动开始时间。
    receipts: list[dict[str, Any]] = []  # 收集两个案例的成功或失败回执。
    try:  # 独立运行 FEN-003，使另一案例不因其异常而缺少证据。
        fen003_receipt = run_fen003(output_root)  # 执行四层制造解网格收敛案例。
    except BaseException as error:  # 捕获 FEniCSx、PETSc、I/O 和 Python 的全部异常。
        fen003_receipt = failure_case("FEN-003", "SYN-FEN-003-MMS-POISSON-P1", error)  # 构造明确失败回执。
        write_json(output_root / "fen-003" / "numerical_contract_receipt.json", fen003_receipt)  # 始终保存失败证据。
    receipts.append(fen003_receipt)  # 将 FEN-003 结果加入整体活动。
    try:  # 独立运行 FEN-014，即使 FEN-003 失败仍尝试留下差分证据。
        fen014_receipt = run_fen014(output_root)  # 执行 XDMF 与 MeshTags 往返求解案例。
    except BaseException as error:  # 捕获导出、重读、标签、求解和提取异常。
        fen014_receipt = failure_case("FEN-014", "SYN-FEN-014-XDMF-MESHTAGS-ROUNDTRIP", error)  # 构造明确失败回执。
        write_json(output_root / "fen-014" / "numerical_contract_receipt.json", fen014_receipt)  # 始终保存失败证据。
    receipts.append(fen014_receipt)  # 将 FEN-014 结果加入整体活动。
    campaign_success = len(receipts) == 2 and all(receipt.get("contract_test_outcome") == "pass" for receipt in receipts)  # 两个合同均通过时整体才成功。
    campaign = {  # 构造一次 Actions 数值活动的主索引回执。
        "schema_version": "v5-fenicsx-synthetic-campaign/1.0",  # 冻结整体活动合同版本。
        "status": "synthetic_contract_passed" if campaign_success else "synthetic_contract_failed",  # 明确状态仅表示合成数值合同，不表示原研究案例成功。
        "contract_test_outcome": "pass" if campaign_success else "fail",  # 提供供独立验证器消费的稳定通过或失败枚举。
        "artifact_kind": "synthetic_numerical_contract_evidence",  # 防止 artifact 被解释为原案例结果。
        "execution_family": "FEniCS/FEniCSx",  # 冻结两个 FEN 案例的执行技术族。
        "started_at_utc": started_at,  # 保存活动开始时间。
        "finished_at_utc": utc_now(),  # 保存活动结束时间。
        "runtime": runtime_identity(),  # 保存统一运行时身份。
        "provenance": source_identity(),  # 保存统一 GitHub 来源。
        "case_summaries": [{"case_id": receipt.get("case_id"), "benchmark_id": receipt.get("benchmark_id"), "contract_test_outcome": receipt.get("contract_test_outcome"), "research_case_execution_status": receipt.get("research_case_execution_status"), "scientific_claim_allowed": receipt.get("scientific_claim_allowed")} for receipt in receipts],  # 提供两个案例的精简索引。
        "execution_counts": {"mesh_generation_calls": 5, "mesh_export_calls": 1, "mesh_import_calls": 1, "linear_solve_calls": 6, "calculix_calls": 0, "model_calls": 0} if campaign_success else {"completed_counts_are_recorded_per_case": True, "calculix_calls": 0, "model_calls": 0},  # 成功时记录确定调用数，失败时避免虚构未完成调用。
        "uses_original_research_inputs": False,  # 明确全部输入在 CI 内合成。
        "research_case_execution_status": "not_executed_missing_current_evidence",  # 在活动顶层重复声明原研究案例仍未执行。
        "research_cases_remain_blocked": True,  # 明确原 profiles 的缺失事实仍有效。
        "scientific_claim_allowed": False,  # 整体活动不能支持工程或论文结论。
        "success_meaning": "两个自包含 DOLFINx 合成基准均真实完成网格、形式、组装、求解和数值检查。",  # 限定绿色 Actions 的唯一含义。
    }  # 结束整体活动回执。
    write_json(output_root / "campaign_receipt.json", campaign)  # 写出主索引供 artifact 审计。
    if COMM.rank == ROOT_RANK:  # 仅由零号进程输出一行精简日志。
        print(json.dumps({"status": campaign["status"], "case_summaries": campaign["case_summaries"], "output_dir": str(output_root)}, ensure_ascii=False))  # 在 Actions 日志打印结果和 artifact 位置。
    return 0 if campaign_success else 1  # 将真实数值合同结果传递给 GitHub Actions。


if __name__ == "__main__":  # 仅在工作流直接执行脚本时运行入口。
    raise SystemExit(main())  # 使用整体活动状态设置进程退出码。
