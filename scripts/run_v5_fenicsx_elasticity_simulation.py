#!/usr/bin/env python3
# 上一行让固定容器使用其 Python 3 解释器；本脚本只生成自包含 DOLFINx 演示证据，不形成科学或工程结论。
"""运行二维平面应力悬臂板的四层 DOLFINx 真实有限元演示，并写出严格可审计工件。"""  # 模块说明冻结本文件的演示性质与主要输出。
from __future__ import annotations  # 启用延迟类型注解，以兼容运行容器并避免类型对象提前求值。

import argparse  # 解析调用方显式提供的工件输出目录。
import hashlib  # 为所有关键工件计算 SHA-256 内容摘要。
import json  # 写出禁止 NaN 与 Infinity 的严格 JSON 报告。
import math  # 提供有限性判断、平方根及相对差计算所需数学函数。
import os  # 读取非敏感的 CI 与固定镜像身份环境变量。
import platform  # 记录执行容器的操作系统与处理器平台身份。
import sys  # 记录 Python 版本并设置进程成功或失败退出码。
import traceback  # 在失败回执中保留完整异常调用栈以便复核。
from datetime import datetime, timezone  # 生成带 UTC 时区的可追溯时间戳。
from pathlib import Path  # 使用跨平台路径对象管理 XDMF、HDF5、PNG 与 JSON 工件。
from typing import Any  # 为可嵌套 JSON 数据和 UFL 表达式提供通用类型注解。

import matplotlib  # 导入绘图库主模块，以便在导入 pyplot 前冻结无界面后端。
matplotlib.use("Agg")  # 使用无显示服务器的 Agg 后端，保证官方容器可生成真实 PNG。
import matplotlib.pyplot as plt  # 绘制真实网格、位移、应力与收敛图。
import matplotlib.tri as mtri  # 用 DOLFINx 三角形拓扑建立 Matplotlib 三角剖分。
from mpi4py import MPI  # 使用 DOLFINx 所需 MPI 通信器并执行全局积分归约。
from petsc4py import PETSc  # 使用 PETSc 标量类型、向量、矩阵与 KSP 收敛状态。
import basix  # 记录实际有限元基函数库版本以支持复现。
import dolfinx  # 记录并核验实际 DOLFINx 版本必须为 0.11.0.post0。
import ffcx  # 记录实际变分形式编译器版本以支持复现。
import numpy as np  # 构造网格标签、处理解向量并计算诊断范数。
import ufl  # 定义小变形线弹性、平面应力与后处理表达式。
from dolfinx import fem, io, mesh  # 创建函数空间、标签、网格以及 XDMF/HDF5 工件。
from dolfinx.fem import petsc as fem_petsc  # 装配未施加边界条件的 PETSc 矩阵和向量。
from dolfinx.fem.petsc import LinearProblem  # 用 preonly 与 LU 组装并求解线性系统。

COMM = MPI.COMM_WORLD  # 使用容器全局通信器；本演示随后强制其大小为一。
ROOT_RANK = 0  # 将共享文件写入限制到零号进程以避免文件竞争。
LENGTH_M = 4.0  # 冻结板长为 4 米，横向坐标范围为零到四米。
HEIGHT_M = 1.0  # 冻结板高为 1 米，纵向坐标范围为零到一米。
THICKNESS_M = 0.01  # 冻结平面应力厚度为 0.01 米，用于刚度与线荷载换算。
YOUNG_MODULUS_PA = 210.0e9  # 冻结杨氏模量为 210 GPa，对应常用钢材演示值但非材料签认值。
POISSON_RATIO = 0.3  # 冻结泊松比为 0.3，满足各向同性线弹性允许区间。
TRACTION_Y_PA = -1.0e6  # 冻结右边界竖向均布面力为负一兆帕，负号表示向下。
MESH_LEVELS = ((16, 4), (32, 8), (64, 16), (128, 32))  # 冻结四层二倍加密矩形网格分辨率。
REGULAR_CELL_TAG = 1  # 用标签一表示探针条带之外的普通单元。
PROBE_CELL_TAG = 2  # 用标签二表示固定物理探针条带单元。
LEFT_FACET_TAG = 11  # 用标签十一表示零位移全固定左边界。
RIGHT_FACET_TAG = 12  # 用标签十二表示施加向下载荷的右边界。
BOTTOM_FACET_TAG = 13  # 用标签十三表示自由下边界。
TOP_FACET_TAG = 14  # 用标签十四表示自由上边界。
PROBE_X_MIN_M = 7.0 * LENGTH_M / 16.0  # 冻结探针条带左界为七个十六分之一板长，即 1.75 米。
PROBE_X_MAX_M = 9.0 * LENGTH_M / 16.0  # 冻结探针条带右界为九个十六分之一板长，即 2.25 米。
RESIDUAL_LIMIT = 1.0e-8  # 冻结力、力矩、自由自由度残差和能量一致性相对门限。
TIP_CHANGE_LIMIT = 2.0e-2  # 冻结最后两层右边平均竖向位移相对变化门限为百分之二。
ENERGY_CHANGE_LIMIT = 2.0e-2  # 冻结最后两层应变能相对变化门限为百分之二。
PROBE_STRESS_CHANGE_LIMIT = 3.0e-2  # 冻结最后两层探针平均等效应力相对变化门限为百分之三。
ROUNDTRIP_LIMIT = 1.0e-10  # 冻结 XDMF/HDF5 关闭重读独立重算指标差异门限。
MESH_NAME = "cantilever_plane_stress_mesh"  # 冻结 XDMF 中可按名称读取的网格名称。
CELL_TAG_NAME = "cantilever_cell_tags"  # 冻结 XDMF 中可按名称读取的单元标签名称。
FACET_TAG_NAME = "cantilever_facet_tags"  # 冻结 XDMF 中可按名称读取的边标签名称。
SCRIPT_RELATIVE_PATH = "scripts/run_v5_fenicsx_elasticity_simulation.py"  # 冻结回执中的仓库相对入口路径。


def utc_now() -> str:  # 返回带显式 UTC 时区的 ISO 8601 时间字符串。
    return datetime.now(timezone.utc).isoformat()  # 使用执行机真实时钟生成可排序时间戳。


def parse_args() -> argparse.Namespace:  # 解析唯一必需参数 output-dir，其值是全部工件的根目录。
    parser = argparse.ArgumentParser(description="Run the DOLFINx 0.11 cantilever plane-stress demonstration.")  # 创建不暗示科研完成的命令行解析器。
    parser.add_argument("--output-dir", required=True, help="Directory for JSON, XDMF/HDF5, and PNG artifacts.")  # 要求调用方明确指定可审计输出位置。
    return parser.parse_args()  # 返回完成类型与必填检查的参数对象。


def validate_json_value(value: Any, location: str = "root") -> None:  # 递归验证 JSON 值；location 用于精确报告非法值路径。
    if isinstance(value, float):  # 仅浮点数可能携带严格 JSON 禁止的 NaN 或 Infinity。
        if not math.isfinite(value):  # 非有限值会破坏机器消费与门禁比较语义。
            raise ValueError(f"non-finite JSON value at {location}: {value}")  # 立即拒绝非法回执并进入失败路径。
        return  # 有限浮点数无需继续递归。
    if isinstance(value, dict):  # 字典需要检查键类型并递归检查每个值。
        for key, nested_value in value.items():  # 逐项保留完整字段路径以便定位错误。
            if not isinstance(key, str):  # 严格 JSON 对象键只能是字符串。
                raise TypeError(f"non-string JSON key at {location}: {key!r}")  # 拒绝可能被编码器隐式改写的键。
            validate_json_value(nested_value, f"{location}.{key}")  # 递归检查当前字段的嵌套值。
        return  # 当前字典全部字段通过后结束本层检查。
    if isinstance(value, (list, tuple)):  # 列表与元组需要逐索引验证其成员。
        for index, nested_value in enumerate(value):  # 为每个成员构造稳定的索引路径。
            validate_json_value(nested_value, f"{location}[{index}]")  # 递归拒绝深层非有限浮点数。


def write_json(path: Path, payload: dict[str, Any]) -> None:  # 将 payload 写为 UTF-8 严格 JSON；path 是目标文件。
    if COMM.rank != ROOT_RANK:  # 非零进程不得写共享文件；虽然本演示强制串行仍保留保护。
        return  # 跳过非根进程写入以防未来误用。
    validate_json_value(payload)  # 在序列化前显式拒绝任何非有限值或非法对象键。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建目标父目录，使成功和失败路径都可留证。
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"  # 固定缩进、键排序、UTF-8 字符和严格有限数规则。
    path.write_text(text, encoding="utf-8")  # 一次性写入完整文档并保留末尾换行。


def sha256_file(path: Path) -> str:  # 计算 path 指向文件的完整 SHA-256 十六进制摘要。
    digest = hashlib.sha256()  # 为当前文件创建独立摘要状态。
    with path.open("rb") as source:  # 以二进制只读模式避免平台文本转换改变内容。
        for chunk in iter(lambda: source.read(1024 * 1024), b""):  # 每次读取一 MiB 以限制峰值内存。
            digest.update(chunk)  # 将当前原始字节块加入摘要。
    return digest.hexdigest()  # 返回六十四字符摘要供回执和独立验证使用。


def file_record(path: Path, output_root: Path) -> dict[str, Any]:  # 返回工件相对路径、字节数与 SHA-256；两参数定义文件及根目录。
    resolved_path = path.resolve()  # 解析文件绝对位置以验证其归属和稳定记录。
    relative_path = resolved_path.relative_to(output_root.resolve())  # 拒绝输出根目录之外的意外工件。
    if not resolved_path.is_file():  # 缺失文件不能被宣称为已生成证据。
        raise FileNotFoundError(f"required artifact is missing: {resolved_path}")  # 将缺失转入失败回执。
    size_bytes = int(resolved_path.stat().st_size)  # 读取真实文件字节数作为非空门禁依据。
    return {"path": relative_path.as_posix(), "size_bytes": size_bytes, "sha256": sha256_file(resolved_path)}  # 返回跨平台相对路径与内容身份。


def runtime_identity() -> dict[str, Any]:  # 记录求解器、编译器、基函数库、Python、平台和 MPI 的实际身份。
    return {  # 返回可直接嵌入成功或失败回执的身份对象。
        "execution_family": "FEniCS/FEniCSx",  # 明确真正调用的是 FEniCSx 技术族而非 CalculiX。
        "dolfinx": str(dolfinx.__version__),  # 记录实际 DOLFINx Python 包版本。
        "basix": str(getattr(basix, "__version__", "unknown")),  # 记录实际 Basix 版本，缺失时写 unknown。
        "ufl": str(getattr(ufl, "__version__", "unknown")),  # 记录实际 UFL 版本，缺失时写 unknown。
        "ffcx": str(getattr(ffcx, "__version__", "unknown")),  # 记录实际 FFCx 版本，缺失时写 unknown。
        "petsc": ".".join(str(value) for value in PETSc.Sys.getVersion()),  # 记录 PETSc 主次补丁版本。
        "python": sys.version,  # 记录完整 Python 构建版本字符串。
        "platform": platform.platform(),  # 记录操作系统、内核和架构摘要。
        "mpi_size": int(COMM.size),  # 记录通信器进程数并供串行限制审计。
        "runtime_image": os.environ.get("FENICSX_RUNTIME_IMAGE", "official-dolfinx/dolfinx:v0.11.0"),  # 记录调用方镜像身份或约定的官方镜像标签。
    }  # 结束运行时身份对象。


def source_identity() -> dict[str, Any]:  # 记录实际执行脚本及可用 CI 来源，不读取任何密钥。
    script_path = Path(__file__).resolve()  # 解析当前脚本真实位置以计算不可歧义的哈希。
    return {  # 返回来源身份对象。
        "runner_path": SCRIPT_RELATIVE_PATH,  # 记录固定仓库相对路径供审计者定位代码。
        "runner_sha256": sha256_file(script_path),  # 记录实际执行字节的 SHA-256。
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),  # 记录可用 GitHub 仓库标识。
        "ref": os.environ.get("GITHUB_REF", ""),  # 记录可用分支或标签完整引用。
        "head_sha": os.environ.get("GITHUB_SHA", ""),  # 记录可用不可变提交 SHA。
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),  # 记录可用 Actions 运行编号。
        "input_provenance": "script_generated_frozen_demonstration",  # 明确几何、材料与载荷由本脚本冻结生成。
        "uses_original_research_inputs": False,  # 明确未读取任何原研究案例输入。
    }  # 结束来源身份对象。


def assemble_real(expression: Any) -> float:  # 装配 UFL 标量 expression 并返回跨 MPI 的实数总和。
    local_value = fem.assemble_scalar(fem.form(expression))  # 编译并装配本地单元或边界贡献。
    global_value = COMM.allreduce(local_value, op=MPI.SUM)  # 将所有进程贡献求和；串行时保持原值。
    return float(np.real(global_value))  # 丢弃理论零虚部并规范为可写 JSON 的 Python 浮点数。


def relative_difference(first: float, second: float) -> float:  # 计算两有限标量的对称用途相对差；参数单位必须相同。
    denominator = max(abs(float(first)), abs(float(second)), 1.0e-300)  # 用两者量级和极小正数避免除零。
    return abs(float(second) - float(first)) / denominator  # 返回无量纲非负相对差。


def strain(displacement: Any) -> Any:  # 返回二维小变形张量；displacement 是二维位移场或试函数。
    return ufl.sym(ufl.grad(displacement))  # 取位移梯度对称部分并忽略几何非线性项。


def stress(displacement: Any) -> Any:  # 返回各向同性平面应力张量；displacement 是二维位移场或试函数。
    shear_modulus = YOUNG_MODULUS_PA / (2.0 * (1.0 + POISSON_RATIO))  # 按冻结 E 与 nu 计算剪切模量，单位 Pa。
    plane_stress_lambda = YOUNG_MODULUS_PA * POISSON_RATIO / (1.0 - POISSON_RATIO**2)  # 计算平面应力等效 Lamé 系数，单位 Pa。
    epsilon = strain(displacement)  # 计算传入位移的无量纲小应变张量。
    return 2.0 * shear_modulus * epsilon + plane_stress_lambda * ufl.tr(epsilon) * ufl.Identity(2)  # 返回二维 Cauchy 应力，单位 Pa。


def von_mises(displacement: Any) -> Any:  # 返回二维平面应力 von Mises 标量；输入为二维位移场。
    sigma = stress(displacement)  # 先按同一平面应力本构计算二维应力张量。
    return ufl.sqrt(sigma[0, 0] ** 2 - sigma[0, 0] * sigma[1, 1] + sigma[1, 1] ** 2 + 3.0 * sigma[0, 1] ** 2)  # 使用平面应力等效应力公式，单位 Pa。


def build_mesh_and_tags(nx: int, ny: int) -> tuple[Any, Any, Any]:  # 以 nx、ny 分割数创建矩形三角网格及完整单元和边界标签。
    points = [np.array([0.0, 0.0], dtype=np.float64), np.array([LENGTH_M, HEIGHT_M], dtype=np.float64)]  # 定义矩形左下和右上角，单位米。
    domain = mesh.create_rectangle(COMM, points, [nx, ny], cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.right)  # 用确定性右对角线生成 P1 三角形拓扑。
    domain.name = MESH_NAME  # 设置稳定网格名称供 XDMF 按名回读。
    topological_dimension = domain.topology.dim  # 读取二维单元拓扑维数。
    facet_dimension = topological_dimension - 1  # 边界 facet 在二维网格中是一维边。
    cell_map = domain.topology.index_map(topological_dimension)  # 获取本地拥有与幽灵单元的索引映射。
    local_cell_count = int(cell_map.size_local + cell_map.num_ghosts)  # 标签覆盖所有当前进程可见单元。
    cell_indices = np.arange(local_cell_count, dtype=np.int32)  # 构造稳定升序本地单元索引。
    midpoint_coordinates = mesh.compute_midpoints(domain, topological_dimension, cell_indices)  # 计算单元物理中点以定义固定条带。
    in_probe = (midpoint_coordinates[:, 0] >= PROBE_X_MIN_M - 1.0e-12) & (midpoint_coordinates[:, 0] <= PROBE_X_MAX_M + 1.0e-12)  # 用米制容差识别闭区间探针条带。
    cell_values = np.where(in_probe, PROBE_CELL_TAG, REGULAR_CELL_TAG).astype(np.int32)  # 将条带标为二，其余单元标为一。
    cell_tags = mesh.meshtags(domain, topological_dimension, cell_indices, cell_values)  # 创建覆盖完整单元集合的 MeshTags。
    cell_tags.name = CELL_TAG_NAME  # 设置稳定单元标签名称供 XDMF 按名回读。
    boundary_markers = (  # 冻结四个几何定位器与其工程语义标签。
        (LEFT_FACET_TAG, lambda x: np.isclose(x[0], 0.0)),  # 左边 x=0 对应全固定边界。
        (RIGHT_FACET_TAG, lambda x: np.isclose(x[0], LENGTH_M)),  # 右边 x=L 对应均布向下载荷边界。
        (BOTTOM_FACET_TAG, lambda x: np.isclose(x[1], 0.0)),  # 下边 y=0 对应自由边界。
        (TOP_FACET_TAG, lambda x: np.isclose(x[1], HEIGHT_M)),  # 上边 y=H 对应自由边界。
    )  # 结束四边定位器定义。
    facet_index_parts: list[np.ndarray] = []  # 收集各边界的本地 facet 索引数组。
    facet_value_parts: list[np.ndarray] = []  # 收集与每个索引同长的整数标签数组。
    for marker_value, locator in boundary_markers:  # 依次定位左、右、下、上四边且不重复角点边。
        located_facets = mesh.locate_entities_boundary(domain, facet_dimension, locator)  # 按几何条件查找当前边的 boundary facets。
        if located_facets.size == 0:  # 任一边为空说明网格或单位约定已损坏。
            raise RuntimeError(f"facet tag {marker_value} contains no facets")  # 在求解前失败并留下异常证据。
        facet_index_parts.append(located_facets.astype(np.int32, copy=False))  # 保存本边索引并保证 DOLFINx 整数类型。
        facet_value_parts.append(np.full(located_facets.shape, marker_value, dtype=np.int32))  # 为本边每个 facet 填充相同语义标签。
    unsorted_facet_indices = np.concatenate(facet_index_parts)  # 合并四边索引形成完整外边界标签集合。
    unsorted_facet_values = np.concatenate(facet_value_parts)  # 合并四边标签并维持与索引一一对应。
    facet_order = np.argsort(unsorted_facet_indices)  # 按实体索引排序以满足 MeshTags 稳定查找要求。
    facet_indices = unsorted_facet_indices[facet_order]  # 应用同一排序得到升序 facet 索引。
    facet_values = unsorted_facet_values[facet_order]  # 应用同一排序保持标签绑定不变。
    if np.unique(facet_indices).size != facet_indices.size:  # 重复 facet 会导致边界作用重复或语义冲突。
        raise RuntimeError("boundary facet tags contain duplicate entities")  # 拒绝不唯一边界映射。
    facet_tags = mesh.meshtags(domain, facet_dimension, facet_indices, facet_values)  # 创建完整四边 MeshTags。
    facet_tags.name = FACET_TAG_NAME  # 设置稳定边标签名称供 XDMF 按名回读。
    domain.topology.create_connectivity(facet_dimension, topological_dimension)  # 为标签 I/O 与边界积分创建边到单元连接。
    domain.topology.create_connectivity(facet_dimension, 0)  # 为 PNG 边界线绘制创建边到顶点连接。
    return domain, cell_tags, facet_tags  # 返回网格、单元标签和边标签供求解与写出。


def solve_model(domain: Any, cell_tags: Any, facet_tags: Any, prefix: str) -> tuple[dict[str, Any], dict[str, Any]]:  # 在给定命名网格上独立求解并返回 JSON 指标及真实场状态。
    topological_dimension = domain.topology.dim  # 读取单元维数以定位边界、单元和网格尺寸。
    facet_dimension = topological_dimension - 1  # 定义二维网格的一维边界实体维数。
    vector_space = fem.functionspace(domain, ("Lagrange", 1, (2,)))  # 创建二维连续 P1 位移函数空间。
    left_facets = facet_tags.find(LEFT_FACET_TAG)  # 按标签十一读取全固定左边界 facets。
    left_dofs = fem.locate_dofs_topological(vector_space, facet_dimension, left_facets)  # 将左边 facets 映射到位移自由度块。
    zero_vector = np.zeros(2, dtype=PETSc.ScalarType)  # 创建两个分量均为零的固定边界值，单位米。
    fixed_boundary = fem.dirichletbc(zero_vector, left_dofs, vector_space)  # 同时约束左边水平和竖向位移。
    trial = ufl.TrialFunction(vector_space)  # 定义待求二维位移试函数。
    test = ufl.TestFunction(vector_space)  # 定义二维虚位移测试函数。
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)  # 创建可按单元标签积分的面积测度，单位平方米。
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)  # 创建可按边标签积分的长度测度，单位米。
    traction = fem.Constant(domain, np.array([0.0, TRACTION_Y_PA], dtype=PETSc.ScalarType))  # 定义右边界二维面力向量，单位 Pa。
    bilinear_form = THICKNESS_M * ufl.inner(stress(trial), strain(test)) * dx  # 构造厚度积分后的线弹性刚度双线性型，单位 N·m。
    linear_form = THICKNESS_M * ufl.dot(traction, test) * ds(RIGHT_FACET_TAG)  # 构造右边界均布面力的外功线性型，单位 N·m。
    problem = LinearProblem(bilinear_form, linear_form, bcs=[fixed_boundary], petsc_options_prefix=prefix, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True})  # 用一次预处理调用和直接 LU 求解冻结系统。
    displacement = problem.solve()  # 触发形式编译、装配、边界处理和 PETSc 真实求解。
    displacement.name = "displacement_m"  # 为 XDMF 输出设置带单位语义的位移场名称。
    displacement.x.scatter_forward()  # 同步可能存在的幽灵自由度后再提取任何指标。
    ksp_reason = int(problem.solver.getConvergedReason())  # 读取 PETSc 正值收敛或负值失败原因码。
    ksp_iterations = int(problem.solver.getIterationNumber())  # 记录 LU 路径实际 KSP 迭代计数供诊断。
    right_length = assemble_real(1.0 * ds(RIGHT_FACET_TAG))  # 装配右边真实边长，单位米。
    if right_length <= 0.0:  # 空或负测度会使边平均位移无定义。
        raise RuntimeError("right boundary has non-positive measure")  # 在指标除法前显式失败。
    right_mean_uy = assemble_real(displacement[1] * ds(RIGHT_FACET_TAG)) / right_length  # 计算右边竖向位移的长度平均值，单位米。
    stiffness_work = assemble_real(THICKNESS_M * ufl.inner(stress(displacement), strain(displacement)) * dx)  # 计算 a(u,u)，单位焦耳。
    load_work = assemble_real(THICKNESS_M * ufl.dot(traction, displacement) * ds(RIGHT_FACET_TAG))  # 计算 L(u) 即柔度，单位焦耳。
    strain_energy = 0.5 * stiffness_work  # 按线弹性定义取二分之一 a(u,u) 得应变能，单位焦耳。
    probe_area = assemble_real(1.0 * dx(PROBE_CELL_TAG))  # 装配固定物理条带面积，单位平方米。
    if probe_area <= 0.0:  # 空探针条带无法产生稳定平均应力。
        raise RuntimeError("probe cell tag has non-positive area")  # 在应力平均除法前显式失败。
    probe_mean_vm = assemble_real(von_mises(displacement) * dx(PROBE_CELL_TAG)) / probe_area  # 计算条带内面积平均 von Mises 应力，单位 Pa。
    dg0_space = fem.functionspace(domain, ("DG", 0))  # 创建每单元常值空间承载可写出的应力诊断场。
    vm_expression = fem.Expression(von_mises(displacement), dg0_space.element.interpolation_points)  # 在 DG0 插值点编译真实等效应力表达式。
    vm_field = fem.Function(dg0_space)  # 分配每单元常值等效应力函数。
    vm_field.name = "von_mises_Pa_DG0"  # 设置包含单位和离散阶次的 XDMF 场名。
    vm_field.interpolate(vm_expression)  # 将真实求解位移导出的等效应力插值到 DG0。
    vm_field.x.scatter_forward()  # 同步应力场幽灵值以支持输出和绘图。
    owned_vm_count = int(dg0_space.dofmap.index_map.size_local * dg0_space.dofmap.index_map_bs)  # 计算当前进程拥有的 DG0 标量自由度数。
    local_vm_values = np.real(vm_field.x.array[:owned_vm_count])  # 读取拥有单元的真实等效应力值。
    local_max_vm = float(np.max(local_vm_values)) if local_vm_values.size else 0.0  # 计算本地最大等效应力，空分区取零。
    maximum_vm = float(COMM.allreduce(local_max_vm, op=MPI.MAX))  # 求全局诊断性最大等效应力，单位 Pa。
    raw_residual_form = ufl.action(bilinear_form, displacement) - linear_form  # 直接形成未施加边界条件的弱式残差 R(v)=a(u,v)-L(v)。
    raw_residual = fem_petsc.assemble_vector(fem.form(raw_residual_form))  # 将原始弱式残差装配为与位移空间一致的 PETSc 向量。
    raw_residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # 把共享实体贡献反向累加到拥有自由度。
    raw_load = fem_petsc.assemble_vector(fem.form(linear_form))  # 独立装配未修改外载向量，仅用于残差归一化尺度。
    raw_load.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # 把共享实体载荷贡献反向累加到拥有自由度。
    raw_load_norm = float(raw_load.norm())  # 读取原始外载向量二范数作为自由残差归一化尺度。
    fixed_dofs_all, first_ghost_position = fixed_boundary.dof_indices()  # 获取展开后的固定标量自由度以及首个非拥有位置。
    fixed_dofs = np.asarray(fixed_dofs_all[:first_ghost_position], dtype=np.int32)  # 仅保留当前进程拥有的固定标量自由度。
    block_size = int(vector_space.dofmap.index_map_bs)  # 读取位移空间每个节点块的标量分量数，预期为二。
    if block_size != 2:  # 三个刚体虚位移场仅对二维位移问题有定义。
        raise RuntimeError(f"unexpected displacement block size: {block_size}")  # 拒绝错误维数而不静默误算反力。
    owned_scalar_count = int(vector_space.dofmap.index_map.size_local * block_size)  # 计算当前进程拥有的展开标量自由度数。
    residual_values = np.real(raw_residual.getArray(readonly=True)[:owned_scalar_count]).copy()  # 仅复制拥有残差，避免幽灵项重复计数。
    free_mask = np.ones(owned_scalar_count, dtype=bool)  # 初始将所有拥有的展开标量自由度视为自由自由度。
    free_mask[fixed_dofs] = False  # 排除固定自由度，使其反力不污染方程残差检查。
    free_residual_norm = float(np.linalg.norm(residual_values[free_mask]))  # 计算自由自由度方程残差二范数，单位牛顿。
    virtual_translation_x = fem.Function(vector_space)  # 分配单位水平刚体平移虚位移场以提取总水平反力。
    virtual_translation_x.interpolate(lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))  # 通过空间插值定义 w=(1,0)，不假定标量自由度奇偶编号。
    virtual_translation_y = fem.Function(vector_space)  # 分配单位竖向刚体平移虚位移场以提取总竖向反力。
    virtual_translation_y.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.ones(x.shape[1]))))  # 通过空间插值定义 w=(0,1)，不假定标量自由度奇偶编号。
    virtual_rotation_z = fem.Function(vector_space)  # 分配绕原点单位刚体转动虚位移场以提取总反力矩。
    virtual_rotation_z.interpolate(lambda x: np.vstack((-x[1], x[0])))  # 通过空间插值定义 w=(-y,x)，使残差功等于绕原点力矩。
    for virtual_field in (virtual_translation_x, virtual_translation_y, virtual_rotation_z):  # 依次限制三个虚位移只在固定自由度上非零。
        owned_virtual_values = virtual_field.x.array[:owned_scalar_count]  # 取得当前进程拥有的虚位移展开数组视图。
        owned_virtual_values[free_mask] = PETSc.ScalarType(0.0)  # 清零所有自由自由度，仅保留支座处刚体虚位移值。
    local_reaction_x = float(np.real(np.vdot(virtual_translation_x.x.array[:owned_scalar_count], residual_values)))  # 用残差与固定水平虚位移内积得到本地水平反力。
    local_reaction_y = float(np.real(np.vdot(virtual_translation_y.x.array[:owned_scalar_count], residual_values)))  # 用残差与固定竖向虚位移内积得到本地竖向反力。
    local_reaction_moment = float(np.real(np.vdot(virtual_rotation_z.x.array[:owned_scalar_count], residual_values)))  # 用残差与固定转动虚位移内积得到本地原点力矩。
    reaction_x = float(COMM.allreduce(local_reaction_x, op=MPI.SUM))  # 汇总全局水平支座反力，单位牛顿。
    reaction_y = float(COMM.allreduce(local_reaction_y, op=MPI.SUM))  # 汇总全局竖向支座反力，单位牛顿。
    reaction_moment = float(COMM.allreduce(local_reaction_moment, op=MPI.SUM))  # 汇总全局支座反力关于原点力矩，单位牛顿米。
    coordinates = ufl.SpatialCoordinate(domain)  # 创建空间坐标用于装配真实外载合力及关于原点力矩。
    external_fx = assemble_real(THICKNESS_M * traction[0] * ds(RIGHT_FACET_TAG))  # 装配右边界水平外力，单位牛顿。
    external_fy = assemble_real(THICKNESS_M * traction[1] * ds(RIGHT_FACET_TAG))  # 装配右边界竖向外力，单位牛顿。
    external_moment = assemble_real(THICKNESS_M * (coordinates[0] * traction[1] - coordinates[1] * traction[0]) * ds(RIGHT_FACET_TAG))  # 装配外载关于原点力矩，单位牛顿米。
    force_scale = max(math.hypot(external_fx, external_fy), 1.0)  # 用总外力大小且至少一牛顿归一化合力不平衡。
    force_balance_relative = math.hypot(reaction_x + external_fx, reaction_y + external_fy) / force_scale  # 计算全局二维合力相对不平衡。
    moment_scale = max(abs(external_moment), force_scale * max(LENGTH_M, HEIGHT_M), 1.0)  # 用真实外力矩或特征力矩归一化。
    moment_balance_relative = abs(reaction_moment + external_moment) / moment_scale  # 计算关于原点的相对力矩不平衡。
    free_residual_relative = free_residual_norm / max(raw_load_norm, 1.0)  # 计算自由自由度残差相对外载离散范数。
    energy_identity_relative = abs(stiffness_work - load_work) / max(abs(stiffness_work), abs(load_work), 1.0)  # 检查离散恒等式 a(u,u)=L(u)。
    owned_cells = np.arange(domain.topology.index_map(topological_dimension).size_local, dtype=np.int32)  # 构造当前进程拥有单元索引用于真实 h 计算。
    local_h_values = domain.h(topological_dimension, owned_cells)  # 调用 DOLFINx 几何接口计算每个拥有单元的真实尺寸。
    local_hmax = float(np.max(local_h_values)) if local_h_values.size else 0.0  # 计算本进程最大单元尺寸，空分区取零。
    hmax = float(COMM.allreduce(local_hmax, op=MPI.MAX))  # 汇总全局真实最大单元尺寸，单位米。
    cell_count = int(domain.topology.index_map(topological_dimension).size_global)  # 读取全局三角形单元数量。
    vector_dof_count = int(vector_space.dofmap.index_map.size_global * block_size)  # 计算全局展开位移标量自由度数。
    node_block_count = int(vector_space.dofmap.index_map.size_global)  # 记录全局位移节点块数量以澄清自由度口径。
    cell_tag_counts = {str(tag): int(COMM.allreduce(int(np.count_nonzero(cell_tags.values == tag)), op=MPI.SUM)) for tag in (REGULAR_CELL_TAG, PROBE_CELL_TAG)}  # 汇总每种单元标签数量。
    facet_tag_counts = {str(tag): int(COMM.allreduce(int(np.count_nonzero(facet_tags.values == tag)), op=MPI.SUM)) for tag in (LEFT_FACET_TAG, RIGHT_FACET_TAG, BOTTOM_FACET_TAG, TOP_FACET_TAG)}  # 汇总每种边标签数量。
    metrics = {  # 组织当前网格层的全部标量结果与数值验证量。
        "mesh": {"hmax_m": hmax, "global_cells": cell_count, "global_vector_scalar_dofs": vector_dof_count, "global_vector_node_blocks": node_block_count, "cell_tag_counts": cell_tag_counts, "facet_tag_counts": facet_tag_counts},  # 记录真实网格尺寸、单元数、自由度与标签覆盖。
        "solver": {"ksp_type": "preonly", "pc_type": "lu", "converged_reason": ksp_reason, "iterations": ksp_iterations},  # 记录冻结求解器设置及真实 KSP 状态。
        "qoi": {"right_edge_mean_uy_m": right_mean_uy, "tip_uy_metric_m": right_mean_uy, "strain_energy_J": strain_energy, "compliance_J": load_work, "probe_mean_von_mises_Pa": probe_mean_vm, "diagnostic_max_von_mises_Pa": maximum_vm},  # 记录冻结 QoI、能量、探针平均应力与非门禁峰值。
        "equilibrium": {"external_force_N": [external_fx, external_fy], "reaction_force_N": [reaction_x, reaction_y], "external_moment_about_origin_Nm": external_moment, "reaction_moment_about_origin_Nm": reaction_moment, "force_balance_relative": force_balance_relative, "moment_balance_relative": moment_balance_relative, "free_dof_residual_norm_N": free_residual_norm, "free_dof_residual_relative": free_residual_relative},  # 记录由未施加边界条件残差得到的合力、力矩及自由残差。
        "energy_identity": {"a_u_u_J": stiffness_work, "L_u_J": load_work, "relative_difference": energy_identity_relative},  # 记录离散能量恒等式两侧及相对差。
    }  # 结束当前层机器指标对象。
    state = {"domain": domain, "cell_tags": cell_tags, "facet_tags": facet_tags, "vector_space": vector_space, "displacement": displacement, "vm_space": dg0_space, "vm_field": vm_field}  # 保留真实场对象供最细层 I/O 与 PNG 使用。
    raw_residual.destroy()  # 释放本函数显式创建的 PETSc 原始残差向量。
    raw_load.destroy()  # 释放本函数显式创建的 PETSc 原始载荷向量。
    return metrics, state  # 返回可序列化指标及仍有效的 DOLFINx 场状态。


def write_mesh_package(path: Path, state: dict[str, Any]) -> None:  # 将最细层命名网格和两类标签写到 path 对应 HDF5 XDMF 包。
    with io.XDMFFile(COMM, str(path), "w", encoding=io.XDMFFile.Encoding.HDF5) as xdmf_file:  # 以 HDF5 数据后端创建新的 XDMF 容器并保证离开作用域时关闭。
        xdmf_file.write_mesh(state["domain"])  # 写出名称为 MESH_NAME 的最细层真实网格。
        xdmf_file.write_meshtags(state["cell_tags"], state["domain"].geometry)  # 写出标签一和二及其单元实体绑定。
        xdmf_file.write_meshtags(state["facet_tags"], state["domain"].geometry)  # 写出标签十一到十四及其边实体绑定。


def read_mesh_package(path: Path) -> tuple[Any, Any, Any]:  # 关闭写端后从 path 新建网格及标签对象，不复用内存状态。
    with io.XDMFFile(COMM, str(path), "r") as xdmf_file:  # 以只读方式打开已关闭的 XDMF/HDF5 包。
        imported_domain = xdmf_file.read_mesh(name=MESH_NAME)  # 按冻结名称读取一个全新的 DOLFINx 网格对象。
        topological_dimension = imported_domain.topology.dim  # 读取回读网格单元维数以建立标签所需连接。
        facet_dimension = topological_dimension - 1  # 定义回读网格的边界实体维数。
        imported_domain.topology.create_connectivity(facet_dimension, topological_dimension)  # 为边标签读取及积分创建边到单元连接。
        imported_domain.topology.create_connectivity(facet_dimension, 0)  # 为回读边界可视化创建边到顶点连接。
        imported_cell_tags = xdmf_file.read_meshtags(imported_domain, name=CELL_TAG_NAME)  # 按名称读取完整单元标签对象。
        imported_facet_tags = xdmf_file.read_meshtags(imported_domain, name=FACET_TAG_NAME)  # 按名称读取完整四边标签对象。
    return imported_domain, imported_cell_tags, imported_facet_tags  # 返回与写出前对象身份无关的回读模型。


def write_solution_fields(output_root: Path, state: dict[str, Any]) -> tuple[Path, Path]:  # 将真实最细层位移和 DG0 等效应力分别写为 HDF5 XDMF。
    displacement_path = output_root / "displacement.xdmf"  # 冻结位移场 XDMF 主文件路径。
    stress_path = output_root / "von_mises.xdmf"  # 冻结 DG0 等效应力 XDMF 主文件路径。
    with io.XDMFFile(COMM, str(displacement_path), "w", encoding=io.XDMFFile.Encoding.HDF5) as displacement_file:  # 创建位移 HDF5 XDMF 并在作用域结束时关闭。
        displacement_file.write_mesh(state["domain"])  # 写出位移场所属的最细层回读网格。
        displacement_file.write_function(state["displacement"])  # 写出真实求得的二维 P1 位移向量场，单位米。
    with io.XDMFFile(COMM, str(stress_path), "w", encoding=io.XDMFFile.Encoding.HDF5) as stress_file:  # 创建应力 HDF5 XDMF 并在作用域结束时关闭。
        stress_file.write_mesh(state["domain"])  # 写出 DG0 应力场所属的同一最细层回读网格。
        stress_file.write_function(state["vm_field"])  # 写出由真实位移导出的每单元 von Mises 场，单位 Pa。
    return displacement_path, stress_path  # 返回两个 XDMF 主文件路径供哈希和门禁检查。


def render_summary(path: Path, state: dict[str, Any], levels: list[dict[str, Any]]) -> dict[str, float]:  # 从实际 DOLFINx 场生成四联 PNG 并返回绘图尺度诊断。
    domain = state["domain"]  # 读取最细层回读网格对象作为全部空间图的数据源。
    vector_space = state["vector_space"]  # 读取最细层二维 P1 位移空间。
    displacement = state["displacement"]  # 读取最细层真实求解位移函数。
    vm_space = state["vm_space"]  # 读取最细层 DG0 等效应力空间。
    vm_field = state["vm_field"]  # 读取最细层真实等效应力函数。
    geometry_points = np.asarray(domain.geometry.x[:, :2], dtype=np.float64)  # 取得网格几何节点二维坐标，单位米。
    triangles = np.asarray(domain.geometry.dofmaps[0][:, :3], dtype=np.int32)  # 取得 DOLFINx 0.11 三角单元到几何节点映射。
    block_size = int(vector_space.dofmap.index_map_bs)  # 读取二维位移空间每节点块分量数。
    dof_coordinates = np.asarray(vector_space.tabulate_dof_coordinates()[:, :2], dtype=np.float64)  # 取得 P1 位移节点块坐标。
    displacement_blocks = np.real(displacement.x.array).reshape((-1, block_size))  # 将实际位移数组重排成节点块乘分量矩阵。
    if dof_coordinates.shape[0] != displacement_blocks.shape[0]:  # P1 节点坐标与位移块必须一一对应。
        raise RuntimeError("plot displacement coordinates do not match vector blocks")  # 拒绝基于错误排列绘制场图。
    displacement_by_coordinate = {tuple(np.round(coordinate, 12)): displacement_blocks[index, :2] for index, coordinate in enumerate(dof_coordinates)}  # 以十二位小数坐标键安全映射位移节点。
    try:  # 显式捕获几何节点无法映射到 P1 位移节点的拓扑错误。
        vertex_displacement = np.vstack([displacement_by_coordinate[tuple(np.round(point, 12))] for point in geometry_points])  # 按几何节点次序重排真实二维位移。
    except KeyError as error:  # 任一几何节点缺少 P1 位移值都会使场图不可信。
        raise RuntimeError(f"plot geometry node has no displacement value: {error}") from error  # 将缺失坐标转换为清晰失败原因。
    displacement_magnitude = np.linalg.norm(vertex_displacement, axis=1)  # 计算每个几何节点的实际位移幅值，单位米。
    maximum_displacement = float(np.max(displacement_magnitude)) if displacement_magnitude.size else 0.0  # 取得实际最大位移供变形图缩放。
    deformation_scale = 0.15 * LENGTH_M / maximum_displacement if maximum_displacement > 0.0 else 1.0  # 将最大显示变形控制为板长约百分之十五。
    deformed_points = geometry_points + deformation_scale * vertex_displacement  # 生成仅用于显示的放大变形节点坐标。
    cell_count = triangles.shape[0]  # 读取实际三角形数量供 DG0 单元值提取。
    cell_vm_values = np.zeros(cell_count, dtype=np.float64)  # 分配按几何单元顺序排列的等效应力数组。
    for cell_index in range(cell_count):  # 遍历每个实际三角形并读取其唯一 DG0 自由度。
        cell_dofs = vm_space.dofmap.cell_dofs(cell_index)  # 获取当前单元对应的 DG0 自由度索引。
        cell_vm_values[cell_index] = float(np.real(vm_field.x.array[int(cell_dofs[0])]))  # 保存当前单元真实等效应力，单位 Pa。
    cell_tag_values = np.full(cell_count, REGULAR_CELL_TAG, dtype=np.int32)  # 先将全部绘图单元设为普通标签一。
    cell_tag_values[np.asarray(state["cell_tags"].indices, dtype=np.int32)] = np.asarray(state["cell_tags"].values, dtype=np.int32)  # 按实际 MeshTags 覆盖探针条带标签二。
    original_triangulation = mtri.Triangulation(geometry_points[:, 0], geometry_points[:, 1], triangles)  # 创建未变形实际网格三角剖分。
    deformed_triangulation = mtri.Triangulation(deformed_points[:, 0], deformed_points[:, 1], triangles)  # 创建放大变形几何三角剖分。
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 8.0), constrained_layout=True)  # 创建十四乘八英寸四联摘要画布。
    mesh_axis = axes[0, 0]  # 选择左上网格和标签面板。
    mesh_colors = mesh_axis.tripcolor(original_triangulation, facecolors=cell_tag_values, shading="flat", cmap="Pastel1", edgecolors="0.72", linewidth=0.2)  # 用实际标签着色并绘制实际单元边。
    figure.colorbar(mesh_colors, ax=mesh_axis, label="cell tag")  # 添加单元标签色标以区分一与二。
    boundary_connectivity = domain.topology.connectivity(domain.topology.dim - 1, 0)  # 读取实际边到顶点连接供四边描绘。
    boundary_colors = {LEFT_FACET_TAG: "tab:red", RIGHT_FACET_TAG: "tab:blue", BOTTOM_FACET_TAG: "tab:green", TOP_FACET_TAG: "tab:orange"}  # 冻结四个边标签的可辨颜色。
    for facet_tag, color in boundary_colors.items():  # 逐标签描绘全部真实边界 facet。
        first_segment = True  # 仅在每个标签第一条边上添加图例避免重复。
        for facet_index in state["facet_tags"].find(facet_tag):  # 遍历当前标签绑定的真实 facet 索引。
            vertices = boundary_connectivity.links(int(facet_index))  # 取得当前边的两个几何顶点索引。
            segment = geometry_points[vertices, :]  # 读取边段两端坐标，单位米。
            mesh_axis.plot(segment[:, 0], segment[:, 1], color=color, linewidth=2.0, label=str(facet_tag) if first_segment else None)  # 绘制标签边并仅首次命名。
            first_segment = False  # 标记当前标签图例已创建。
    mesh_axis.legend(title="facet tag", loc="upper center", ncol=4)  # 在网格面板集中显示 11 至 14 标签语义键。
    mesh_axis.set_title("Actual finest mesh and tags")  # 标明左上面板来自真实最细 DOLFINx 网格。
    mesh_axis.set_aspect("equal")  # 保持米制 x、y 比例不变形。
    stress_axis = axes[0, 1]  # 选择右上放大变形等效应力面板。
    stress_colors = stress_axis.tripcolor(deformed_triangulation, facecolors=cell_vm_values / 1.0e6, shading="flat", cmap="viridis")  # 以 MPa 显示实际 DG0 等效应力。
    figure.colorbar(stress_colors, ax=stress_axis, label="von Mises (MPa)")  # 添加等效应力单位明确的色标。
    stress_axis.set_title(f"Deformed DG0 stress, scale={deformation_scale:.3g}")  # 显示真实场与仅可视化变形倍率。
    stress_axis.set_aspect("equal")  # 保持变形几何两个坐标方向相同比例。
    displacement_axis = axes[1, 0]  # 选择左下实际位移幅值面板。
    displacement_colors = displacement_axis.tripcolor(original_triangulation, displacement_magnitude * 1.0e3, shading="gouraud", cmap="magma")  # 以毫米显示 P1 节点实际位移幅值。
    figure.colorbar(displacement_colors, ax=displacement_axis, label="|u| (mm)")  # 添加实际位移幅值单位色标。
    displacement_axis.set_title("Actual displacement magnitude")  # 标明左下面板直接来自求解位移场。
    displacement_axis.set_aspect("equal")  # 保持未变形物理几何比例。
    convergence_axis = axes[1, 1]  # 选择右下网格收敛面板。
    h_values = np.array([float(level["mesh"]["hmax_m"]) for level in levels], dtype=np.float64)  # 提取四层真实 hmax，单位米。
    tip_values = np.abs(np.array([float(level["qoi"]["right_edge_mean_uy_m"]) for level in levels], dtype=np.float64))  # 提取四层右边平均竖向位移绝对值。
    energy_values = np.array([float(level["qoi"]["strain_energy_J"]) for level in levels], dtype=np.float64)  # 提取四层应变能。
    probe_values = np.array([float(level["qoi"]["probe_mean_von_mises_Pa"]) for level in levels], dtype=np.float64)  # 提取四层探针平均等效应力。
    convergence_axis.plot(h_values, tip_values / tip_values[-1], "o-", label="|mean uy| / finest")  # 绘制相对最细层的位移 QoI 收敛轨迹。
    convergence_axis.plot(h_values, energy_values / energy_values[-1], "s-", label="energy / finest")  # 绘制相对最细层的应变能收敛轨迹。
    convergence_axis.plot(h_values, probe_values / probe_values[-1], "^-", label="probe VM / finest")  # 绘制相对最细层的探针平均应力收敛轨迹。
    convergence_axis.set_xscale("log")  # 用对数横轴清楚显示二倍网格加密。
    convergence_axis.invert_xaxis()  # 让由粗到细的阅读方向从左向右。
    convergence_axis.grid(True, which="both", alpha=0.3)  # 添加轻量网格线帮助比较最后两层变化。
    convergence_axis.legend()  # 显示三个冻结收敛指标图例。
    convergence_axis.set_xlabel("actual hmax (m)")  # 标注横轴为 DOLFINx 实测最大单元尺寸。
    convergence_axis.set_ylabel("quantity / finest value")  # 标注纵轴为相对最细层无量纲值。
    convergence_axis.set_title("Four-level mesh convergence")  # 标明右下面板使用全部四层真实求解结果。
    figure.suptitle("DOLFINx 0.11 plane-stress cantilever demonstration")  # 设置不暗示研究结论的总标题。
    figure.savefig(path, dpi=180)  # 以每英寸一百八十点写出清晰 PNG 工件。
    plt.close(figure)  # 显式释放 Matplotlib 画布与数组引用。
    return {"maximum_displacement_m": maximum_displacement, "deformation_display_scale": float(deformation_scale), "maximum_dg0_von_mises_Pa": float(np.max(cell_vm_values))}  # 返回绘图数据诊断供主回执复核。


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: Any) -> None:  # 向 checks 追加一个命名门禁；四参数分别是列表、名称、布尔结果与证据。
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})  # 规范化布尔值并保留可审计证据。


def write_engineering_reports(output_root: Path, levels: list[dict[str, Any]], checks: list[dict[str, Any]], convergence: dict[str, Any], roundtrip: dict[str, Any], numerical_passed: bool) -> list[Path]:  # 写出技能合同要求的十二份轻量 JSON 报告。
    report_payloads: dict[str, dict[str, Any]] = {}  # 创建文件名到严格 JSON 对象的映射。
    common_limit = {"scientific_claim_allowed": False, "research_case_execution_status": "not_executed_missing_current_evidence", "research_cases_remain_blocked": True, "calculix_calls": 0}  # 冻结所有报告共同的使用边界。
    report_payloads["analysis_charter.json"] = {"schema_version": "bridge-analysis-charter/1.0", "gate_status": "BLOCKED", "intended_use_class": "demonstration_only", "gate": {"id": "G0", "status": "BLOCKED", "reason": "formal project standards and original research-case evidence are absent"}, "model_purpose": "lightweight solver-backed FEniCSx workflow demonstration", "analysis_scope": "2D small-strain isotropic linear elasticity in plane stress", "frozen_assumptions": {"L_m": LENGTH_M, "H_m": HEIGHT_M, "thickness_m": THICKNESS_M, "E_Pa": YOUNG_MODULUS_PA, "nu": POISSON_RATIO, "right_traction_y_Pa": TRACTION_Y_PA, "left_boundary": "fully_fixed"}, "allowed_use": ["workflow integration", "numerical gate regression", "artifact I/O demonstration"], "prohibited_use": ["design approval", "code compliance", "publication claim", "original research-case substitution"], **common_limit}  # 冻结用途、物理范围、关键假定和 G0 阻断，并提供 validator 顶层门字段。
    report_payloads["standards_manifest.json"] = {"schema_version": "standards-manifest/1.0", "gate": "G0", "status": "BLOCKED", "gate_status": "BLOCKED", "standards": [], "formal_standards": [], "missing_inputs": ["project governing standard", "edition and amendment", "load and resistance factors", "acceptance clauses"], "threshold_source": "user-frozen demonstration contract only", **common_limit}  # 明确没有正式标准且阈值不是规范验算值，并提供空 standards 合同。
    report_payloads["response_metric_register.json"] = {"schema_version": "response-metric-register/1.0", "metrics": [{"id": "right_edge_mean_uy", "unit": "m", "extraction": "integral uy on facet tag 12 divided by edge length", "last_pair_limit": TIP_CHANGE_LIMIT}, {"id": "strain_energy", "unit": "J", "extraction": "0.5*a(u,u)", "last_pair_limit": ENERGY_CHANGE_LIMIT}, {"id": "compliance", "unit": "J", "extraction": "L(u)", "identity": "a(u,u)=L(u)"}, {"id": "probe_mean_von_mises", "unit": "Pa", "extraction": "area mean on cell tag 2", "last_pair_limit": PROBE_STRESS_CHANGE_LIMIT}, {"id": "diagnostic_max_von_mises", "unit": "Pa", "extraction": "maximum DG0 interpolation", "convergence_gate": False}], "threshold_source": "user-frozen demonstration contract", **common_limit}  # 注册空间选择、单位、提取规则与预冻结门限。
    report_payloads["approval_matrix.json"] = {"schema_version": "approval-matrix/1.0", "decisions": [{"use": "automated demonstration execution", "approval": "allowed when numerical gates pass"}, {"use": "scientific or engineering conclusion", "approval": "blocked pending responsible engineer, formal standards, original evidence, and independent check"}], **common_limit}  # 区分自动演示与专业签认权限。
    report_payloads["scope_exclusion_register.json"] = {"schema_version": "scope-exclusion-register/1.0", "excluded": ["geometric nonlinearity", "plasticity and damage", "plane strain or three-dimensional effects", "buckling", "dynamics", "fatigue", "connection and local detail design", "code compliance", "singular peak stress as a convergence target", "original research cases"], **common_limit}  # 列出本轻量模型明确不覆盖的物理与用途。
    report_payloads["solution_verification_report.json"] = {"schema_version": "solution-verification-report/1.0", "verification_outcome": "pass_for_demonstration_only" if numerical_passed else "fail", "engineering_acceptance_allowed": False, "gate": {"id": "G13", "status": "PASSED_FOR_DEMONSTRATION_ONLY" if numerical_passed else "FAILED"}, "checks": checks, "note": "G13 numerical verification cannot clear blocked G0 or establish engineering validity", **common_limit}  # 汇总方程解验证、明确禁止工程接受并提供 validator 结果字段。
    report_payloads["verified_result_set.json"] = {"schema_version": "verified-result-set/1.0", "status": "verified_for_demonstration_only" if numerical_passed else "not_verified", "mesh_sequence": levels, "finest_result": levels[-1], "roundtrip": roundtrip, "allowed_use": "demonstration_only", **common_limit}  # 保存四层及最细层结果并以精确枚举限定为演示用途。
    report_payloads["global_equilibrium_report.json"] = {"schema_version": "global-equilibrium-report/1.0", "origin_m": [0.0, 0.0], "method": "unconstrained UFL residual tested by fixed-support rigid translations and rotation", "levels": [{"mesh": level["mesh"], "equilibrium": level["equilibrium"], "energy_identity": level["energy_identity"]} for level in levels], "relative_limit": RESIDUAL_LIMIT, **common_limit}  # 报告各层合力、原点力矩、自由残差和能量平衡。
    report_payloads["substructure_free_body_report.json"] = {"schema_version": "substructure-free-body-report/1.0", "status": "not_performed", "reason": "this lightweight demonstration contains one whole cantilever plate and no independently isolated substructure", "global_free_body_available": True, **common_limit}  # 明确整体板没有可独立切分的子结构自由体。
    report_payloads["mesh_and_step_convergence_report.json"] = {"schema_version": "mesh-and-step-convergence-report/1.0", "mesh_family": "structured rectangles split by right diagonal into P1 triangles", "levels": [{"nx": level["nx"], "ny": level["ny"], "hmax_m": level["mesh"]["hmax_m"], "cells": level["mesh"]["global_cells"], "vector_scalar_dofs": level["mesh"]["global_vector_scalar_dofs"]} for level in levels], "last_pair": convergence, "step_convergence": {"status": "not_applicable", "reason": "single-step linear static solve with direct LU"}, **common_limit}  # 保存预冻结网格序列和最后一对收敛判定。
    report_payloads["solver_warning_disposition.json"] = {"schema_version": "solver-warning-disposition/1.0", "status": "no_failure_reason_reported" if all(int(level["solver"]["converged_reason"]) > 0 for level in levels) else "solver_failure_present", "evidence": [{"nx": level["nx"], "ny": level["ny"], **level["solver"]} for level in levels], "limitation": "the API convergence reason is captured; no independent textual log parser is used", **common_limit}  # 处置 KSP 状态并披露未解析文本日志的限制。
    report_payloads["solution_issues.json"] = {"schema_version": "solution-issues/1.0", "open_issues": [{"severity": "blocking_for_scientific_use", "issue": "formal governing standards are absent"}, {"severity": "blocking_for_scientific_use", "issue": "original research cases remain unexecuted"}, {"severity": "known_numerical_limitation", "issue": "clamped-corner maximum stress is mesh-sensitive and diagnostic only"}, {"severity": "known_model_limitation", "issue": "P1 displacement with DG0 stress and two-dimensional plane stress is a lightweight idealization"}], **common_limit}  # 保留所有未关闭问题且不以数值通过掩盖用途阻断。
    written_paths: list[Path] = []  # 收集成功写出的报告路径供哈希清单使用。
    for filename, payload in report_payloads.items():  # 逐一写出每份独立技能合同报告。
        report_path = output_root / filename  # 将冻结文件名定位到本次工件根目录。
        write_json(report_path, payload)  # 使用严格 JSON 写入并拒绝非有限结果。
        written_paths.append(report_path)  # 保存路径供主回执建立哈希与字节数清单。
    return written_paths  # 返回十二份报告的绝对路径列表。


def run_simulation(output_root: Path) -> bool:  # 执行完整四层求解、往返重算、可视化、报告与主回执，并返回数值合同是否通过。
    started_at = utc_now()  # 记录完整活动开始 UTC 时间。
    if COMM.size != 1:  # 本回执中的拥有自由度残差和文件合同明确限定串行。
        raise RuntimeError(f"serial execution required, received MPI size {COMM.size}")  # 拒绝并行误用而不输出错误合力。
    if str(dolfinx.__version__) != "0.11.0.post0":  # 用户冻结的官方运行时版本必须精确匹配。
        raise RuntimeError(f"DOLFINx 0.11.0.post0 required, received {dolfinx.__version__}")  # 拒绝未经验证的 API 版本漂移。
    output_root.mkdir(parents=True, exist_ok=True)  # 在任何数值工作前创建工件根目录。
    levels: list[dict[str, Any]] = []  # 收集四层真实求解的机器指标。
    finest_state: dict[str, Any] | None = None  # 保存最细层真实场供关闭回读、输出和绘图。
    for level_index, (nx, ny) in enumerate(MESH_LEVELS):  # 按十六乘四到一百二十八乘三十二逐层求解。
        domain, cell_tags, facet_tags = build_mesh_and_tags(nx, ny)  # 生成当前层真实网格与完整标签。
        metrics, state = solve_model(domain, cell_tags, facet_tags, prefix=f"cantilever_l{level_index}_")  # 使用唯一 PETSc 前缀执行当前层独立 LU 求解。
        metrics["level"] = level_index  # 记录零起始网格层编号供机器排序。
        metrics["nx"] = nx  # 记录当前水平方向矩形分割数。
        metrics["ny"] = ny  # 记录当前竖直方向矩形分割数。
        levels.append(metrics)  # 将当前层全部结果加入收敛序列。
        if level_index == len(MESH_LEVELS) - 1:  # 仅最细层需要保留场对象并做 I/O 往返。
            finest_state = state  # 保存最细层网格、标签、位移和应力场。
    if finest_state is None:  # 空网格序列属于脚本合同错误。
        raise RuntimeError("no finest state was generated")  # 阻止无模型写出伪回执。
    mesh_package_path = output_root / "mesh_tags.xdmf"  # 冻结命名网格与标签包主路径。
    write_mesh_package(mesh_package_path, finest_state)  # 写出最细层命名网格及完整 tags。
    COMM.barrier()  # 确保 XDMF 与 HDF5 写端均关闭完成后再启动读端。
    imported_domain, imported_cell_tags, imported_facet_tags = read_mesh_package(mesh_package_path)  # 关闭后创建独立回读模型对象。
    roundtrip_metrics, roundtrip_state = solve_model(imported_domain, imported_cell_tags, imported_facet_tags, prefix="cantilever_roundtrip_")  # 在回读模型上独立装配并求解一次。
    displacement_path, stress_path = write_solution_fields(output_root, roundtrip_state)  # 输出回读重算的位移和 DG0 应力 XDMF/HDF5。
    png_path = output_root / "elasticity_simulation_summary.png"  # 冻结与说明文档一致的真实场四联摘要图路径。
    plot_diagnostics = render_summary(png_path, roundtrip_state, levels)  # 从实际 DOLFINx fields 而非 AI 生成 PNG。
    coarse_of_last_pair = levels[-2]  # 读取六十四乘十六层作为最后一对较粗层。
    finest = levels[-1]  # 读取一百二十八乘三十二层作为最后一对最细层。
    tip_change = relative_difference(coarse_of_last_pair["qoi"]["right_edge_mean_uy_m"], finest["qoi"]["right_edge_mean_uy_m"])  # 计算最后一对位移 QoI 相对变化。
    energy_change = relative_difference(coarse_of_last_pair["qoi"]["strain_energy_J"], finest["qoi"]["strain_energy_J"])  # 计算最后一对应变能相对变化。
    probe_change = relative_difference(coarse_of_last_pair["qoi"]["probe_mean_von_mises_Pa"], finest["qoi"]["probe_mean_von_mises_Pa"])  # 计算最后一对探针平均应力相对变化。
    convergence = {"coarse_level": [coarse_of_last_pair["nx"], coarse_of_last_pair["ny"]], "fine_level": [finest["nx"], finest["ny"]], "right_edge_mean_uy_relative_change": tip_change, "right_edge_mean_uy_limit": TIP_CHANGE_LIMIT, "strain_energy_relative_change": energy_change, "strain_energy_limit": ENERGY_CHANGE_LIMIT, "probe_mean_von_mises_relative_change": probe_change, "probe_mean_von_mises_limit": PROBE_STRESS_CHANGE_LIMIT}  # 保存预冻结最后一对收敛合同。
    roundtrip_metric_paths = (  # 冻结关闭回读前后必须比较的 QoI、能量、应力与反力路径。
        ("right_edge_mean_uy_m", ("qoi", "right_edge_mean_uy_m")),  # 比较右边平均竖向位移。
        ("strain_energy_J", ("qoi", "strain_energy_J")),  # 比较应变能。
        ("compliance_J", ("qoi", "compliance_J")),  # 比较柔度或外载功。
        ("probe_mean_von_mises_Pa", ("qoi", "probe_mean_von_mises_Pa")),  # 比较固定条带平均应力。
        ("reaction_fx_N", ("equilibrium", "reaction_force_N", 0)),  # 比较水平支座合力。
        ("reaction_fy_N", ("equilibrium", "reaction_force_N", 1)),  # 比较竖向支座合力。
        ("reaction_moment_Nm", ("equilibrium", "reaction_moment_about_origin_Nm")),  # 比较支座关于原点力矩。
    )  # 结束往返比较指标路径定义。
    roundtrip_differences: dict[str, float] = {}  # 收集每个指标的关闭回读相对差。
    for metric_name, metric_path in roundtrip_metric_paths:  # 逐个解析嵌套指标路径并计算相对差。
        reference_value: Any = finest  # 从内存最细层结果根对象开始解析。
        imported_value: Any = roundtrip_metrics  # 从回读独立求解结果根对象开始解析。
        for path_component in metric_path:  # 依次使用字符串键或整数索引进入嵌套值。
            reference_value = reference_value[path_component]  # 读取写出前最细层对应值。
            imported_value = imported_value[path_component]  # 读取关闭回读重算对应值。
        roundtrip_scale = max(abs(float(reference_value)), abs(float(imported_value)), 1.0)  # 对理论零反力使用至少一个相应单位的尺度，避免舍入噪声被零分母放大。
        roundtrip_differences[metric_name] = abs(float(imported_value) - float(reference_value)) / roundtrip_scale  # 保存带稳定零值处理的无量纲相对差。
    roundtrip_maximum = max(roundtrip_differences.values())  # 取得全部冻结往返指标中的最大相对差。
    roundtrip = {"method": "close XDMF/HDF5, read named mesh and tags into fresh objects, independently assemble and solve", "metric_relative_differences": roundtrip_differences, "maximum_relative_difference": roundtrip_maximum, "limit": ROUNDTRIP_LIMIT, "imported_metrics": roundtrip_metrics}  # 保存完整独立回读证据。
    primary_records = {"mesh_tags_xdmf": file_record(mesh_package_path, output_root), "mesh_tags_h5": file_record(mesh_package_path.with_suffix(".h5"), output_root), "displacement_xdmf": file_record(displacement_path, output_root), "displacement_h5": file_record(displacement_path.with_suffix(".h5"), output_root), "von_mises_xdmf": file_record(stress_path, output_root), "von_mises_h5": file_record(stress_path.with_suffix(".h5"), output_root), "summary_png": file_record(png_path, output_root)}  # 以 validator 冻结的七个逻辑键记录真实字节数与哈希。
    checks: list[dict[str, Any]] = []  # 创建统一数值和工件门禁列表。
    add_check(checks, "all_ksp_reasons_positive", all(int(level["solver"]["converged_reason"]) > 0 for level in levels) and int(roundtrip_metrics["solver"]["converged_reason"]) > 0, {"levels": [level["solver"]["converged_reason"] for level in levels], "roundtrip": roundtrip_metrics["solver"]["converged_reason"]})  # 要求五次真实 KSP 求解均以正原因码收敛。
    for level in levels:  # 对四个网格层分别建立残差与能量硬门禁。
        level_label = f"n{level['nx']}x{level['ny']}"  # 创建稳定层级标签供检查名称使用。
        external_force_error = math.hypot(level["equilibrium"]["external_force_N"][0], level["equilibrium"]["external_force_N"][1] + 10000.0) / 10000.0  # 按厚度乘边长验证预期外力为零与负一万牛顿。
        external_moment_error = abs(level["equilibrium"]["external_moment_about_origin_Nm"] + 40000.0) / 40000.0  # 验证右边均布力关于原点的预期力矩为负四万牛顿米。
        add_check(checks, f"{level_label}_external_load_sanity", external_force_error <= 1.0e-10 and external_moment_error <= 1.0e-10, {"expected_force_N": [0.0, -10000.0], "actual_force_N": level["equilibrium"]["external_force_N"], "force_relative_error": external_force_error, "expected_moment_Nm": -40000.0, "actual_moment_Nm": level["equilibrium"]["external_moment_about_origin_Nm"], "moment_relative_error": external_moment_error, "limit": 1.0e-10})  # 防止厚度、边标签、方向或力矩原点错误。
        add_check(checks, f"{level_label}_response_sign_sanity", level["equilibrium"]["reaction_force_N"][1] > 0.0 and level["equilibrium"]["reaction_moment_about_origin_Nm"] > 0.0 and level["qoi"]["right_edge_mean_uy_m"] < 0.0, {"reaction_y_N": level["equilibrium"]["reaction_force_N"][1], "reaction_moment_Nm": level["equilibrium"]["reaction_moment_about_origin_Nm"], "right_edge_mean_uy_m": level["qoi"]["right_edge_mean_uy_m"]})  # 要求支座竖向反力和抗弯力矩向上为正且受载边位移向下。
        add_check(checks, f"{level_label}_force_balance", level["equilibrium"]["force_balance_relative"] <= RESIDUAL_LIMIT, {"actual": level["equilibrium"]["force_balance_relative"], "limit": RESIDUAL_LIMIT})  # 检查当前层全局合力相对不平衡。
        add_check(checks, f"{level_label}_moment_balance", level["equilibrium"]["moment_balance_relative"] <= RESIDUAL_LIMIT, {"actual": level["equilibrium"]["moment_balance_relative"], "limit": RESIDUAL_LIMIT})  # 检查当前层关于原点力矩相对不平衡。
        add_check(checks, f"{level_label}_free_residual", level["equilibrium"]["free_dof_residual_relative"] <= RESIDUAL_LIMIT, {"actual": level["equilibrium"]["free_dof_residual_relative"], "limit": RESIDUAL_LIMIT})  # 检查当前层自由自由度原始残差。
        add_check(checks, f"{level_label}_energy_identity", level["energy_identity"]["relative_difference"] <= RESIDUAL_LIMIT, {"actual": level["energy_identity"]["relative_difference"], "limit": RESIDUAL_LIMIT})  # 检查当前层 a(u,u)=L(u)。
    add_check(checks, "last_pair_right_edge_mean_uy", tip_change <= TIP_CHANGE_LIMIT, {"actual": tip_change, "limit": TIP_CHANGE_LIMIT})  # 检查最后一对位移 QoI 收敛。
    add_check(checks, "last_pair_strain_energy", energy_change <= ENERGY_CHANGE_LIMIT, {"actual": energy_change, "limit": ENERGY_CHANGE_LIMIT})  # 检查最后一对应变能收敛。
    add_check(checks, "last_pair_probe_mean_von_mises", probe_change <= PROBE_STRESS_CHANGE_LIMIT, {"actual": probe_change, "limit": PROBE_STRESS_CHANGE_LIMIT})  # 检查最后一对固定条带平均应力收敛。
    finest_sanity_passed = -3.0e-3 <= finest["qoi"]["right_edge_mean_uy_m"] <= -3.0e-4 and 1.0 <= finest["qoi"]["strain_energy_J"] <= 20.0 and 1.0e6 <= finest["qoi"]["probe_mean_von_mises_Pa"] <= 30.0e6  # 用宽松量级窗识别单位、厚度或边界条件灾难性错误。
    add_check(checks, "finest_physical_magnitude_sanity", finest_sanity_passed, {"right_edge_mean_uy_m": {"actual": finest["qoi"]["right_edge_mean_uy_m"], "range": [-3.0e-3, -3.0e-4]}, "strain_energy_J": {"actual": finest["qoi"]["strain_energy_J"], "range": [1.0, 20.0]}, "probe_mean_von_mises_Pa": {"actual": finest["qoi"]["probe_mean_von_mises_Pa"], "range": [1.0e6, 30.0e6]}, "purpose": "demonstration sanity only, not engineering acceptance"})  # 检查最细层三个主要响应处于冻结演示量级窗。
    add_check(checks, "xdmf_roundtrip", roundtrip_maximum <= ROUNDTRIP_LIMIT, {"actual": roundtrip_maximum, "limit": ROUNDTRIP_LIMIT, "metrics": roundtrip_differences})  # 检查关闭回读重算全部冻结指标。
    add_check(checks, "primary_artifacts_nonempty", all(record["size_bytes"] > 0 for record in primary_records.values()), primary_records)  # 检查七个命名 PNG、XDMF 与 HDF5 工件全部存在且非空。
    numerical_passed = bool(checks) and all(bool(check["passed"]) for check in checks)  # 仅全部硬门禁通过才接受演示数值合同。
    report_paths = write_engineering_reports(output_root, levels, checks, convergence, roundtrip, numerical_passed)  # 写出十二份技能合同报告，无论数值通过与否都保留证据。
    report_records = {path.stem: file_record(path, output_root) for path in report_paths}  # 为全部报告按稳定文件 stem 计算字节数与 SHA-256。
    mesh_level_summary = [{"divisions": [level["nx"], level["ny"]], "global_cells": level["mesh"]["global_cells"], "global_dofs": level["mesh"]["global_vector_scalar_dofs"], "ksp_converged_reason": level["solver"]["converged_reason"]} for level in levels]  # 构造 validator 要求的四行扁平网格摘要。
    roundtrip_comparison = {"outcome": "pass" if roundtrip_maximum <= ROUNDTRIP_LIMIT else "fail", "imported_ksp_converged_reason": roundtrip_metrics["solver"]["converged_reason"], "maximum_relative_difference": roundtrip_maximum, "limit": ROUNDTRIP_LIMIT}  # 构造 validator 要求的第五次求解与差分摘要。
    receipt = {  # 构造本次演示活动的唯一主机器回执。
        "schema_version": "v5-fenicsx-elasticity-simulation/1.0",  # 冻结主回执契约版本。
        "status": "structural_simulation_passed" if numerical_passed else "structural_simulation_failed",  # 使用 validator 冻结的结构演示专用成功或失败枚举。
        "contract_test_outcome": "pass" if numerical_passed else "fail",  # 提供 CI 可直接消费的稳定枚举。
        "artifact_kind": "solver_backed_lightweight_demonstration",  # 防止把该工件误认为原研究案例结果。
        "execution_family": "FEniCS/FEniCSx",  # 在顶层明确实际执行技术族供独立 validator 校验。
        "started_at_utc": started_at,  # 保存完整活动开始时间。
        "finished_at_utc": utc_now(),  # 保存全部文件和报告写出后的结束时间。
        "runtime": runtime_identity(),  # 保存真实 FEniCSx、PETSc 和容器身份。
        "provenance": source_identity(),  # 保存实际脚本哈希与可用 CI 来源。
        "problem_contract": {"geometry_m": {"length": LENGTH_M, "height": HEIGHT_M, "thickness": THICKNESS_M}, "material": {"model": "isotropic_linear_elastic_plane_stress", "E_Pa": YOUNG_MODULUS_PA, "nu": POISSON_RATIO}, "boundary_conditions": {"facet_11": "fully fixed", "facet_12": {"traction_Pa": [0.0, TRACTION_Y_PA]}, "facet_13": "free", "facet_14": "free"}, "cells": {"tag_1": "regular", "tag_2": f"probe strip {PROBE_X_MIN_M} <= x <= {PROBE_X_MAX_M} m"}, "element": "continuous P1 triangle displacement", "stress_output": "DG0 explicit plane-stress von Mises", "solver": "PETSc preonly+lu", "mesh_levels": [list(level) for level in MESH_LEVELS]},  # 保存完整冻结模型合同。
        "levels": levels,  # 保存四层网格、QoI、KSP、平衡和能量结果。
        "mesh_levels": mesh_level_summary,  # 保存 validator 消费的四层 divisions、单元、自由度和 KSP 扁平摘要。
        "convergence": convergence,  # 保存最后一对预冻结收敛量。
        "roundtrip": roundtrip,  # 保存关闭回读独立重算差异。
        "roundtrip_comparison": roundtrip_comparison,  # 保存 validator 消费的往返结果和第五次 KSP 原因码。
        "plot_diagnostics": plot_diagnostics,  # 保存实际场绘图的位移和变形尺度诊断。
        "checks": checks,  # 保存全部硬门禁及逐项证据。
        "files": primary_records,  # 以七个冻结逻辑键保存 validator 将独立复核的主工件哈希和字节数。
        "report_files": report_records,  # 在独立顶层对象保存十二份 JSON 报告哈希且不混入七类格式检查。
        "execution_counts": {"mesh_generation_calls": 4, "mesh_export_calls": 3, "mesh_import_calls": 1, "linear_solve_calls": 5, "qoi_extraction_calls": 5, "calculix_calls": 0, "model_calls": 0, "ai_image_calls": 0},  # 记录五次真实 FEniCSx 求解以及零次 CalculiX、模型服务与 AI 图像调用。
        "research_case_execution_status": "not_executed_missing_current_evidence",  # 明确原研究案例仍未执行。
        "uses_original_research_inputs": False,  # 明确本演示全部输入由脚本冻结生成且未使用原研究输入。
        "research_cases_remain_blocked": True,  # 明确本演示不能解除研究案例阻断。
        "scientific_claim_allowed": False,  # 无论数值门禁是否通过都禁止科学或工程结论。
        "governance_gate": {"id": "G0", "status": "BLOCKED", "reason": "no formal governing standard or original research-case evidence"},  # 保持章程入口质量门阻断。
        "success_meaning": "the frozen lightweight DOLFINx demonstration passed its numerical and artifact contracts only",  # 限定绿色状态的唯一含义。
    }  # 结束主回执对象。
    write_json(output_root / "simulation_receipt.json", receipt)  # 最后写出包含所有报告哈希的主严格 JSON 回执。
    if not numerical_passed:  # 数值门禁失败时除主回执外还必须留下显式失败回执。
        failure_payload = {"schema_version": "v5-fenicsx-elasticity-failure/1.0", "status": "numerical_gate_failure", "failed_checks": [check for check in checks if not check["passed"]], "simulation_receipt": file_record(output_root / "simulation_receipt.json", output_root), "scientific_claim_allowed": False, "research_case_execution_status": "not_executed_missing_current_evidence", "calculix_calls": 0}  # 汇总失败门禁并链接主回执内容身份。
        write_json(output_root / "failure_receipt.json", failure_payload)  # 写出数值失败回执供 CI 上传和快速定位。
    return numerical_passed  # 返回演示数值合同状态供进程退出码使用。


def main() -> int:  # 解析参数、执行活动并保证异常时写出 failure_receipt.json 后退出一。
    args = parse_args()  # 读取调用方显式工件目录。
    output_root = Path(args.output_dir).resolve()  # 将工件根目录解析为绝对路径以防相对路径漂移。
    output_root.mkdir(parents=True, exist_ok=True)  # 在进入 try 前创建目录以最大化异常留证机会。
    try:  # 捕获网格、求解、I/O、绘图、门禁和序列化的所有运行时异常。
        passed = run_simulation(output_root)  # 执行完整四层真实 DOLFINx 演示活动。
        if COMM.rank == ROOT_RANK:  # 仅根进程向 CI 标准输出写一行摘要。
            print(json.dumps({"status": "pass" if passed else "fail", "output_dir": str(output_root), "scientific_claim_allowed": False}, ensure_ascii=False))  # 输出不冒充科学结论的精简状态。
        return 0 if passed else 1  # 数值合同通过返回零，否则返回一。
    except BaseException as error:  # 捕获所有可报告异常，包括 PETSc、DOLFINx、I/O 与严格 JSON 错误。
        failure_payload = {"schema_version": "v5-fenicsx-elasticity-failure/1.0", "status": "exception", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "runtime": runtime_identity(), "provenance": source_identity(), "scientific_claim_allowed": False, "research_case_execution_status": "not_executed_missing_current_evidence", "research_cases_remain_blocked": True, "calculix_calls": 0}  # 构造包含完整栈和不越权声明的异常回执。
        write_json(output_root / "failure_receipt.json", failure_payload)  # 即使主回执未形成也写出独立失败证据。
        if COMM.rank == ROOT_RANK:  # 仅根进程向 CI 标准错误输出一行摘要。
            print(json.dumps({"status": "exception", "error": str(error), "output_dir": str(output_root)}, ensure_ascii=False), file=sys.stderr)  # 输出可搜索异常摘要且不泄露环境变量。
        return 1  # 所有异常路径均以非零状态结束。


if __name__ == "__main__":  # 仅当工作流直接执行本脚本时启动数值活动。
    raise SystemExit(main())  # 将 main 返回的零或一准确传递给容器和 CI。
