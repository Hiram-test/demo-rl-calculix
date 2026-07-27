from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sqrt
from typing import Any

import numpy as np

from .diagnostics import (
    DiagnosticResult,
    SkillRecord,
    energy_consistency_skill,
    peak_growth_skill,
    physical_variant_skill,
    theory_cross_check_skill,
)
from .fem import Mesh, Solution, edge_traction_load, nearest_element, nearest_node, point_load, solve_linear_plane_stress
from .meshes import (
    central_crack_mesh,
    circular_hole_mesh,
    nodes_on_coordinate,
    rectangle_mesh,
    rounded_rect_hole_mesh,
    select_edges_by_midpoint,
)


@dataclass
class ScenarioRun:
    scenario_id: str
    title: str
    user_question: str
    level_rows: list[dict[str, Any]]
    diagnostic: DiagnosticResult
    meshes: dict[str, Mesh] = field(default_factory=dict)
    solutions: dict[str, Solution] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "user_question": self.user_question,
            "level_rows": self.level_rows,
            "diagnostic": self.diagnostic.to_dict(),
            "metadata": self.metadata,
        }


def _minimal_rigid_constraints(mesh: Mesh) -> dict[int, float]:
    xmin, ymin = mesh.nodes.min(axis=0)
    xmax, _ = mesh.nodes.max(axis=0)
    n1 = nearest_node(mesh, (float(xmin), float(ymin)))
    n2 = nearest_node(mesh, (float(xmax), float(ymin)))
    return {2 * n1: 0.0, 2 * n1 + 1: 0.0, 2 * n2 + 1: 0.0}


def _tension_x_load(mesh: Mesh, sigma: float, thickness: float) -> np.ndarray:
    return edge_traction_load(mesh, mesh.edge_sets["right"], (sigma, 0.0), thickness) + edge_traction_load(
        mesh, mesh.edge_sets["left"], (-sigma, 0.0), thickness
    )


def _tension_y_load(mesh: Mesh, sigma: float, thickness: float) -> np.ndarray:
    return edge_traction_load(mesh, mesh.edge_sets["top"], (0.0, sigma), thickness) + edge_traction_load(
        mesh, mesh.edge_sets["bottom"], (0.0, -sigma), thickness
    )


def _remote_average_displacement(mesh: Mesh, solution: Solution, edge_name: str, component: int) -> float:
    nodes = sorted({n for e in mesh.edge_sets[edge_name] for n in e})
    return float(np.mean(solution.displacements[nodes, component]))


def _far_field_stress(solution: Solution, *, axis: int, x_fraction: float = 0.35) -> float:
    centers = solution.element_centers
    xmax = np.max(np.abs(centers[:, 0]))
    mask = np.abs(centers[:, 0]) >= x_fraction * xmax
    return float(np.median(solution.element_stress[mask, axis]))


def run_load_introduction_case() -> ScenarioRun:
    sid = "bearing_load_introduction"
    title = "支座/横梁载荷引入：单节点反力与有限承压宽度"
    question = "我把支座反力施加在一个节点，网格越细最大应力越大；应该怎样加密，结果还能不能用？"
    young, poisson, thickness = 210000.0, 0.30, 12.0
    length, height, total_load = 1000.0, 100.0, -600000.0
    levels = [(40, 8), (80, 16), (160, 32)]
    rows: list[dict[str, Any]] = []
    meshes: dict[str, Mesh] = {}
    solutions: dict[str, Solution] = {}

    I = thickness * height**3 / 12.0
    x_section = 0.15 * length
    theory_tip = abs(total_load) * length**3 / (3.0 * young * I)
    theory_section_stress = abs(total_load) * (length - x_section) * (height / 2.0) / I
    theory_energy = 0.5 * abs(total_load) * theory_tip

    for idx, (nx, ny) in enumerate(levels, start=1):
        mesh = rectangle_mesh(length, height, nx, ny)
        left_nodes = nodes_on_coordinate(mesh, x=0.0)
        constraints = {2 * int(n): 0.0 for n in left_nodes}
        constraints.update({2 * int(n) + 1: 0.0 for n in left_nodes})
        load_node = nearest_node(mesh, (length, 0.0))
        f = point_load(mesh, load_node, (0.0, total_load))
        sol = solve_linear_plane_stress(mesh, young, poisson, thickness, f, constraints)
        centers = sol.element_centers
        load_roi = centers[:, 0] > 0.82 * length
        peak = float(np.max(sol.element_von_mises[load_roi]))
        tip_disp = abs(_remote_average_displacement(mesh, sol, "right", 1))
        xdist = np.abs(centers[:, 0] - x_section)
        band = xdist <= np.min(xdist) + 1e-9
        section_stress = float(np.max(np.abs(sol.element_stress[band, 0])))
        row = {
            "level": idx,
            "mesh": f"{nx}x{ny}",
            "elements": int(len(mesh.elements)),
            "h_local": height / ny,
            "peak_stress": peak,
            "tip_displacement": tip_disp,
            "fixed_qoi": section_stress,
            "strain_energy": sol.strain_energy,
            "energy_balance_rel": sol.energy_balance_rel,
        }
        rows.append(row)
        meshes[f"point_L{idx}"] = mesh
        solutions[f"point_L{idx}"] = sol

    nx, ny = levels[-1]
    patch_mesh = rectangle_mesh(length, height, nx, ny)
    left_nodes = nodes_on_coordinate(patch_mesh, x=0.0)
    constraints = {2 * int(n): 0.0 for n in left_nodes}
    constraints.update({2 * int(n) + 1: 0.0 for n in left_nodes})
    patch_width = 40.0
    patch_edges = select_edges_by_midpoint(patch_mesh, "right", lambda _x, y: abs(y) <= patch_width / 2.0 + 1e-9)
    actual_length = sum(float(np.linalg.norm(patch_mesh.nodes[b] - patch_mesh.nodes[a])) for a, b in patch_edges)
    traction_y = total_load / (actual_length * thickness)
    f_patch = edge_traction_load(patch_mesh, patch_edges, (0.0, traction_y), thickness)
    patch_sol = solve_linear_plane_stress(patch_mesh, young, poisson, thickness, f_patch, constraints)
    centers = patch_sol.element_centers
    patch_peak = float(np.max(patch_sol.element_von_mises[centers[:, 0] > 0.82 * length]))
    patch_tip = abs(_remote_average_displacement(patch_mesh, patch_sol, "right", 1))
    xdist = np.abs(centers[:, 0] - x_section)
    band = xdist <= np.min(xdist) + 1e-9
    patch_section = float(np.max(np.abs(patch_sol.element_stress[band, 0])))
    patch_row = {
        "mesh": f"{nx}x{ny}",
        "elements": int(len(patch_mesh.elements)),
        "h_local": height / ny,
        "peak_stress": patch_peak,
        "tip_displacement": patch_tip,
        "fixed_qoi": patch_section,
        "strain_energy": patch_sol.strain_energy,
        "energy_balance_rel": patch_sol.energy_balance_rel,
        "patch_width": actual_length,
    }
    meshes["patch_final"] = patch_mesh
    solutions["patch_final"] = patch_sol

    skills: list[SkillRecord] = [
        energy_consistency_skill(rows),
        theory_cross_check_skill(
            name="Euler-Bernoulli悬臂梁挠度",
            numerical=rows[-1]["tip_displacement"],
            theoretical=theory_tip,
            tolerance=0.12,
            quantity="自由端平均位移",
        ),
        theory_cross_check_skill(
            name="梁截面弯曲正应力",
            numerical=rows[-1]["fixed_qoi"],
            theoretical=theory_section_stress,
            tolerance=0.16,
            quantity="距固定端15%长度处截面应力",
        ),
        peak_growth_skill(rows, peak_key="peak_stress", stable_keys=["tip_displacement", "fixed_qoi", "strain_energy"]),
        physical_variant_skill(
            baseline=rows[-1],
            variant=patch_row,
            peak_key="peak_stress",
            preserved_keys=["tip_displacement", "fixed_qoi", "strain_energy"],
            description="把单节点反力改成保持同一合力和作用位置的有限承压宽度，并核对远场响应未被破坏。",
            preserved_tolerance=0.12,
        ),
    ]
    diagnostic = DiagnosticResult(
        user_question=question,
        diagnosis="单节点载荷形成局部数值奇异峰值；继续在该节点周围加密不会得到可用的设计应力。",
        plain_explanation="整体位移、截面弯曲应力和应变能已经稳定，但载荷节点附近的最大应力仍随网格增长。把同一支座反力落实为有限承压宽度后，局部峰值显著降低，而远场响应基本保持。",
        applied_plan=[
            "保留总反力与合力作用线不变。",
            f"把单节点力改为右端约{actual_length:.1f} mm宽的均布承压载荷。",
            "载荷区采用局部细网格，评价量改为承压区外的截面应力、平均位移和能量。",
            "报告中保留原始峰值，但明确标记为不可直接用于设计。",
        ],
        supported_use=["整体刚度判断", "承压区外截面应力", "载荷引入方案之间的相对比较"],
        unsupported_use=["单节点处最大应力作为材料强度或疲劳应力"],
        evidence={
            "theory_tip_displacement": theory_tip,
            "theory_section_stress": theory_section_stress,
            "theory_strain_energy": theory_energy,
            "physical_variant": patch_row,
        },
        skill_trace=skills,
    )
    return ScenarioRun(sid, title, question, rows, diagnostic, meshes, solutions, {"variant_row": patch_row, "young": young, "poisson": poisson, "thickness": thickness})


def run_circular_opening_case() -> ScenarioRun:
    sid = "web_circular_opening"
    title = "腹板圆孔：孔边应力集中与定向O形加密"
    question = "桥梁腹板圆孔边的应力对网格很敏感，我应该全局加密还是只加密孔边？"
    young, poisson, thickness = 210000.0, 0.30, 10.0
    width = height = 240.0
    radius = 20.0
    sigma = 100.0
    levels = [(48, 6), (96, 10), (160, 16)]
    rows: list[dict[str, Any]] = []
    meshes: dict[str, Mesh] = {}
    solutions: dict[str, Solution] = {}

    for idx, (ntheta, nr) in enumerate(levels, start=1):
        mesh = circular_hole_mesh(width, height, radius, ntheta, nr, cluster=2.4)
        f = _tension_x_load(mesh, sigma, thickness)
        sol = solve_linear_plane_stress(mesh, young, poisson, thickness, f, _minimal_rigid_constraints(mesh))
        centers = sol.element_centers
        rr = np.linalg.norm(centers, axis=1)
        near_hole = rr < radius + 0.24 * radius
        peak_sx = float(np.max(sol.element_stress[near_hole, 0]))
        remote_disp = _remote_average_displacement(mesh, sol, "right", 0) - _remote_average_displacement(mesh, sol, "left", 0)
        far_sigma = _far_field_stress(sol, axis=0)
        row = {
            "level": idx,
            "mesh": f"theta={ntheta}, radial={nr}",
            "elements": int(len(mesh.elements)),
            "h_local": 2.0 * pi * radius / ntheta,
            "peak_stress": peak_sx,
            "fixed_qoi": peak_sx,
            "remote_displacement": remote_disp,
            "far_field_stress": far_sigma,
            "strain_energy": sol.strain_energy,
            "energy_balance_rel": sol.energy_balance_rel,
        }
        rows.append(row)
        meshes[f"L{idx}"] = mesh
        solutions[f"L{idx}"] = sol

    theory_peak = 3.0 * sigma
    skills = [
        energy_consistency_skill(rows),
        theory_cross_check_skill(
            name="Kirsch无限板圆孔解",
            numerical=rows[-1]["peak_stress"],
            theoretical=theory_peak,
            tolerance=0.15,
            quantity="孔顶/孔底切向应力",
        ),
        theory_cross_check_skill(
            name="远场均匀拉应力",
            numerical=rows[-1]["far_field_stress"],
            theoretical=sigma,
            tolerance=0.06,
            quantity="远离孔洞区域的σx",
        ),
    ]
    diagnostic = DiagnosticResult(
        user_question=question,
        diagnosis="这是可收敛的几何应力集中，不需要盲目全局加密；孔边环向与径向定向加密最有效。",
        plain_explanation="孔边应力随网格逐步接近理论应力集中系数3，远场应力与能量同时稳定。系统已采用孔边O形网格，并把单元预算集中到孔周。",
        applied_plan=[
            "沿孔周保持连续环向网格，避免阶梯状孔边。",
            "孔边第一圈单元尺寸按圆周方向控制，径向采用渐增过渡。",
            "至少用三档孔边网格检查孔边应力、远场应力和能量。",
            "达到理论解与网格间变化阈值后停止继续加密。",
        ],
        supported_use=["圆孔弹性应力集中", "孔边局部加密方案", "远场刚度和应力校核"],
        unsupported_use=["焊趾、裂纹或塑性区的直接类比"],
        evidence={"kirsch_peak_stress": theory_peak, "stress_concentration_factor": 3.0},
        skill_trace=skills,
    )
    return ScenarioRun(sid, title, question, rows, diagnostic, meshes, solutions, {"young": young, "poisson": poisson, "thickness": thickness})


def run_crack_case() -> ScenarioRun:
    sid = "cracked_tension_panel"
    title = "钢板裂纹：尖端应力递增与能量释放率/J等价值"
    question = "裂纹尖端应力越加密越高，我是不是还要继续加密？能不能换成J积分或能量方法判断？"
    young, poisson, thickness = 210000.0, 0.30, 10.0
    width = height = 200.0
    half_crack = 20.0
    sigma = 100.0
    levels = [(40, 40), (60, 60), (80, 80)]
    rows: list[dict[str, Any]] = []
    meshes: dict[str, Mesh] = {}
    solutions: dict[str, Solution] = {}

    half_width = width / 2.0
    k_theory = sigma * sqrt(pi * half_crack) * sqrt(1.0 / cos(pi * half_crack / (2.0 * half_width)))
    g_theory = k_theory**2 / young

    for idx, (nx, ny) in enumerate(levels, start=1):
        dx = width / nx
        mesh = central_crack_mesh(width, height, half_crack, nx, ny)
        f = _tension_y_load(mesh, sigma, thickness)
        sol = solve_linear_plane_stress(mesh, young, poisson, thickness, f, _minimal_rigid_constraints(mesh))

        ext_mesh = central_crack_mesh(width, height, half_crack + dx, nx, ny)
        ext_f = _tension_y_load(ext_mesh, sigma, thickness)
        ext_sol = solve_linear_plane_stress(ext_mesh, young, poisson, thickness, ext_f, _minimal_rigid_constraints(ext_mesh))
        g_num = (ext_sol.strain_energy - sol.strain_energy) / (2.0 * dx * thickness)

        centers = sol.element_centers
        tip1 = np.array([half_crack, 0.0])
        tip2 = np.array([-half_crack, 0.0])
        dist = np.minimum(np.linalg.norm(centers - tip1, axis=1), np.linalg.norm(centers - tip2, axis=1))
        roi = dist < 3.5 * dx
        peak_sy = float(np.max(sol.element_stress[roi, 1]))
        row = {
            "level": idx,
            "mesh": f"{nx}x{ny}",
            "elements": int(len(mesh.elements)),
            "h_local": dx,
            "peak_stress": peak_sy,
            "fixed_qoi": g_num,
            "G_numeric": g_num,
            "K_from_G": sqrt(max(g_num, 0.0) * young),
            "strain_energy": sol.strain_energy,
            "energy_balance_rel": max(sol.energy_balance_rel, ext_sol.energy_balance_rel),
        }
        rows.append(row)
        meshes[f"L{idx}"] = mesh
        solutions[f"L{idx}"] = sol
        meshes[f"L{idx}_extended"] = ext_mesh
        solutions[f"L{idx}_extended"] = ext_sol

    skills = [
        energy_consistency_skill(rows),
        peak_growth_skill(rows, peak_key="peak_stress", stable_keys=["G_numeric", "strain_energy"]),
        theory_cross_check_skill(
            name="有限宽中心裂纹LEFM解",
            numerical=rows[-1]["G_numeric"],
            theoretical=g_theory,
            tolerance=0.35,
            quantity="线弹性J=G能量释放率",
        ),
    ]
    diagnostic = DiagnosticResult(
        user_question=question,
        diagnosis="裂纹尖端逐点应力具有线弹性奇异性，最大应力本来就不会随网格收敛；应把评价量切换到J=G或K。",
        plain_explanation="随着裂纹尖端单元变小，尖端最大应力继续升高，但通过两条相邻裂纹长度模型得到的能量释放率趋于稳定，并可与线弹性断裂理论相互校核。因此继续追逐最大应力没有意义，网格应服务于J/G或K的稳定性。",
        applied_plan=[
            "裂纹线与单元边对齐，尖端附近使用连续分级网格。",
            "每档网格同时计算原裂纹和一个单元长度的微增裂纹。",
            "由应变能增量计算线弹性J=G，并换算K进行理论校核。",
            "停止条件改为G/K的网格变化和理论误差，而不是尖端最大应力。",
            "若材料明显屈服，报告会要求补充真实应力—应变曲线并切换弹塑性J；当前报告不冒充弹塑性结论。",
        ],
        supported_use=["线弹性裂纹驱动力", "裂纹尖端加密方向", "J/G/K的交叉验证"],
        unsupported_use=["用尖端最大应力判定强度", "无材料曲线时的弹塑性J", "裂纹扩展寿命预测"],
        evidence={"K_theory": k_theory, "G_theory": g_theory, "plane_condition": "plane_stress"},
        skill_trace=skills,
    )
    return ScenarioRun(sid, title, question, rows, diagnostic, meshes, solutions, {"young": young, "poisson": poisson, "thickness": thickness})


def run_diaphragm_opening_case() -> ScenarioRun:
    sid = "diaphragm_rectangular_opening"
    title = "横隔板矩形开孔：尖角奇异峰值与圆角实体化"
    question = "横隔板矩形开孔角部最大应力一直上升，局部网格应该怎么处理，是否需要改圆角？"
    young, poisson, thickness = 210000.0, 0.30, 12.0
    width, height = 260.0, 180.0
    hx, hy = 45.0, 25.0
    sigma = 90.0
    levels = [(48, 6), (96, 10), (160, 16)]
    rows: list[dict[str, Any]] = []
    meshes: dict[str, Mesh] = {}
    solutions: dict[str, Solution] = {}
    fixed_offset = 15.0
    target = np.array([hx + fixed_offset / sqrt(2.0), hy + fixed_offset / sqrt(2.0)])

    for idx, (ntheta, nr) in enumerate(levels, start=1):
        mesh = rounded_rect_hole_mesh(width, height, hx, hy, 0.0, ntheta, nr, cluster=2.5)
        f = _tension_x_load(mesh, sigma, thickness)
        sol = solve_linear_plane_stress(mesh, young, poisson, thickness, f, _minimal_rigid_constraints(mesh))
        centers = sol.element_centers
        corners = np.array([[sx * hx, sy * hy] for sx in (-1, 1) for sy in (-1, 1)], dtype=float)
        dcorner = np.min(np.linalg.norm(centers[:, None, :] - corners[None, :, :], axis=2), axis=1)
        near = dcorner < 0.22 * min(hx, hy)
        peak = float(np.max(sol.element_von_mises[near]))
        eid = nearest_element(sol, tuple(target))
        fixed = float(sol.element_von_mises[eid])
        remote_disp = _remote_average_displacement(mesh, sol, "right", 0) - _remote_average_displacement(mesh, sol, "left", 0)
        far_sigma = _far_field_stress(sol, axis=0)
        row = {
            "level": idx,
            "mesh": f"theta={ntheta}, radial={nr}",
            "elements": int(len(mesh.elements)),
            "h_local": 2.0 * (hx + hy) / ntheta,
            "peak_stress": peak,
            "fixed_qoi": fixed,
            "remote_displacement": remote_disp,
            "far_field_stress": far_sigma,
            "strain_energy": sol.strain_energy,
            "energy_balance_rel": sol.energy_balance_rel,
        }
        rows.append(row)
        meshes[f"sharp_L{idx}"] = mesh
        solutions[f"sharp_L{idx}"] = sol

    ntheta, nr = levels[-1]
    radius = 16.0
    round_mesh = rounded_rect_hole_mesh(width, height, hx, hy, radius, ntheta, nr, cluster=2.5)
    round_f = _tension_x_load(round_mesh, sigma, thickness)
    round_sol = solve_linear_plane_stress(round_mesh, young, poisson, thickness, round_f, _minimal_rigid_constraints(round_mesh))
    centers = round_sol.element_centers
    inner_dist = np.linalg.norm(centers, axis=1)
    near = inner_dist < 1.5 * np.hypot(hx, hy)
    round_peak = float(np.max(round_sol.element_von_mises[near]))
    round_fixed = float(round_sol.element_von_mises[nearest_element(round_sol, tuple(target))])
    round_remote = _remote_average_displacement(round_mesh, round_sol, "right", 0) - _remote_average_displacement(round_mesh, round_sol, "left", 0)
    round_row = {
        "mesh": f"theta={ntheta}, radial={nr}",
        "elements": int(len(round_mesh.elements)),
        "h_local": 2.0 * (hx + hy) / ntheta,
        "peak_stress": round_peak,
        "fixed_qoi": round_fixed,
        "remote_displacement": round_remote,
        "far_field_stress": _far_field_stress(round_sol, axis=0),
        "strain_energy": round_sol.strain_energy,
        "energy_balance_rel": round_sol.energy_balance_rel,
        "corner_radius": radius,
    }
    meshes["rounded_final"] = round_mesh
    solutions["rounded_final"] = round_sol

    skills = [
        energy_consistency_skill(rows),
        peak_growth_skill(rows, peak_key="peak_stress", stable_keys=["fixed_qoi", "remote_displacement", "strain_energy"]),
        theory_cross_check_skill(
            name="远场均匀拉应力/圣维南区",
            numerical=rows[-1]["far_field_stress"],
            theoretical=sigma,
            tolerance=0.08,
            quantity="远离开孔的σx",
        ),
        physical_variant_skill(
            baseline=rows[-1],
            variant=round_row,
            peak_key="peak_stress",
            preserved_keys=["remote_displacement", "strain_energy"],
            description="把零半径矩形尖角改为明确的16 mm物理圆角，并保持外部尺寸、材料和远场载荷不变。",
            preserved_tolerance=0.18,
        ),
    ]
    diagnostic = DiagnosticResult(
        user_question=question,
        diagnosis="零圆角开孔角部是几何奇异点；峰值随网格增长不能作为局部设计应力。系统已同时落实固定距离评价和16 mm圆角实体化对照。",
        plain_explanation="尖角附近最大应力继续上升，但距角部固定15 mm处的应力、远场位移和应变能趋于稳定。加入明确圆角后，局部峰值下降，而整体响应保持在同一量级。",
        applied_plan=[
            "保留尖角模型作为原始证据，不再以其最大应力作为验收指标。",
            "在角部外固定15 mm物理位置提取应力并做网格收敛。",
            "建立16 mm圆角对照模型，角部采用贴合边界的径向分级网格。",
            "若实际构造圆角未知，报告必须标注该尺寸假设，不替用户虚构焊趾或切割半径。",
        ],
        supported_use=["整体刚度", "远场应力", "固定距离结构应力", "明确圆角模型的局部比较"],
        unsupported_use=["零圆角尖点最大应力", "未确认实际构造半径时的疲劳定量结论"],
        evidence={"fixed_offset": fixed_offset, "rounded_variant": round_row},
        skill_trace=skills,
    )
    return ScenarioRun(sid, title, question, rows, diagnostic, meshes, solutions, {"variant_row": round_row, "young": young, "poisson": poisson, "thickness": thickness})


def run_all_scenarios() -> list[ScenarioRun]:
    return [
        run_load_introduction_case(),
        run_circular_opening_case(),
        run_crack_case(),
        run_diaphragm_opening_case(),
    ]
