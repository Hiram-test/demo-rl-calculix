from __future__ import annotations  # 启用现代类型注解并延迟解析前向引用。

import json  # 保存轻量求解清单和搜索轨迹。
import math  # 计算圆柱几何、连续体位移和角度变化。
import random  # 为可复现的一维粒子群搜索生成随机数。
from dataclasses import asdict, dataclass  # 定义可序列化的网格和结果对象。
from pathlib import Path  # 管理隔离实验输出目录。
from typing import Any  # 标注动态 JSON 结构。

RADIUS_MM = 40.0  # 定义圆轴半径并与工程问题中的直径八十毫米一致。
MODEL_LENGTH_MM = 100.0  # 定义中部观测段长度。
YOUNG_MPA = 210000.0  # 定义普通钢材弹性模量。
POISSON = 0.30  # 定义普通钢材泊松比。
VISIBLE_FORCE_N = 180000.0  # 定义模型可见的轴向拉力。
VISIBLE_TORQUE_NMM = 2000000.0  # 定义模型可见的扭矩并使用牛毫米单位。
GAUGE_LENGTH_MM = 30.0  # 定义表面短标线长度。
GAUGE_START_Z_MM = 35.0  # 定义标线起点轴向位置。
GAUGE_START_THETA_RAD = 0.10  # 定义标线起点圆周位置并避开规则节点。
ANGLE_MIN_DEG = 0.0  # 定义方向搜索下限。
ANGLE_MAX_DEG = 60.0  # 定义方向搜索上限。


@dataclass(frozen=True)  # 使用不可变对象避免运行中悄然改写网格参数。
class MeshConfig:  # 描述表面离散网格的方向和分辨率。
    circumferential: int  # 记录圆周方向划分数量。
    radial: int  # 记录径向划分数量并保留与后续有限元接口兼容。
    axial: int  # 记录轴向划分数量。
    helix_angle_deg: float  # 记录轴向网格线相对轴线的倾斜角。

    def validate(self) -> None:  # 检查轻量网格配置是否合法。
        if self.circumferential < 8:  # 检查圆周划分是否足够形成基本方向差异。
            raise ValueError("circumferential divisions must be at least eight")  # 拒绝过粗圆周网格。
        if self.radial < 1:  # 检查径向划分是否为正数。
            raise ValueError("radial divisions must be positive")  # 拒绝非法径向参数。
        if self.axial < 4:  # 检查轴向划分是否足够覆盖标线。
            raise ValueError("axial divisions must be at least four")  # 拒绝过粗轴向网格。
        if abs(self.helix_angle_deg) > 40.0:  # 限制网格倾斜角以避免极端几何。
            raise ValueError("helix angle must remain within plus or minus forty degrees")  # 拒绝极端斜交网格。


@dataclass(frozen=True)  # 使用不可变对象保存一次轻量离散场结果。
class SolveResult:  # 描述网格节点、节点位移和当前载荷工况。
    config: MeshConfig  # 保存实际使用的网格配置。
    nodes: dict[int, tuple[float, float, float]]  # 保存参考节点坐标。
    displacements: dict[int, tuple[float, float, float]]  # 保存解析场在节点上的位移采样。
    solver: str  # 记录当前为无求解器烟测后端。
    cache_dir: str  # 记录当前结果清单目录。
    force_n: float = VISIBLE_FORCE_N  # 保存当前轴向拉力并提供测试兼容默认值。
    torque_nmm: float = VISIBLE_TORQUE_NMM  # 保存当前扭矩并提供测试兼容默认值。


def _material_state(force_n: float, torque_nmm: float) -> dict[str, float]:  # 计算均匀拉扭连续体状态。
    area = math.pi * RADIUS_MM**2  # 计算实心圆轴横截面积。
    polar_moment = math.pi * RADIUS_MM**4 / 2.0  # 计算实心圆轴极惯性矩。
    shear_modulus = YOUNG_MPA / (2.0 * (1.0 + POISSON))  # 由弹性模量和泊松比计算剪切模量。
    axial_strain = force_n / (YOUNG_MPA * area)  # 计算均匀轴向正应变。
    twist_rate = torque_nmm / (shear_modulus * polar_moment)  # 计算单位长度扭转角。
    return {"area_mm2": area, "polar_moment_mm4": polar_moment, "shear_modulus_mpa": shear_modulus, "axial_strain": axial_strain, "twist_rate_rad_per_mm": twist_rate}  # 返回后处理所需连续体参数。


def _mesh_rotation_rate(config: MeshConfig) -> float:  # 把展开面网格倾角转换为截面旋转率。
    return math.tan(math.radians(config.helix_angle_deg)) / RADIUS_MM  # 使用圆柱展开关系计算每毫米旋转弧度。


def _generate_nodes(config: MeshConfig) -> dict[int, tuple[float, float, float]]:  # 生成可控正交或斜交方向的圆柱节点。
    config.validate()  # 在生成节点前检查网格参数。
    nodes: dict[int, tuple[float, float, float]] = {}  # 初始化节点坐标表。
    node_id = 1  # 初始化稳定节点编号。
    rotation_rate = _mesh_rotation_rate(config)  # 读取网格截面旋转率。
    for axial_index in range(config.axial + 1):  # 遍历全部轴向截面。
        z_value = MODEL_LENGTH_MM * axial_index / config.axial  # 计算当前截面轴向坐标。
        phase = rotation_rate * z_value  # 计算当前截面圆周相位。
        for radial_index in range(config.radial + 1):  # 遍历从圆心到表面的节点环。
            radius = RADIUS_MM * radial_index / config.radial  # 计算当前节点环半径。
            if radial_index == 0:  # 单独处理圆心节点以避免重复扇区节点。
                nodes[node_id] = (0.0, 0.0, z_value)  # 写入当前截面的圆心节点。
                node_id += 1  # 递增节点编号。
                continue  # 跳过圆心的圆周循环。
            for sector in range(config.circumferential):  # 遍历当前节点环的全部圆周位置。
                theta = 2.0 * math.pi * sector / config.circumferential + phase  # 计算含斜交相位的角坐标。
                nodes[node_id] = (radius * math.cos(theta), radius * math.sin(theta), z_value)  # 写入当前节点坐标。
                node_id += 1  # 递增节点编号。
    return nodes  # 返回完整离散节点表。


def _boundary_displacement(point: tuple[float, float, float], force_n: float, torque_nmm: float) -> tuple[float, float, float]:  # 计算均匀拉扭解析位移场并保留旧测试接口名称。
    x_value, y_value, z_value = point  # 拆分节点坐标。
    state = _material_state(force_n, torque_nmm)  # 读取当前工况连续体参数。
    axial_strain = state["axial_strain"]  # 读取轴向应变。
    twist_angle = state["twist_rate_rad_per_mm"] * z_value  # 计算当前截面小转角。
    displacement_x = -POISSON * axial_strain * x_value - twist_angle * y_value  # 计算泊松收缩和扭转共同产生的横向位移。
    displacement_y = -POISSON * axial_strain * y_value + twist_angle * x_value  # 计算另一横向位移分量。
    displacement_z = axial_strain * z_value  # 计算轴向伸长位移。
    return displacement_x, displacement_y, displacement_z  # 返回三个方向位移。


def solve(config: MeshConfig, cache_root: Path, force_n: float = VISIBLE_FORCE_N, torque_nmm: float = VISIBLE_TORQUE_NMM) -> SolveResult:  # 生成无求解器依赖的离散解析场。
    config.validate()  # 检查传入网格配置。
    nodes = _generate_nodes(config)  # 生成当前方向和密度的离散节点。
    displacements = {node_id: _boundary_displacement(point, force_n, torque_nmm) for node_id, point in nodes.items()}  # 在全部节点采样连续体位移场。
    cache_root.mkdir(parents=True, exist_ok=True)  # 创建轻量结果清单目录。
    case_name = f"c{config.circumferential}_r{config.radial}_a{config.axial}_h{config.helix_angle_deg:g}_f{force_n:g}_t{torque_nmm:g}"  # 构造稳定工况名称。
    case_dir = cache_root / case_name  # 定位当前工况清单目录。
    case_dir.mkdir(parents=True, exist_ok=True)  # 创建当前工况目录。
    manifest = {"backend": "analytical_surface_grid_smoke", "mesh": asdict(config), "force_n": force_n, "torque_nmm": torque_nmm, "nodes": len(nodes)}  # 组织可审计轻量清单。
    (case_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存当前工况清单。
    return SolveResult(config=config, nodes=nodes, displacements=displacements, solver="analytical surface-grid smoke backend", cache_dir=str(case_dir), force_n=force_n, torque_nmm=torque_nmm)  # 返回与后续真实有限元接口一致的结果对象。


def _surface_point(beta_deg: float, at_end: bool) -> tuple[float, float, float]:  # 计算标线真实起点或终点坐标。
    beta_rad = math.radians(beta_deg)  # 把标线初始方向转换为弧度。
    distance = GAUGE_LENGTH_MM if at_end else 0.0  # 选择起点或终点沿线距离。
    z_value = GAUGE_START_Z_MM + distance * math.cos(beta_rad)  # 计算轴向坐标。
    theta = GAUGE_START_THETA_RAD + distance * math.sin(beta_rad) / RADIUS_MM  # 计算圆周角坐标。
    return RADIUS_MM * math.cos(theta), RADIUS_MM * math.sin(theta), z_value  # 返回圆柱表面三维坐标。


def _surface_nodes(result: SolveResult) -> list[tuple[int, tuple[float, float, float]]]:  # 选出当前网格全部表面节点。
    tolerance = RADIUS_MM * 1.0e-8  # 定义表面半径判断容差。
    return [(node_id, point) for node_id, point in result.nodes.items() if abs(math.hypot(point[0], point[1]) - RADIUS_MM) <= tolerance]  # 返回表面节点编号和坐标。


def _nearest_surface_node(result: SolveResult, target: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:  # 查找距离真实标线端点最近的表面节点。
    candidates = _surface_nodes(result)  # 读取当前网格全部表面节点。
    node_id, point = min(candidates, key=lambda item: (item[1][0] - target[0])**2 + (item[1][1] - target[1])**2 + (item[1][2] - target[2])**2)  # 选择欧氏距离最小节点。
    return point, result.displacements[node_id]  # 返回参考坐标和节点位移。


def _deformed_point(point: tuple[float, float, float], displacement: tuple[float, float, float]) -> tuple[float, float, float]:  # 计算节点或精确端点的变形后坐标。
    return point[0] + displacement[0], point[1] + displacement[1], point[2] + displacement[2]  # 返回参考坐标与位移之和。


def _unwrap_angle_difference(value: float) -> float:  # 把圆周角差规范到负π到正π。
    while value > math.pi:  # 检查角差是否超过正π。
        value -= 2.0 * math.pi  # 减去一整圈。
    while value < -math.pi:  # 检查角差是否小于负π。
        value += 2.0 * math.pi  # 加上一整圈。
    return value  # 返回连续圆周角差。


def _line_angle(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:  # 在圆柱展开坐标中计算标线方向角。
    theta_a = math.atan2(point_a[1], point_a[0])  # 计算起点圆周角。
    theta_b = math.atan2(point_b[1], point_b[0])  # 计算终点圆周角。
    delta_theta = _unwrap_angle_difference(theta_b - theta_a)  # 计算连续圆周角差。
    mean_radius = 0.5 * (math.hypot(point_a[0], point_a[1]) + math.hypot(point_b[0], point_b[1]))  # 计算变形前后适用的平均半径。
    circumferential_distance = mean_radius * delta_theta  # 把圆周角差转换为展开面距离。
    axial_distance = point_b[2] - point_a[2]  # 计算轴向距离。
    return math.degrees(math.atan2(circumferential_distance, axial_distance))  # 返回相对轴线的方向角。


def line_rotation(result: SolveResult, beta_deg: float, extraction: str) -> float:  # 计算指定初始方向的加载前后转角幅值。
    true_a = _surface_point(beta_deg, False)  # 计算真实标线起点。
    true_b = _surface_point(beta_deg, True)  # 计算真实标线终点。
    if extraction == "nearest_node":  # 检查是否使用最近节点读取方法。
        reference_a, displacement_a = _nearest_surface_node(result, true_a)  # 把起点吸附到最近表面节点。
        reference_b, displacement_b = _nearest_surface_node(result, true_b)  # 把终点吸附到最近表面节点。
    elif extraction == "surface_interpolation":  # 检查是否使用精确端点插值方法。
        reference_a = true_a  # 保持真实起点位置。
        reference_b = true_b  # 保持真实终点位置。
        displacement_a = _boundary_displacement(reference_a, result.force_n, result.torque_nmm)  # 在真实起点直接评价连续体位移。
        displacement_b = _boundary_displacement(reference_b, result.force_n, result.torque_nmm)  # 在真实终点直接评价连续体位移。
    else:  # 处理未知提取协议。
        raise ValueError("unsupported extraction protocol")  # 拒绝未注册的结果读取方法。
    if reference_a == reference_b:  # 检查粗网格下两个端点是否吸附到同一节点。
        return 0.0  # 对退化标线返回零信号并保留其离散缺陷。
    initial_angle = _line_angle(reference_a, reference_b)  # 计算离散标线加载前方向。
    deformed_a = _deformed_point(reference_a, displacement_a)  # 计算加载后起点。
    deformed_b = _deformed_point(reference_b, displacement_b)  # 计算加载后终点。
    final_angle = _line_angle(deformed_a, deformed_b)  # 计算离散标线加载后方向。
    return abs(final_angle - initial_angle)  # 返回方向变化幅值。


def angle_sweep(result: SolveResult, extraction: str, step_deg: float = 0.5) -> dict[str, Any]:  # 在给定离散场上扫描全部候选标线方向。
    if step_deg <= 0.0:  # 检查扫描步长是否合法。
        raise ValueError("step_deg must be positive")  # 拒绝非正扫描步长。
    count = int(round((ANGLE_MAX_DEG - ANGLE_MIN_DEG) / step_deg)) + 1  # 计算候选方向数量。
    samples: list[dict[str, float]] = []  # 初始化方向响应曲线。
    for index in range(count):  # 遍历全部候选方向。
        beta_deg = ANGLE_MIN_DEG + index * step_deg  # 计算当前候选方向。
        delta_beta_deg = line_rotation(result, beta_deg, extraction)  # 计算当前方向的转角幅值。
        samples.append({"beta_deg": beta_deg, "delta_beta_deg": delta_beta_deg})  # 保存当前黑箱响应。
    best = max(samples, key=lambda item: item["delta_beta_deg"])  # 选择转角幅值最大的候选方向。
    return {"mesh": asdict(result.config), "extraction": extraction, "best_beta_deg": best["beta_deg"], "best_delta_beta_deg": best["delta_beta_deg"], "sample_count": len(samples), "samples": samples, "solver": result.solver}  # 返回完整扫描结果。


def particle_search(result: SolveResult, extraction: str, seed: int = 20260731) -> dict[str, Any]:  # 使用轻量一维粒子群搜索最大转角方向。
    generator = random.Random(seed)  # 创建确定性随机数生成器。
    particle_count = 10  # 设置粒子数量。
    iteration_count = 12  # 设置迭代次数。
    positions = [generator.uniform(ANGLE_MIN_DEG, ANGLE_MAX_DEG) for _ in range(particle_count)]  # 初始化粒子位置。
    velocities = [generator.uniform(-4.0, 4.0) for _ in range(particle_count)]  # 初始化粒子速度。
    personal_best = list(positions)  # 初始化个体最优位置。
    personal_value = [line_rotation(result, position, extraction) for position in positions]  # 评价初始粒子响应。
    global_index = max(range(particle_count), key=lambda index: personal_value[index])  # 找到初始全局最优粒子。
    global_best = personal_best[global_index]  # 保存初始全局最优位置。
    global_value = personal_value[global_index]  # 保存初始全局最优响应。
    trace: list[dict[str, Any]] = []  # 初始化粒子群迭代轨迹。
    evaluations = particle_count  # 记录已完成黑箱评价次数。
    for iteration in range(iteration_count):  # 执行固定轮数粒子群更新。
        for index in range(particle_count):  # 遍历全部粒子。
            inertia = 0.65 * velocities[index]  # 计算惯性速度项。
            cognitive = 1.45 * generator.random() * (personal_best[index] - positions[index])  # 计算个体学习项。
            social = 1.45 * generator.random() * (global_best - positions[index])  # 计算群体学习项。
            velocities[index] = max(-8.0, min(8.0, inertia + cognitive + social))  # 更新并限制粒子速度。
            positions[index] = max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, positions[index] + velocities[index]))  # 更新并限制粒子位置。
            value = line_rotation(result, positions[index], extraction)  # 评价更新后方向响应。
            evaluations += 1  # 累加黑箱评价次数。
            if value > personal_value[index]:  # 检查是否改进个体最优。
                personal_best[index] = positions[index]  # 更新个体最优位置。
                personal_value[index] = value  # 更新个体最优响应。
            if value > global_value:  # 检查是否改进全局最优。
                global_best = positions[index]  # 更新全局最优位置。
                global_value = value  # 更新全局最优响应。
        trace.append({"iteration": iteration + 1, "best_beta_deg": global_best, "best_delta_beta_deg": global_value})  # 保存当前迭代最优结果。
    return {"mesh": asdict(result.config), "extraction": extraction, "best_beta_deg": global_best, "best_delta_beta_deg": global_value, "evaluations": evaluations, "trace": trace, "solver": result.solver}  # 返回完整粒子群搜索结果。


def analytical_optimum(force_n: float, torque_nmm: float) -> dict[str, float]:  # 用高分辨率连续体搜索生成隐藏解析参照。
    reference = solve(MeshConfig(128, 2, 128, 0.0), Path("artifacts") / "shaft_grid_reference", force_n=force_n, torque_nmm=torque_nmm)  # 构造仅供理论参照的连续体场对象。
    scan = angle_sweep(reference, "surface_interpolation", 0.01)  # 使用精确端点和高分辨率角度扫描。
    return {"beta_deg": float(scan["best_beta_deg"]), "delta_beta_deg": float(scan["best_delta_beta_deg"])}  # 返回隐藏最优方向和最大转角。


def compact_scan(scan: dict[str, Any]) -> dict[str, Any]:  # 把完整方向曲线压缩为模型可见工程摘要。
    return {"mesh": scan["mesh"], "extraction": scan["extraction"], "recommended_angle_deg": round(float(scan["best_beta_deg"]), 4), "predicted_angle_change_deg": round(float(scan["best_delta_beta_deg"]), 8), "evaluated_directions": int(scan["sample_count"]), "backend": scan.get("solver", "unknown")}  # 返回工程决策所需紧凑字段。
