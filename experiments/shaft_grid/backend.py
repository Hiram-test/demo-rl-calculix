from __future__ import annotations  # 启用现代类型注解并延迟解析前向引用。

import hashlib  # 为有限元配置和缓存目录生成稳定摘要。
import json  # 保存求解缓存、公开观测和隐藏评分结果。
import math  # 计算圆柱坐标、材料常数和角度变化。
import os  # 读取可选的求解器命令配置。
import random  # 为确定性粒子群搜索生成可复现实验粒子。
import re  # 解析 CalculiX 文本结果中的节点位移。
import shutil  # 检查真实 CalculiX 可执行文件是否存在。
import subprocess  # 启动真实 CalculiX 线弹性求解。
from dataclasses import asdict, dataclass  # 定义可序列化的网格配置对象。
from pathlib import Path  # 管理隔离求解缓存和输入输出文件。
from typing import Any  # 标注动态 JSON 结果对象。

RADIUS_MM = 40.0  # 定义圆轴半径并与工程问题中的直径八十毫米一致。
MODEL_LENGTH_MM = 100.0  # 只建远离端部的中部观测段以控制真实求解成本。
YOUNG_MPA = 210000.0  # 使用普通钢材的弹性模量。
POISSON = 0.30  # 使用工程问题给定的泊松比。
VISIBLE_FORCE_N = 180000.0  # 使用非竞赛特殊比例的可见轴向拉力以防背诵答案。
VISIBLE_TORQUE_NMM = 2000000.0  # 使用二千牛米扭矩并转换为牛毫米。
GAUGE_LENGTH_MM = 30.0  # 定义表面短标线的物理长度。
GAUGE_START_Z_MM = 35.0  # 把标线起点放在观测段内部并避开端部。
GAUGE_START_THETA_RAD = 0.10  # 让标线不恰好落在任意标准网格节点上。
ANGLE_MIN_DEG = 0.0  # 定义可搜索的最小初始标线方向。
ANGLE_MAX_DEG = 60.0  # 定义覆盖本实验全部解析最优值的最大方向。


@dataclass(frozen=True)  # 使用不可变对象防止运行中悄然改写网格配置。
class MeshConfig:  # 描述一个真实有限元网格与表面网格方向。
    circumferential: int  # 记录圆周方向单元数量。
    radial: int  # 记录半径方向单元层数。
    axial: int  # 记录轴向单元层数。
    helix_angle_deg: float  # 记录表面轴向网格线相对轴线的螺旋角。

    def validate(self) -> None:  # 在写入有限元输入前检查网格参数合法性。
        if self.circumferential < 12 or self.circumferential % 4 != 0:  # 要求圆周网格足够且四象限兼容。
            raise ValueError("circumferential divisions must be a multiple of four and at least twelve")  # 拒绝不稳定圆周网格。
        if self.radial < 2:  # 要求至少有中心楔形区和一个外环区。
            raise ValueError("radial divisions must be at least two")  # 拒绝无法形成实体截面的配置。
        if self.axial < 4:  # 要求观测段内有足够轴向层数。
            raise ValueError("axial divisions must be at least four")  # 拒绝过粗轴向配置。
        if abs(self.helix_angle_deg) > 35.0:  # 限制初始网格扭斜以避免严重畸变单元。
            raise ValueError("helix angle must remain within plus or minus thirty-five degrees")  # 拒绝高畸变网格。


@dataclass(frozen=True)  # 使用不可变结果对象保存一次真实求解的核心数据。
class SolveResult:  # 描述网格、节点坐标、节点位移和求解审计信息。
    config: MeshConfig  # 保存实际使用的网格配置。
    nodes: dict[int, tuple[float, float, float]]  # 保存全部节点的参考坐标。
    displacements: dict[int, tuple[float, float, float]]  # 保存 CalculiX 返回的节点位移。
    solver: str  # 记录真实求解器身份。
    cache_dir: str  # 记录该配置对应的隔离缓存目录。


def _material_state(force_n: float, torque_nmm: float) -> dict[str, float]:  # 计算给定拉力和扭矩对应的连续体表面应变状态。
    area = math.pi * RADIUS_MM**2  # 计算实心圆轴横截面积。
    polar_moment = math.pi * RADIUS_MM**4 / 2.0  # 计算实心圆轴极惯性矩。
    shear_modulus = YOUNG_MPA / (2.0 * (1.0 + POISSON))  # 由弹性模量和泊松比计算剪切模量。
    axial_strain = force_n / (YOUNG_MPA * area)  # 计算均匀轴向正应变。
    surface_shear = torque_nmm * RADIUS_MM / (shear_modulus * polar_moment)  # 计算圆轴表面的工程剪应变。
    twist_rate = surface_shear / RADIUS_MM  # 把表面剪应变换算为单位长度扭转角。
    return {"area_mm2": area, "polar_moment_mm4": polar_moment, "shear_modulus_mpa": shear_modulus, "axial_strain": axial_strain, "surface_shear_strain": surface_shear, "twist_rate_rad_per_mm": twist_rate}  # 返回全部连续体状态供边界和评分复用。


def _node_key(layer: int, ring: int, sector: int, config: MeshConfig) -> int:  # 为结构化圆柱节点生成稳定整数编号。
    ring_width = 1 + config.radial * config.circumferential  # 计算每个轴向截面的节点数量。
    if ring == 0:  # 单独处理圆心节点以避免重复扇区编号。
        return layer * ring_width + 1  # 返回当前截面唯一圆心节点编号。
    return layer * ring_width + 2 + (ring - 1) * config.circumferential + sector % config.circumferential  # 返回当前圆环扇区节点编号。


def _mesh_rotation_rate(config: MeshConfig) -> float:  # 把表面网格螺旋角转换为截面随轴向的旋转率。
    return math.tan(math.radians(config.helix_angle_deg)) / RADIUS_MM  # 由圆柱展开几何关系计算每毫米旋转弧度。


def _generate_nodes(config: MeshConfig) -> dict[int, tuple[float, float, float]]:  # 生成带可控螺旋方向的结构化圆柱节点。
    config.validate()  # 在生成节点前执行网格合同检查。
    nodes: dict[int, tuple[float, float, float]] = {}  # 初始化节点坐标表。
    rotation_rate = _mesh_rotation_rate(config)  # 读取网格随轴向旋转的速率。
    for layer in range(config.axial + 1):  # 遍历全部轴向截面。
        z_value = MODEL_LENGTH_MM * layer / config.axial  # 计算当前截面轴向坐标。
        phase = rotation_rate * z_value  # 计算当前截面的网格旋转相位。
        nodes[_node_key(layer, 0, 0, config)] = (0.0, 0.0, z_value)  # 写入当前截面的圆心节点。
        for ring in range(1, config.radial + 1):  # 遍历从圆心到表面的同心节点环。
            radius = RADIUS_MM * ring / config.radial  # 计算当前节点环半径。
            for sector in range(config.circumferential):  # 遍历当前节点环的全部扇区。
                theta = 2.0 * math.pi * sector / config.circumferential + phase  # 计算含螺旋相位的节点角坐标。
                node_id = _node_key(layer, ring, sector, config)  # 生成当前节点编号。
                nodes[node_id] = (radius * math.cos(theta), radius * math.sin(theta), z_value)  # 写入当前节点笛卡尔坐标。
    return nodes  # 返回完整参考网格节点表。


def _generate_elements(config: MeshConfig) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:  # 生成中心楔形和外部六面体单元连接关系。
    wedges: list[tuple[int, ...]] = []  # 初始化中心三棱柱单元列表。
    hexes: list[tuple[int, ...]] = []  # 初始化外环八节点六面体单元列表。
    for layer in range(config.axial):  # 遍历相邻轴向截面形成的单元层。
        for sector in range(config.circumferential):  # 遍历圆周全部扇区。
            next_sector = (sector + 1) % config.circumferential  # 计算周期闭合的下一扇区。
            wedges.append((_node_key(layer, 0, 0, config), _node_key(layer, 1, sector, config), _node_key(layer, 1, next_sector, config), _node_key(layer + 1, 0, 0, config), _node_key(layer + 1, 1, sector, config), _node_key(layer + 1, 1, next_sector, config)))  # 连接中心线与第一圆环形成三棱柱。
            for ring in range(1, config.radial):  # 遍历相邻圆环形成六面体。
                hexes.append((_node_key(layer, ring, sector, config), _node_key(layer, ring + 1, sector, config), _node_key(layer, ring + 1, next_sector, config), _node_key(layer, ring, next_sector, config), _node_key(layer + 1, ring, sector, config), _node_key(layer + 1, ring + 1, sector, config), _node_key(layer + 1, ring + 1, next_sector, config), _node_key(layer + 1, ring, next_sector, config)))  # 写入一个外环八节点六面体连接。
    return wedges, hexes  # 返回两类实体单元连接表。


def _boundary_displacement(point: tuple[float, float, float], force_n: float, torque_nmm: float) -> tuple[float, float, float]:  # 计算圣维南拉扭解在端面节点上的等效位移。
    x_value, y_value, z_value = point  # 拆分节点笛卡尔坐标。
    state = _material_state(force_n, torque_nmm)  # 读取当前工况连续体应变状态。
    axial_strain = state["axial_strain"]  # 读取轴向应变。
    twist_rate = state["twist_rate_rad_per_mm"]  # 读取单位长度扭转角。
    displacement_x = -POISSON * axial_strain * x_value + twist_rate * z_value * y_value  # 组合泊松收缩与负向扭转的横向位移。
    displacement_y = -POISSON * axial_strain * y_value - twist_rate * z_value * x_value  # 组合泊松收缩与负向扭转的另一横向位移。
    displacement_z = axial_strain * z_value  # 计算均匀轴向伸长位移。
    return displacement_x, displacement_y, displacement_z  # 返回端面三个方向的规定值。


def _config_hash(config: MeshConfig, force_n: float, torque_nmm: float) -> str:  # 为网格和工况生成稳定缓存键。
    payload = json.dumps({"config": asdict(config), "force_n": force_n, "torque_nmm": torque_nmm}, sort_keys=True, separators=(",", ":"))  # 规范序列化全部求解输入。
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]  # 返回适合作为目录名的短摘要。


def _write_input(path: Path, config: MeshConfig, force_n: float, torque_nmm: float) -> dict[int, tuple[float, float, float]]:  # 写入真实 CalculiX 三维实体输入文件。
    nodes = _generate_nodes(config)  # 生成当前配置的全部节点。
    wedges, hexes = _generate_elements(config)  # 生成两类实体单元。
    lines: list[str] = []  # 初始化 CalculiX 输入文本行。
    lines.append("*HEADING")  # 写入模型标题关键字。
    lines.append("Shaft surface-line grid-direction benchmark")  # 写入模型说明文本。
    lines.append("*NODE, NSET=NALL")  # 开始定义全部节点并建立输出节点集。
    for node_id, point in sorted(nodes.items()):  # 按编号稳定写入节点坐标。
        lines.append(f"{node_id},{point[0]:.12g},{point[1]:.12g},{point[2]:.12g}")  # 写入一个节点的编号和三维坐标。
    element_id = 1  # 初始化全局单元编号。
    lines.append("*ELEMENT, TYPE=C3D6, ELSET=EWEDGE")  # 开始定义中心三棱柱单元集。
    for connectivity in wedges:  # 遍历全部中心楔形单元。
        lines.append(f"{element_id}," + ",".join(str(node) for node in connectivity))  # 写入当前三棱柱连接关系。
        element_id += 1  # 递增全局单元编号。
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EHEX")  # 开始定义外环六面体单元集。
    for connectivity in hexes:  # 遍历全部外环六面体单元。
        lines.append(f"{element_id}," + ",".join(str(node) for node in connectivity))  # 写入当前六面体连接关系。
        element_id += 1  # 递增全局单元编号。
    lines.append("*MATERIAL, NAME=STEEL")  # 定义线弹性钢材。
    lines.append("*ELASTIC")  # 开始写入弹性常数。
    lines.append(f"{YOUNG_MPA},{POISSON}")  # 写入弹性模量和泊松比。
    lines.append("*SOLID SECTION, ELSET=EWEDGE, MATERIAL=STEEL")  # 给中心三棱柱分配钢材实体截面。
    lines.append("")  # 写入 CalculiX 接受的空截面数据行。
    lines.append("*SOLID SECTION, ELSET=EHEX, MATERIAL=STEEL")  # 给外环六面体分配钢材实体截面。
    lines.append("")  # 写入 CalculiX 接受的空截面数据行。
    lines.append("*STEP")  # 开始静力分析步。
    lines.append("*STATIC")  # 选择线性静力求解过程。
    end_tolerance = 1.0e-9  # 定义识别两端节点的几何容差。
    lines.append("*BOUNDARY")  # 开始写入两端圣维南位移边界。
    for node_id, point in sorted(nodes.items()):  # 遍历全部节点并筛选两端截面。
        if abs(point[2]) <= end_tolerance or abs(point[2] - MODEL_LENGTH_MM) <= end_tolerance:  # 检查节点是否位于任一端面。
            prescribed = _boundary_displacement(point, force_n, torque_nmm)  # 计算当前端面节点的三个规定位移。
            for degree, value in enumerate(prescribed, start=1):  # 遍历三个平移自由度。
                lines.append(f"{node_id},{degree},{degree},{value:.16g}")  # 写入一个节点自由度的规定值。
    lines.append("*NODE PRINT, NSET=NALL")  # 请求把全部节点位移写入文本结果文件。
    lines.append("U")  # 指定节点输出量为三向位移。
    lines.append("*END STEP")  # 结束静力分析步。
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 把完整输入写入隔离求解目录。
    return nodes  # 返回节点坐标供结果解析和后处理使用。


def _parse_dat(path: Path, expected_nodes: set[int]) -> dict[int, tuple[float, float, float]]:  # 从 CalculiX 文本结果中提取节点位移。
    if not path.exists():  # 检查求解器是否生成文本结果文件。
        raise RuntimeError("CalculiX did not produce a .dat file")  # 在结果缺失时明确失败。
    pattern = re.compile(r"^\s*(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$")  # 定义节点编号和三个位移分量的严格行模式。
    displacements: dict[int, tuple[float, float, float]] = {}  # 初始化节点位移表。
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():  # 逐行读取求解器文本输出。
        match = pattern.match(line)  # 尝试匹配一个纯节点位移数据行。
        if match is None:  # 跳过标题、空行和其他结果表。
            continue  # 继续检查下一行。
        node_id = int(match.group(1))  # 读取候选节点编号。
        if node_id not in expected_nodes:  # 排除其他表中碰巧符合格式的编号。
            continue  # 跳过非模型节点行。
        displacements[node_id] = (float(match.group(2)), float(match.group(3)), float(match.group(4)))  # 保存当前节点的三个位移分量。
    if len(displacements) != len(expected_nodes):  # 检查是否完整获得全部节点位移。
        missing = sorted(expected_nodes.difference(displacements))[:10]  # 提取少量缺失节点用于诊断。
        raise RuntimeError(f"incomplete CalculiX node displacement table; missing examples: {missing}")  # 拒绝使用不完整结果。
    return displacements  # 返回完整节点位移表。


def solve(config: MeshConfig, cache_root: Path, force_n: float = VISIBLE_FORCE_N, torque_nmm: float = VISIBLE_TORQUE_NMM) -> SolveResult:  # 执行或读取一个真实 CalculiX 网格工况。
    config.validate()  # 在访问缓存前验证网格参数。
    cache_dir = cache_root / _config_hash(config, force_n, torque_nmm)  # 定位当前配置的独立缓存目录。
    cache_dir.mkdir(parents=True, exist_ok=True)  # 创建缓存目录并保留其他配置结果。
    cache_path = cache_dir / "solution.json"  # 定义结构化求解缓存文件。
    if cache_path.exists():  # 检查相同配置是否已经完成真实求解。
        payload = json.loads(cache_path.read_text(encoding="utf-8"))  # 读取结构化缓存。
        nodes = {int(key): tuple(value) for key, value in payload["nodes"].items()}  # 恢复节点坐标并转换键类型。
        displacements = {int(key): tuple(value) for key, value in payload["displacements"].items()}  # 恢复节点位移并转换键类型。
        return SolveResult(config=config, nodes=nodes, displacements=displacements, solver=str(payload["solver"]), cache_dir=str(cache_dir))  # 返回缓存中的真实求解结果。
    solver_command = os.environ.get("CALCULIX_CCX", "ccx")  # 读取可选求解器命令并默认使用 ccx。
    solver_path = shutil.which(solver_command)  # 在当前运行环境中定位真实求解器。
    if solver_path is None:  # 检查 CalculiX 是否已安装。
        raise RuntimeError("CalculiX executable is required for the live shaft-grid experiment")  # 禁止用伪造结果替代真实求解。
    input_path = cache_dir / "shaft.inp"  # 定义当前配置的 CalculiX 输入文件。
    nodes = _write_input(input_path, config, force_n, torque_nmm)  # 生成真实三维实体模型。
    completed = subprocess.run([solver_path, "-i", "shaft"], cwd=cache_dir, text=True, capture_output=True, check=False, timeout=180)  # 启动受限时长的真实 CalculiX 求解。
    (cache_dir / "solver_stdout.txt").write_text(completed.stdout, encoding="utf-8")  # 保存求解器标准输出供审计。
    (cache_dir / "solver_stderr.txt").write_text(completed.stderr, encoding="utf-8")  # 保存求解器错误输出供审计。
    if completed.returncode != 0:  # 检查真实求解是否成功结束。
        raise RuntimeError(f"CalculiX failed with return code {completed.returncode}")  # 在失败时保留目录并停止使用结果。
    displacements = _parse_dat(cache_dir / "shaft.dat", set(nodes))  # 解析全部节点位移结果。
    payload = {"config": asdict(config), "force_n": force_n, "torque_nmm": torque_nmm, "solver": "CalculiX ccx", "nodes": {str(key): list(value) for key, value in nodes.items()}, "displacements": {str(key): list(value) for key, value in displacements.items()}}  # 组织可审计结构化缓存。
    cache_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")  # 保存当前真实求解缓存供后续角度搜索复用。
    return SolveResult(config=config, nodes=nodes, displacements=displacements, solver="CalculiX ccx", cache_dir=str(cache_dir))  # 返回新完成的真实求解结果。


def _wrap_angle(angle: float) -> float:  # 把任意弧度差规范到负π到正π区间。
    return (angle + math.pi) % (2.0 * math.pi) - math.pi  # 返回最短有向角差。


def _surface_node_id(layer: int, sector: int, config: MeshConfig) -> int:  # 返回圆轴最外表面的节点编号。
    return _node_key(layer, config.radial, sector, config)  # 复用结构化节点编号规则定位外圆环。


def _exact_endpoint(beta_deg: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:  # 根据初始方向生成圆柱表面标线的两个精确端点。
    beta = math.radians(beta_deg)  # 把候选方向转换为弧度。
    delta_z = GAUGE_LENGTH_MM * math.cos(beta)  # 计算标线沿轴向的物理投影。
    delta_s = GAUGE_LENGTH_MM * math.sin(beta)  # 计算标线沿圆周展开方向的物理投影。
    first = (RADIUS_MM * math.cos(GAUGE_START_THETA_RAD), RADIUS_MM * math.sin(GAUGE_START_THETA_RAD), GAUGE_START_Z_MM)  # 构造固定起点的笛卡尔坐标。
    second_theta = GAUGE_START_THETA_RAD + delta_s / RADIUS_MM  # 计算第二端点的圆周角坐标。
    second = (RADIUS_MM * math.cos(second_theta), RADIUS_MM * math.sin(second_theta), GAUGE_START_Z_MM + delta_z)  # 构造第二端点的笛卡尔坐标。
    return first, second  # 返回两个精确物理端点。


def _nearest_surface_node(point: tuple[float, float, float], result: SolveResult) -> int:  # 找到给定圆柱表面点最近的结构网格节点。
    theta = math.atan2(point[1], point[0])  # 计算目标点圆周角。
    z_value = point[2]  # 读取目标点轴向坐标。
    layer = min(result.config.axial, max(0, int(round(z_value * result.config.axial / MODEL_LENGTH_MM))))  # 吸附到最近轴向节点层。
    layer_z = MODEL_LENGTH_MM * layer / result.config.axial  # 计算实际节点层轴向坐标。
    phase = _mesh_rotation_rate(result.config) * layer_z  # 计算该节点层的网格相位。
    sector_float = (_wrap_angle(theta - phase) % (2.0 * math.pi)) * result.config.circumferential / (2.0 * math.pi)  # 把目标角转换为当前层的扇区坐标。
    sector = int(round(sector_float)) % result.config.circumferential  # 吸附到最近圆周节点。
    return _surface_node_id(layer, sector, result.config)  # 返回最近表面节点编号。


def _interpolated_displacement(point: tuple[float, float, float], result: SolveResult) -> tuple[float, float, float]:  # 在螺旋表面四节点单元内插值精确标线端点位移。
    theta = math.atan2(point[1], point[0])  # 计算目标点圆周角坐标。
    z_value = min(MODEL_LENGTH_MM, max(0.0, point[2]))  # 把目标轴向坐标限制在模型范围内。
    axial_coordinate = z_value * result.config.axial / MODEL_LENGTH_MM  # 转换为连续轴向网格坐标。
    lower_layer = min(result.config.axial - 1, max(0, int(math.floor(axial_coordinate))))  # 定位目标点下方轴向节点层。
    eta = axial_coordinate - lower_layer  # 计算轴向局部插值坐标。
    phase = _mesh_rotation_rate(result.config) * z_value  # 计算目标轴向位置的连续网格相位。
    sector_coordinate = ((_wrap_angle(theta - phase)) % (2.0 * math.pi)) * result.config.circumferential / (2.0 * math.pi)  # 转换为连续圆周扇区坐标。
    lower_sector = int(math.floor(sector_coordinate)) % result.config.circumferential  # 定位目标点左侧圆周节点线。
    xi = sector_coordinate - math.floor(sector_coordinate)  # 计算圆周局部插值坐标。
    upper_sector = (lower_sector + 1) % result.config.circumferential  # 计算周期闭合的右侧圆周节点线。
    node_ids = (_surface_node_id(lower_layer, lower_sector, result.config), _surface_node_id(lower_layer, upper_sector, result.config), _surface_node_id(lower_layer + 1, lower_sector, result.config), _surface_node_id(lower_layer + 1, upper_sector, result.config))  # 收集当前表面四节点单元。
    weights = ((1.0 - xi) * (1.0 - eta), xi * (1.0 - eta), (1.0 - xi) * eta, xi * eta)  # 计算双线性插值权重。
    values = [0.0, 0.0, 0.0]  # 初始化三个方向的插值位移。
    for node_id, weight in zip(node_ids, weights):  # 遍历四个节点及其权重。
        displacement = result.displacements[node_id]  # 读取当前节点位移。
        for component in range(3):  # 遍历三个平移分量。
            values[component] += weight * displacement[component]  # 累加当前节点对插值位移的贡献。
    return values[0], values[1], values[2]  # 返回精确端点的插值位移。


def _line_angle(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:  # 在圆柱局部展开面内计算一条短线的方向角。
    first_radius = math.hypot(first[0], first[1])  # 计算第一端点到轴线的距离。
    second_radius = math.hypot(second[0], second[1])  # 计算第二端点到轴线的距离。
    first_theta = math.atan2(first[1], first[0])  # 计算第一端点圆周角。
    second_theta = math.atan2(second[1], second[0])  # 计算第二端点圆周角。
    circumferential = 0.5 * (first_radius + second_radius) * _wrap_angle(second_theta - first_theta)  # 计算两端点间局部圆周展开距离。
    axial = second[2] - first[2]  # 计算两端点间轴向距离。
    return math.atan2(circumferential, axial)  # 返回相对轴线的有向方向角。


def delta_beta(result: SolveResult, beta_deg: float, extraction: str) -> float:  # 从真实位移场计算指定初始方向的标线转角幅值。
    exact_first, exact_second = _exact_endpoint(beta_deg)  # 构造候选方向对应的精确物理端点。
    if extraction == "nearest_node":  # 使用常见但可能产生网格吸附的最近节点方法。
        first_id = _nearest_surface_node(exact_first, result)  # 找到起点最近表面节点。
        second_id = _nearest_surface_node(exact_second, result)  # 找到终点最近表面节点。
        reference_first = result.nodes[first_id]  # 使用实际吸附节点作为变形前起点。
        reference_second = result.nodes[second_id]  # 使用实际吸附节点作为变形前终点。
        first_displacement = result.displacements[first_id]  # 读取吸附起点节点位移。
        second_displacement = result.displacements[second_id]  # 读取吸附终点节点位移。
    elif extraction == "surface_interpolation":  # 使用精确端点与表面形函数插值减少网格方向偏置。
        reference_first = exact_first  # 保留标线真实起点位置。
        reference_second = exact_second  # 保留标线真实终点位置。
        first_displacement = _interpolated_displacement(exact_first, result)  # 插值得到精确起点位移。
        second_displacement = _interpolated_displacement(exact_second, result)  # 插值得到精确终点位移。
    else:  # 拒绝未注册的结果提取协议。
        raise ValueError("unsupported extraction protocol")  # 防止静默切换观测定义。
    deformed_first = tuple(reference_first[index] + first_displacement[index] for index in range(3))  # 计算变形后起点坐标。
    deformed_second = tuple(reference_second[index] + second_displacement[index] for index in range(3))  # 计算变形后终点坐标。
    before = _line_angle(reference_first, reference_second)  # 计算实际观测线变形前方向。
    after = _line_angle(deformed_first, deformed_second)  # 计算实际观测线变形后方向。
    return abs(math.degrees(_wrap_angle(after - before)))  # 返回便于工程比较的转角幅值。


def analytical_delta(beta_deg: float, force_n: float, torque_nmm: float) -> float:  # 计算连续体小变形模型中的独立解析转角参照。
    beta = math.radians(beta_deg)  # 把初始方向转换为弧度。
    state = _material_state(force_n, torque_nmm)  # 读取当前载荷对应的连续体应变状态。
    initial_axial = math.cos(beta)  # 取单位材料线的轴向分量。
    initial_circumferential = math.sin(beta)  # 取单位材料线的圆周分量。
    final_axial = (1.0 + state["axial_strain"]) * initial_axial  # 计算拉伸后材料线轴向分量。
    final_circumferential = -state["surface_shear_strain"] * initial_axial + (1.0 - POISSON * state["axial_strain"]) * initial_circumferential  # 计算扭转和泊松收缩后的圆周分量。
    final_beta = math.atan2(final_circumferential, final_axial)  # 计算连续体模型中的变形后方向。
    return abs(math.degrees(_wrap_angle(final_beta - beta)))  # 返回解析转角幅值。


def analytical_optimum(force_n: float, torque_nmm: float) -> dict[str, float]:  # 通过高分辨率独立扫描获得隐藏解析真值。
    best_beta = ANGLE_MIN_DEG  # 初始化解析最优方向。
    best_value = -1.0  # 初始化解析最大转角。
    steps = 6000  # 使用足够密集的角度网格消除评分器离散误差。
    for index in range(steps + 1):  # 遍历完整候选方向区间。
        beta = ANGLE_MIN_DEG + (ANGLE_MAX_DEG - ANGLE_MIN_DEG) * index / steps  # 计算当前解析候选方向。
        value = analytical_delta(beta, force_n, torque_nmm)  # 计算当前候选方向解析转角。
        if value > best_value:  # 检查当前候选是否改进最大值。
            best_beta = beta  # 更新解析最优方向。
            best_value = value  # 更新解析最大转角。
    return {"beta_deg": best_beta, "delta_beta_deg": best_value}  # 返回隐藏真值供最终盲评使用。


def angle_sweep(result: SolveResult, extraction: str, step_deg: float = 0.5) -> dict[str, Any]:  # 在同一真实位移场上廉价扫描全部标线方向。
    samples: list[dict[str, float]] = []  # 初始化角度响应样本表。
    beta = ANGLE_MIN_DEG  # 从最小方向开始扫描。
    while beta <= ANGLE_MAX_DEG + 1.0e-9:  # 遍历包含右端点的候选区间。
        samples.append({"beta_deg": round(beta, 8), "delta_beta_deg": delta_beta(result, beta, extraction)})  # 保存当前方向及其真实有限元转角。
        beta += step_deg  # 递增扫描方向。
    best = max(samples, key=lambda item: item["delta_beta_deg"])  # 找到离散扫描中的最大响应点。
    return {"mesh": asdict(result.config), "extraction": extraction, "best_beta_deg": best["beta_deg"], "best_delta_beta_deg": best["delta_beta_deg"], "sample_count": len(samples), "samples": samples}  # 返回扫描摘要和完整响应曲线。


def particle_search(result: SolveResult, extraction: str, seed: int = 20260731) -> dict[str, Any]:  # 使用一维连续粒子群在真实有限元后处理中搜索最大转角。
    generator = random.Random(seed)  # 创建不影响全局状态的确定性随机数生成器。
    particle_count = 12  # 设置足够覆盖一维目标的粒子数量。
    iteration_count = 18  # 设置有限但可复现的粒子更新轮数。
    positions = [generator.uniform(ANGLE_MIN_DEG, ANGLE_MAX_DEG) for _ in range(particle_count)]  # 随机初始化全部粒子方向。
    velocities = [generator.uniform(-6.0, 6.0) for _ in range(particle_count)]  # 随机初始化受限角速度。
    personal_positions = list(positions)  # 初始化每个粒子的历史最佳位置。
    personal_values = [delta_beta(result, position, extraction) for position in positions]  # 评价全部初始粒子。
    global_index = max(range(particle_count), key=lambda index: personal_values[index])  # 找到初始全局最佳粒子。
    global_position = personal_positions[global_index]  # 保存初始全局最佳方向。
    global_value = personal_values[global_index]  # 保存初始全局最大转角。
    trace: list[dict[str, float]] = [{"iteration": 0, "best_beta_deg": global_position, "best_delta_beta_deg": global_value}]  # 记录初始粒子群状态。
    for iteration in range(1, iteration_count + 1):  # 逐轮更新粒子群。
        for index in range(particle_count):  # 遍历全部粒子。
            inertia = 0.62 * velocities[index]  # 计算当前速度的惯性分量。
            cognitive = 1.45 * generator.random() * (personal_positions[index] - positions[index])  # 计算粒子自身经验吸引分量。
            social = 1.55 * generator.random() * (global_position - positions[index])  # 计算群体最佳吸引分量。
            velocities[index] = max(-8.0, min(8.0, inertia + cognitive + social))  # 更新并限制粒子角速度。
            positions[index] = max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, positions[index] + velocities[index]))  # 更新并限制粒子方向。
            value = delta_beta(result, positions[index], extraction)  # 评价更新后的真实有限元目标值。
            if value > personal_values[index]:  # 检查粒子是否刷新自身历史最佳。
                personal_positions[index] = positions[index]  # 更新粒子历史最佳方向。
                personal_values[index] = value  # 更新粒子历史最大转角。
                if value > global_value:  # 检查粒子是否刷新群体全局最佳。
                    global_position = positions[index]  # 更新群体最佳方向。
                    global_value = value  # 更新群体最大转角。
        trace.append({"iteration": iteration, "best_beta_deg": global_position, "best_delta_beta_deg": global_value})  # 保存当前轮粒子群最佳状态。
    return {"mesh": asdict(result.config), "extraction": extraction, "best_beta_deg": global_position, "best_delta_beta_deg": global_value, "evaluations": particle_count * (iteration_count + 1), "trace": trace}  # 返回连续黑箱搜索结果。


def compact_scan(scan: dict[str, Any]) -> dict[str, Any]:  # 把完整角度曲线压缩为可回传给模型的工程观测。
    return {"mesh": scan["mesh"], "extraction": scan["extraction"], "recommended_angle_deg": round(float(scan["best_beta_deg"]), 4), "predicted_angle_change_deg": round(float(scan["best_delta_beta_deg"]), 8), "evaluated_directions": int(scan["sample_count"])}  # 只公开工程决策所需的紧凑字段。
