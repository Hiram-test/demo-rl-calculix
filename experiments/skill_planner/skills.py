from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

from math import sqrt  # 依据平面应力线弹性关系换算应力强度因子。
from typing import Any  # 表示Skill参数、证据上下文和公开结果。

import numpy as np  # 计算序列差异、均值和Richardson拟合。
from scipy.optimize import least_squares  # 对非等比网格执行广义Richardson外推。

from experiments.hidden_executor.executor import _load_backend  # 只读加载既有真实裂纹有限元后端。
from experiments.hidden_executor.executor_adapter_v6 import _displacement_rows  # 复用已经验证的真实裂纹面节点位移提取。
from experiments.skill_planner.registry import SkillContext  # 使用统一Skill运行上下文。
from experiments.skill_planner.registry import SkillDefinition  # 定义隐藏Skill能力合同。
from experiments.skill_planner.registry import SkillRegistry  # 构造第二API可见的隐藏Skill目录。


def _fracture_value(backend: Any, nx: int) -> dict[str, Any]:  # 在指定网格上用节点对齐裂纹微增计算G和K。
    extension = float(backend.WIDTH / nx)  # 使用当前结构网格一个完整步长作为裂纹增量。
    base = backend._solve(nx, backend.HALF_CRACK)  # 读取或求解原始裂纹模型。
    extended = backend._solve(nx, backend.HALF_CRACK + extension)  # 求解裂纹延长一个网格步长后的模型。
    energy_change = float(extended["strain_energy_n_mm"] - base["strain_energy_n_mm"])  # 计算总应变能变化。
    added_surface = float(2.0 * extension * backend.THICKNESS)  # 计算两个对称裂尖新增裂纹表面积。
    energy_release = float(energy_change / added_surface)  # 用有限裂纹增量近似能量释放率。
    stress_intensity = float(sqrt(max(energy_release, 0.0) * backend.YOUNG))  # 按二维平面应力关系计算K。
    return {"nx": int(nx), "h_local_mm": float(backend.WIDTH / nx), "crack_extension_mm": extension, "base_strain_energy_n_mm": float(base["strain_energy_n_mm"]), "extended_strain_energy_n_mm": float(extended["strain_energy_n_mm"]), "energy_release_rate_n_per_mm": energy_release, "stress_intensity_mpa_sqrt_mm": stress_intensity}  # 返回完整可审计断裂参量行。


def _fracture_energy_sequence(arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:  # 对多档网格计算能量差G和K序列。
    del context  # 该Skill只使用真实后端和显式参数，不读取模型历史。
    backend = _load_backend()  # 加载真实有限元后端和求解缓存。
    requested_levels = arguments.get("mesh_levels", list(backend.INITIAL_LEVELS))  # 读取第二API指定网格或使用初始三档。
    levels = [min(160, max(20, int(round(float(value) / 20.0) * 20))) for value in requested_levels]  # 把网格吸附到后端允许的二十整数倍。
    levels = list(dict.fromkeys(levels))  # 保持顺序并删除重复网格等级。
    if len(levels) < 2:  # 检查收敛比较至少需要两档网格。
        raise ValueError("fracture energy sequence requires at least two mesh levels")  # 拒绝无法比较的单点请求。
    rows = [_fracture_value(backend, nx) for nx in levels]  # 执行全部真实裂纹微增计算。
    finest = float(rows[-1]["stress_intensity_mpa_sqrt_mm"])  # 读取最后一档作为当前最细参考值。
    for row in rows:  # 遍历结果计算相对最细网格差异。
        value = float(row["stress_intensity_mpa_sqrt_mm"])  # 读取当前K值。
        row["difference_from_finest_percent"] = float(100.0 * (value - finest) / finest) if finest else 0.0  # 保存相对差异百分比。
    return {"status": "completed", "executed_change": "保持几何、材料弹性参数、载荷和边界不变，对多档网格分别计算裂纹微增前后的能量差并换算G与K", "actual_parameters": {"mesh_levels": levels, "extension_rule": "one grid step at each mesh level", "plane_condition": "plane_stress", "young_mpa": float(backend.YOUNG)}, "observations": {"method": "每档网格将两个裂尖各延长本档一个结构网格步长，以应变能差除以新增裂纹表面积得到G，再用K=sqrt(EG)换算", "rows": rows}, "limitations": ["不同网格使用各自一个网格步长作为裂纹增量，因此收敛比较同时包含空间离散和有限差分步长效应", "该方法属于有限裂纹增量能量差，不是轮廓J积分或位移外推", "模型保持二维平面应力线弹性，尚未验证真实材料塑性"]}  # 返回不含Skill名称的公开物理证据。


def _refine_and_fracture(arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:  # 在新目标网格上复算G和K并与基准比较。
    del context  # 该Skill只使用显式目标和真实后端。
    backend = _load_backend()  # 加载真实有限元后端。
    target_h = float(arguments["target_h_mm"])  # 读取第二API从冻结提案提取的目标尺寸。
    if target_h <= 0.0:  # 检查目标网格尺寸物理有效性。
        raise ValueError("target_h_mm must be positive")  # 拒绝非正尺寸。
    baseline_nx = int(arguments.get("baseline_nx", max(backend.INITIAL_LEVELS)))  # 读取基准网格或使用初始最密档。
    baseline_nx = min(160, max(20, int(round(baseline_nx / 20.0) * 20)))  # 吸附基准网格到后端合同。
    target_nx = int(round((backend.WIDTH / target_h) / 20.0) * 20)  # 把目标尺寸转换为结构网格划分数。
    target_nx = min(160, max(20, target_nx))  # 限制在已验证真实求解预算范围内。
    baseline = _fracture_value(backend, baseline_nx)  # 计算基准网格断裂参量。
    refined = _fracture_value(backend, target_nx)  # 计算目标网格断裂参量。
    baseline_k = float(baseline["stress_intensity_mpa_sqrt_mm"])  # 读取基准K值。
    refined_k = float(refined["stress_intensity_mpa_sqrt_mm"])  # 读取目标K值。
    change = float(100.0 * (refined_k - baseline_k) / baseline_k) if baseline_k else 0.0  # 计算有符号相对变化。
    used_h = float(backend.WIDTH / target_nx)  # 计算实际使用的结构网格尺寸。
    limitations = ["两档网格使用各自一个网格步长作为裂纹增量", "该方法使用能量差换算而非轮廓积分", "模型保持二维平面应力线弹性"]  # 声明固有方法边界。
    if abs(used_h - target_h) > 1.0e-9:  # 检查目标尺寸是否发生后端参数吸附。
        limitations.append("请求的目标尺寸已吸附到后端允许的结构网格划分")  # 透明报告实际参数修复。
    return {"status": "completed", "executed_change": f"保持几何、材料、载荷和边界不变，把网格从{baseline['h_local_mm']:g} mm调整到{used_h:g} mm，并按相同能量差协议复算G和K", "actual_parameters": {"requested_target_h_mm": target_h, "used_target_h_mm": used_h, "used_target_nx": target_nx, "baseline_nx": baseline_nx}, "observations": {"baseline": baseline, "refined": refined, "relative_change_percent": change}, "limitations": limitations}  # 返回目标加密与断裂量联合证据。


def _mesh_refine(arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:  # 在模型假设不变时执行一档真实网格求解。
    del context  # 该Skill只使用显式参数和真实后端。
    backend = _load_backend()  # 加载真实有限元后端。
    has_target = "target_h_mm" in arguments  # 检查是否以物理尺寸指定网格。
    has_nx = "nx" in arguments  # 检查是否以划分数指定网格。
    if has_target == has_nx:  # 检查两个参数必须且只能提供一个。
        raise ValueError("mesh refine requires exactly one of target_h_mm or nx")  # 拒绝歧义或缺失参数。
    requested_h = float(arguments["target_h_mm"]) if has_target else float(backend.WIDTH / int(arguments["nx"]))  # 统一转换为目标尺寸。
    requested_nx = int(round(backend.WIDTH / requested_h))  # 把目标尺寸转换为原始划分数。
    used_nx = min(160, max(20, int(round(requested_nx / 20.0) * 20)))  # 吸附到后端允许网格等级。
    raw = backend._public_result(backend._solve(used_nx))  # 执行真实求解并提取公开有限元结果。
    used_h = float(backend.WIDTH / used_nx)  # 计算实际网格尺寸。
    limitations: list[str] = []  # 初始化参数修复说明。
    if abs(used_h - requested_h) > 1.0e-9:  # 检查请求值和实际值是否一致。
        limitations.append("目标网格已吸附到后端允许的二十整数倍划分")  # 透明报告参数修复。
    return {"status": "completed", "executed_change": f"保持几何、材料、载荷和边界不变，把结构网格调整为{used_nx}×{used_nx}", "actual_parameters": {"requested_target_h_mm": requested_h, "used_target_h_mm": used_h, "used_nx": used_nx}, "observations": raw, "limitations": limitations}  # 返回真实网格证据。


def _crack_face_displacement(arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:  # 提取真实裂纹面节点位移并可选换算位移法K。
    del context  # 该Skill直接读取真实后端位移解。
    backend = _load_backend()  # 加载真实有限元后端和网格对象。
    distances = [float(value) for value in arguments.get("distances_mm", [2.5])]  # 读取距裂尖目标位置或使用2.5毫米默认值。
    derive_k = bool(arguments.get("derive_k", False))  # 读取是否需要按裂纹面张开位移换算K。
    if not distances or any(value <= 0.0 or value > backend.HALF_CRACK for value in distances):  # 检查距离范围。
        raise ValueError("distances_mm must lie inside the crack face")  # 拒绝裂纹面外取样。
    rows = _displacement_rows(backend, distances, derive_k)  # 调用已经验证的真实上下裂纹面节点位移提取。
    observations: dict[str, Any] = {"rows": rows}  # 组织公开位移证据。
    if derive_k:  # 检查是否需要汇总位移法K的网格差异。
        values = [float(row["stress_intensity_from_opening_mpa_sqrt_mm"]) for row in rows]  # 提取全部位移法K值。
        mean_value = float(np.mean(values)) if values else 0.0  # 计算序列平均值。
        observations["relative_spread_percent"] = float(100.0 * (max(values) - min(values)) / mean_value) if mean_value else None  # 计算相对离散度。
    return {"status": "completed", "executed_change": "不改变有限元模型，提取三档网格裂纹面真实节点坐标和张开位移" + ("并换算位移法K" if derive_k else ""), "actual_parameters": {"requested_distances_mm": distances, "derive_k": derive_k}, "observations": observations, "limitations": ["实际取样位置必须吸附到每档网格已有裂纹面节点", "位移法K使用二维平面应力线弹性裂尖渐近场", "若取样点不在K主导区或塑性区不可忽略，绝对值可能失真"]}  # 返回不泄露Skill名称的真实位移证据。


def _find_rows_with_field(value: Any, field: str) -> list[dict[str, Any]]:  # 递归查找公开历史中同时含网格尺寸和目标字段的结果行。
    found: list[dict[str, Any]] = []  # 初始化匹配行列表。
    if isinstance(value, dict):  # 检查当前值是否为对象。
        if field in value and "h_local_mm" in value:  # 检查当前对象是否可用于网格外推。
            found.append(value)  # 保存包含目标量和网格尺寸的行。
        for child in value.values():  # 遍历对象全部子值。
            found.extend(_find_rows_with_field(child, field))  # 递归收集子结构匹配行。
    if isinstance(value, list):  # 检查当前值是否为数组。
        for child in value:  # 遍历数组元素。
            found.extend(_find_rows_with_field(child, field))  # 递归收集数组中的匹配行。
    return found  # 返回全部候选结果行。


def _richardson_extrapolation(arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:  # 对已有网格序列执行广义Richardson外推。
    source_field = str(arguments["source_field"])  # 读取需要外推的公开物理字段名。
    requested_round = arguments.get("source_round")  # 读取可选指定历史轮次。
    history = context.public_history  # 读取第一API可见的真实公开历史。
    candidates = [item for item in history if requested_round is None or int(item.get("round", -1)) == int(requested_round)]  # 按可选轮次筛选证据。
    rows: list[dict[str, Any]] = []  # 初始化目标量网格序列。
    for item in reversed(candidates):  # 从最新公开轮次向前搜索。
        rows = _find_rows_with_field(item.get("execution_feedback", {}), source_field)  # 查找当前轮次目标字段。
        if len(rows) >= 3:  # 检查是否已经找到足够外推的网格点。
            break  # 使用最新满足条件的真实序列。
    unique: dict[float, float] = {}  # 按网格尺寸去重目标值。
    for row in rows:  # 遍历候选结果行。
        unique[float(row["h_local_mm"])] = float(row[source_field])  # 保存每个网格尺寸最后出现的真实值。
    if len(unique) < 3:  # 检查广义外推至少需要三档网格。
        raise ValueError(f"at least three mesh levels containing {source_field} are required")  # 拒绝证据不足的外推。
    ordered = sorted(unique.items(), reverse=True)  # 按从粗到细排列网格尺寸和结果值。
    h = np.asarray([item[0] for item in ordered], dtype=float)  # 构造网格尺寸数组。
    values = np.asarray([item[1] for item in ordered], dtype=float)  # 构造目标量数组。
    initial_f0 = float(values[-1])  # 使用最细网格值初始化连续极限。
    initial_c = float((values[0] - values[-1]) / max(h[0], 1.0e-12))  # 使用粗细差初始化误差系数。
    def residual(parameters: np.ndarray) -> np.ndarray:  # 定义非等比网格的幂律离散误差残差。
        f0, coefficient, order = parameters  # 读取连续极限、误差系数和观测阶次。
        return f0 + coefficient * np.power(h, order) - values  # 返回各档网格拟合残差。
    fit = least_squares(residual, x0=np.asarray([initial_f0, initial_c, 1.0]), bounds=(np.asarray([-np.inf, -np.inf, 0.05]), np.asarray([np.inf, np.inf, 10.0])))  # 在正观测阶次范围内拟合三参数模型。
    if not fit.success:  # 检查数值外推是否收敛。
        raise RuntimeError("Richardson extrapolation fit did not converge")  # 拒绝伪造外推结果。
    extrapolated, coefficient, observed_order = [float(value) for value in fit.x]  # 读取拟合参数。
    finest = float(values[-1])  # 读取最细网格原始值。
    estimated_error = float(100.0 * abs(finest - extrapolated) / abs(extrapolated)) if extrapolated else None  # 估计最细网格相对连续极限误差。
    return {"status": "completed", "executed_change": "不运行新的有限元模型，使用已有三档以上网格结果执行广义Richardson外推", "actual_parameters": {"source_field": source_field, "source_round": requested_round}, "observations": {"input_rows": [{"h_local_mm": float(item[0]), source_field: float(item[1])} for item in ordered], "extrapolated_value": extrapolated, "observed_order": observed_order, "error_coefficient": coefficient, "finest_relative_error_percent": estimated_error, "fit_residual_norm": float(np.linalg.norm(fit.fun))}, "limitations": ["外推假设目标量的主导离散误差可由单一幂律表示", "只有三档网格时拟合对数值噪声和非渐近区较敏感", "外推只能评估数值离散误差，不能验证物理模型适用性"]}  # 返回纯后处理证据。


def _material_request(arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:  # 记录隐藏执行器无法自行生成的外部材料事实。
    del context  # 外部数据请求不读取或修改有限元结果。
    fields = [str(value) for value in arguments["fields"]]  # 读取第二API忠实拆出的材料需求清单。
    return {"status": "information_required", "executed_change": "未运行新的有限元模型，记录完成工程适用性判断所需的外部材料事实", "actual_parameters": {}, "observations": {"requested_information": fields}, "limitations": ["这些数据不能由当前线弹性有限元结果反推出，必须由用户、材料标准或试验提供"]}  # 返回真实外部阻塞而不重复计算已有量。


def build_registry() -> SkillRegistry:  # 构造当前隐藏Skill目录和确定性处理函数。
    registry = SkillRegistry()  # 初始化空Skill注册表。
    registry.register(SkillDefinition(skill_id="fracture.energy_sequence", description="对两个及以上真实网格分别执行节点对齐裂纹微增能量差计算，返回G、K和网格差异。", input_schema={"mesh_levels": {"type": "array[number]", "required": False, "description": "结构网格划分数；缺省时使用初始证据中的三档网格。"}}, output_fields=["energy_release_rate_n_per_mm", "stress_intensity_mpa_sqrt_mm", "difference_from_finest_percent"], effects=["改变裂纹长度一个网格步长并执行多个线弹性求解", "保持原几何、材料、载荷和边界作为基准"], limitations=["有限裂纹增量随网格变化", "二维平面应力线弹性"], handler=_fracture_energy_sequence))  # 注册多档能量断裂参量Skill。
    registry.register(SkillDefinition(skill_id="fracture.refine_and_energy", description="把网格调整到明确目标尺寸，并在基准和目标网格上按同一能量差协议复算G和K。", input_schema={"target_h_mm": {"type": "number", "required": True, "description": "冻结提案明确给出的目标结构网格尺寸。"}, "baseline_nx": {"type": "integer", "required": False, "description": "基准网格划分数。"}}, output_fields=["baseline", "refined", "relative_change_percent"], effects=["改变网格分辨率", "对基准和目标网格执行裂纹微增求解"], limitations=["目标尺寸可能吸附到后端允许等级", "二维平面应力线弹性"], handler=_refine_and_fracture))  # 注册加密与断裂参量联合Skill。
    registry.register(SkillDefinition(skill_id="mesh.refine", description="保持物理模型不变，执行一个明确网格尺寸或划分数的真实有限元求解。", input_schema={"target_h_mm": {"type": "number", "required": False, "description": "目标结构网格尺寸。"}, "nx": {"type": "integer", "required": False, "description": "目标结构网格划分数。"}}, output_fields=["tip_peak_sigma_y_mpa", "remote_opening_mm", "strain_energy_n_mm", "energy_balance_relative"], effects=["改变网格分辨率", "保持几何、材料、载荷和边界不变"], limitations=["仅支持后端已验证的结构网格等级"], handler=_mesh_refine))  # 注册普通网格求解Skill。
    registry.register(SkillDefinition(skill_id="fracture.crack_face_displacement", description="从真实有限元解提取上下裂纹面节点坐标和张开位移，并可按平面应力裂尖位移场换算K。", input_schema={"distances_mm": {"type": "array[number]", "required": False, "description": "距两个裂尖向裂纹内部的目标物理距离。"}, "derive_k": {"type": "boolean", "required": False, "description": "是否由半张开位移换算模式I应力强度因子。"}}, output_fields=["tip_samples", "mean_half_opening_mm", "stress_intensity_from_opening_mpa_sqrt_mm", "relative_spread_percent"], effects=["只读提取节点位移", "不改变有限元模型"], limitations=["取样位置吸附到已有节点", "要求取样点位于K主导区且塑性可忽略"], handler=_crack_face_displacement))  # 注册真实裂纹面位移Skill。
    registry.register(SkillDefinition(skill_id="postprocess.richardson", description="从前轮公开网格序列读取指定物理字段，执行非等比网格的广义Richardson外推。", input_schema={"source_field": {"type": "string", "required": True, "description": "前轮公开结果中的目标字段名。"}, "source_round": {"type": "integer", "required": False, "description": "可选指定证据轮次。"}}, output_fields=["extrapolated_value", "observed_order", "finest_relative_error_percent", "fit_residual_norm"], effects=["只读取已有公开证据", "不运行新的有限元模型"], limitations=["假设主导离散误差服从单一幂律", "三点拟合对非渐近区敏感"], handler=_richardson_extrapolation))  # 注册通用网格外推Skill。
    registry.register(SkillDefinition(skill_id="material.request", description="记录必须由用户、标准或试验提供的材料屈服、塑性、硬化或断裂数据。", input_schema={"fields": {"type": "array[string]", "required": True, "description": "冻结提案明确请求的外部材料事实。"}}, output_fields=["requested_information"], effects=["不运行有限元模型", "生成外部数据阻塞"], limitations=["不能由当前线弹性结果反推材料事实"], handler=_material_request))  # 注册外部材料信息Skill。
    return registry  # 返回完整隐藏Skill注册表。
