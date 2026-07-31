from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

from math import pi  # 计算Irwin塑性区尺寸所需圆周系数。
from typing import Any  # 表示动态Skill参数、上下文和公开结果。

from experiments.skill_planner import skills as previous_skills  # 复用已经通过真实双API验证的Skill目录。
from experiments.skill_planner.registry import SkillContext  # 使用统一Skill运行上下文。
from experiments.skill_planner.registry import SkillDefinition  # 声明新增Skill输入、输出和限制。
from experiments.skill_planner.registry import SkillRegistry  # 返回扩展后的不可变Skill注册表。


def _irwin_plastic_zone(arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:  # 根据冻结提案中的K和屈服强度执行Irwin条件敏感性计算。
    del context  # 该Skill只使用已由确定性来源门验证的显式参数。
    stress_intensity = float(arguments["stress_intensity_mpa_sqrt_mm"])  # 读取已有线弹性应力强度因子。
    yield_strengths = [float(value) for value in arguments["yield_strengths_mpa"]]  # 读取冻结提案明确提出的屈服强度序列。
    half_crack = float(arguments["half_crack_mm"])  # 读取裂纹半长用于小范围屈服比值。
    ligament = float(arguments["ligament_mm"])  # 读取剩余韧带尺度用于边界影响比值。
    plane_condition = str(arguments.get("plane_condition", "plane_stress"))  # 读取平面应力或平面应变条件。
    if stress_intensity <= 0.0:  # 检查K必须具有正物理量级。
        raise ValueError("stress_intensity_mpa_sqrt_mm must be positive")  # 拒绝无效断裂参量。
    if not yield_strengths or any(value <= 0.0 for value in yield_strengths):  # 检查屈服强度序列必须非空且为正。
        raise ValueError("yield_strengths_mpa must contain positive values")  # 拒绝无效材料假设。
    if half_crack <= 0.0 or ligament <= 0.0:  # 检查几何尺度必须为正。
        raise ValueError("half_crack_mm and ligament_mm must be positive")  # 拒绝无效几何尺度。
    if plane_condition not in {"plane_stress", "plane_strain"}:  # 检查二维约束条件属于Skill合同。
        raise ValueError("plane_condition must be plane_stress or plane_strain")  # 拒绝未知公式分支。
    coefficient = 1.0 / (2.0 * pi) if plane_condition == "plane_stress" else 1.0 / (6.0 * pi)  # 选择Irwin一阶塑性区系数。
    rows: list[dict[str, Any]] = []  # 初始化每个屈服强度的公开条件结果。
    for yield_strength in yield_strengths:  # 逐个计算冻结提案要求的材料敏感性点。
        radius = float(coefficient * (stress_intensity / yield_strength) ** 2)  # 计算当前条件下的Irwin塑性区半径。
        crack_ratio = float(100.0 * radius / half_crack)  # 计算塑性区相对裂纹半长百分比。
        ligament_ratio = float(100.0 * radius / ligament)  # 计算塑性区相对剩余韧带百分比。
        rows.append({"yield_strength_mpa": yield_strength, "plastic_zone_radius_mm": radius, "plastic_zone_to_half_crack_percent": crack_ratio, "plastic_zone_to_ligament_percent": ligament_ratio})  # 保存完整条件结果。
    return {"status": "completed", "executed_change": "不运行新的有限元模型，使用已有K值和冻结提案中的屈服强度假设计算Irwin塑性区敏感性", "actual_parameters": {"stress_intensity_mpa_sqrt_mm": stress_intensity, "yield_strengths_mpa": yield_strengths, "half_crack_mm": half_crack, "ligament_mm": ligament, "plane_condition": plane_condition}, "observations": {"formula": "rp=(1/(2*pi))*(K/sigma_y)^2 for plane stress; rp=(1/(6*pi))*(K/sigma_y)^2 for plane strain", "rows": rows}, "limitations": ["这是线弹性断裂力学的一阶塑性区估算，不等同于弹塑性有限元结果", "屈服强度若为假设值，结论只能作为条件敏感性而不能替代真实材料数据", "未包含硬化、约束三维效应和裂尖钝化"]}  # 返回不泄露Skill名称的物理后处理证据。


def build_registry() -> SkillRegistry:  # 构造包含Irwin条件评估能力的扩展隐藏Skill目录。
    registry = previous_skills.build_registry()  # 复用已验证的能量、位移、网格、Richardson和材料请求Skill。
    registry.register(SkillDefinition(skill_id="fracture.irwin_plastic_zone", description="使用已有模式I应力强度因子和一个或多个明确屈服强度，计算平面应力或平面应变Irwin塑性区尺寸及其相对裂纹和韧带比值。", input_schema={"stress_intensity_mpa_sqrt_mm": {"type": "number", "required": True, "description": "来自冻结提案或前轮公开证据的模式I应力强度因子。"}, "yield_strengths_mpa": {"type": "array[number]", "required": True, "description": "冻结提案明确提出或外部数据提供的屈服强度序列。"}, "half_crack_mm": {"type": "number", "required": True, "description": "裂纹半长。"}, "ligament_mm": {"type": "number", "required": True, "description": "裂尖至最近自由边界的剩余韧带尺度。"}, "plane_condition": {"type": "string", "required": False, "description": "plane_stress或plane_strain。"}}, output_fields=["plastic_zone_radius_mm", "plastic_zone_to_half_crack_percent", "plastic_zone_to_ligament_percent"], effects=["只读取已有K值和显式材料假设", "不运行新的有限元模型"], limitations=["一阶Irwin估算不能替代弹塑性有限元", "假设屈服强度只能形成条件结论"], handler=_irwin_plastic_zone))  # 注册新的模型适用性后处理Skill。
    return registry  # 返回扩展后的隐藏Skill注册表。
