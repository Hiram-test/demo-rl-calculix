#!/usr/bin/env python3  # 使用仓库环境中的 Python 运行开放发现实验。
from __future__ import annotations  # 允许使用现代类型注解并保持向前兼容。

import argparse  # 解析实验模式、模型和预算参数。
import hashlib  # 为实际提示词生成可审计摘要。
import json  # 读写模型消息、工具结果和完整轨迹。
import os  # 从环境变量读取 DeepSeek 密钥与模型名。
import sys  # 把仓库根目录加入模块搜索路径。
from math import cos, pi, sqrt  # 计算有限宽裂纹的解析参照量。
from pathlib import Path  # 统一处理输入输出目录与指南文件。
from typing import Any  # 表示模型 JSON 和工具结果的动态字段。

import numpy as np  # 处理网格坐标、位移和应力数组。
from openai import OpenAI  # 通过兼容接口调用 DeepSeek。

ROOT = Path(__file__).resolve().parents[1]  # 定位仓库根目录。
sys.path.insert(0, str(ROOT))  # 允许直接导入仓库内的有限元模块。

from bridge_mesh_suite.fem import edge_traction_load  # 构造上下边缘均布拉力。
from bridge_mesh_suite.fem import nearest_node  # 定位最小刚体约束所需节点。
from bridge_mesh_suite.fem import solve_linear_plane_stress  # 执行二维线弹性有限元求解。
from bridge_mesh_suite.meshes import central_crack_mesh  # 生成中心贯穿裂纹结构网格。

WIDTH = 200.0  # 定义试件宽度，单位为毫米。
HEIGHT = 200.0  # 定义试件高度，单位为毫米。
HALF_CRACK = 20.0  # 定义初始半裂纹长度，单位为毫米。
YOUNG = 210000.0  # 定义钢材弹性模量，单位为兆帕。
POISSON = 0.30  # 定义钢材泊松比。
THICKNESS = 10.0  # 定义平面应力模型厚度，单位为毫米。
REMOTE_STRESS = 100.0  # 定义上下边缘远场拉应力，单位为兆帕。
INITIAL_LEVELS = (40, 60, 80)  # 提供三档初始网格证据但不计算断裂专用量。
ALLOWED_ACTIONS = {  # 列出可调用工具而不规定任何必选顺序。
    "refine_current_model",  # 在相同模型假设下增加一档网格证据。
    "probe_fixed_physical_locations",  # 在固定物理距离比较应力而非跟随最近积分点。
    "compare_region_average",  # 比较裂尖附近固定物理区域的平均响应。
    "compare_nearby_geometry_energy",  # 对一个小几何扰动比较总势能变化。
    "crosscheck_with_closed_form",  # 请求一个适用范围受限的解析交叉检查。
    "request_material_curve",  # 在怀疑材料非线性时请求真实材料数据。
    "request_load_or_boundary_detail",  # 在载荷或边界不清楚时请求补充信息。
    "finish",  # 在证据足以形成暂定答复时主动结束。
}  # 完成开放工具目录定义。

BARE_PRINCIPLES = (  # 定义不包含有限元领域知识的通用对照提示。
    "你是科学证据分析代理。你必须根据用户问题和逐轮返回的数据形成至少两个竞争假设，"  # 要求竞争假设而不暗示具体路线。
    "区分直接观测、解释和未知，每轮只选择一个最能区分假设的下一步动作，"  # 约束为最小可判别实验。
    "引用实际证据字段，证据不足时允许请求信息或停止并保留不确定性。"  # 允许诚实停止并禁止编造事实。
)  # 完成通用对照原则。
GUIDED_PRINCIPLES = (  # 定义工程师高层知识框架而不指定裂纹路线。
    "你是有限元与结构工程证据分析代理。先固定用户要作出的工程决策、评价位置和可接受误差，"  # 强调工程决策与固定评价协议。
    "再区分直接观测、物理解释和未知。点值、尖角、裂尖、点载荷和约束附近的量可能不随网格收敛，"  # 提供工程师应掌握的通用奇异性知识。
    "因此应同时检查固定物理位置或区域的响应、位移、能量、平衡、路径或积分量以及模型假设的适用性。"  # 提供广义证据类别但不点名目标工具。
    "每轮形成至少两个竞争假设，只选择一个控制变量清楚且最能区分假设的最小下一步计算。"  # 要求实验设计而不规定计算类型。
    "工具目录只是工作台，任何工具都不是必选项；你可以继续加密、改变评价协议、请求缺失信息或直接停止。"  # 明确禁止把工具存在视为剧本。
)  # 完成工程师高层原则。
OUTPUT_CONTRACT = (  # 定义可审计短理由而不要求输出隐藏思维链。
    "只输出一个合法 JSON 对象，字段必须为："  # 限制输出格式便于执行与审计。
    "competing_hypotheses 数组且至少两个；"  # 要求竞争假设。
    "evidence_refs 数组，元素必须引用证据包中的字段；"  # 要求证据锚定。
    "uncertainties 数组；"  # 要求保留未知。
    "controlled_variables 数组；"  # 要求说明控制变量。
    "engineering_reason 字符串，只写可公开的简短工程理由；"  # 保存决策摘要而不保存私密长推理。
    "action 对象，包含 name 和 arguments；"  # 输出唯一下一步动作。
    "provisional_answer 字符串。"  # 每轮保留当前暂定答复。
)  # 完成输出合同。
TOOL_DESCRIPTIONS = {  # 定义模型可见的开放工具说明。
    "refine_current_model": {"arguments": {"nx": "40到140之间且能被20整除的整数"}, "returns": "相同几何、材料、载荷和边界下的新网格峰值、远场位移、总能量和平衡误差"},  # 提供继续加密选项。
    "probe_fixed_physical_locations": {"arguments": {"distances_mm": "正数数组"}, "returns": "所有已求解网格在裂尖前方固定物理距离处的法向应力"},  # 提供固定位置取样选项。
    "compare_region_average": {"arguments": {"radius_mm": "正数"}, "returns": "所有已求解网格在两个裂尖固定半径区域内的平均法向应力"},  # 提供区域平均选项。
    "compare_nearby_geometry_energy": {"arguments": {"nx": "已求解或允许的新网格等级", "extension_mm": "正数且不超过5毫米"}, "returns": "保持其余条件不变时，小幅延长裂纹前后的总能量及单位新增裂纹表面的能量变化"},  # 提供几何扰动能量选项。
    "crosscheck_with_closed_form": {"arguments": {}, "returns": "当前理想化几何与线弹性假设下的有限宽解析参照及适用性警告"},  # 提供解析校核选项。
    "request_material_curve": {"arguments": {"reason": "请求原因"}, "returns": "记录缺失材料数据，当前不执行非线性求解"},  # 提供材料信息请求选项。
    "request_load_or_boundary_detail": {"arguments": {"question": "需要用户补充的问题"}, "returns": "记录缺失的载荷或边界事实"},  # 提供模型事实请求选项。
    "finish": {"arguments": {}, "returns": "结束实验并冻结当前暂定答复"},  # 提供自由终止选项。
}  # 完成工具说明目录。

_SOLVE_CACHE: dict[tuple[int, float], dict[str, Any]] = {}  # 缓存真实有限元结果以避免重复求解。


def _minimal_constraints(mesh: Any) -> dict[int, float]:  # 构造只消除刚体运动的最小约束。
    xmin, ymin = mesh.nodes.min(axis=0)  # 读取左下边界坐标。
    xmax, _ = mesh.nodes.max(axis=0)  # 读取右下边界横坐标。
    left_bottom = nearest_node(mesh, (float(xmin), float(ymin)))  # 定位左下节点。
    right_bottom = nearest_node(mesh, (float(xmax), float(ymin)))  # 定位右下节点。
    return {2 * left_bottom: 0.0, 2 * left_bottom + 1: 0.0, 2 * right_bottom + 1: 0.0}  # 固定左下双向与右下竖向自由度。


def _tension_load(mesh: Any) -> np.ndarray:  # 构造保持合力对称的上下边缘拉力。
    top = edge_traction_load(mesh, mesh.edge_sets["top"], (0.0, REMOTE_STRESS), THICKNESS)  # 在上边缘施加向上拉力。
    bottom = edge_traction_load(mesh, mesh.edge_sets["bottom"], (0.0, -REMOTE_STRESS), THICKNESS)  # 在下边缘施加向下拉力。
    return top + bottom  # 返回总载荷向量。


def _remote_opening(mesh: Any, solution: Any) -> float:  # 计算上下边缘平均竖向位移差作为远场响应。
    top_nodes = sorted({node for edge in mesh.edge_sets["top"] for node in edge})  # 收集上边缘节点。
    bottom_nodes = sorted({node for edge in mesh.edge_sets["bottom"] for node in edge})  # 收集下边缘节点。
    top_mean = float(np.mean(solution.displacements[top_nodes, 1]))  # 计算上边缘平均竖向位移。
    bottom_mean = float(np.mean(solution.displacements[bottom_nodes, 1]))  # 计算下边缘平均竖向位移。
    return top_mean - bottom_mean  # 返回上下边缘相对张开量。


def _solve(nx: int, half_crack: float = HALF_CRACK) -> dict[str, Any]:  # 执行或复用指定网格和裂纹长度的真实求解。
    key = (int(nx), round(float(half_crack), 8))  # 构造稳定缓存键。
    if key in _SOLVE_CACHE:  # 检查是否已有相同求解结果。
        return _SOLVE_CACHE[key]  # 直接返回缓存结果。
    if nx < 20 or nx > 160 or nx % 20 != 0:  # 限制网格等级以控制计算预算。
        raise ValueError("nx must be between 20 and 160 and divisible by 20")  # 拒绝不受控的网格参数。
    mesh = central_crack_mesh(WIDTH, HEIGHT, half_crack, nx, nx)  # 生成裂纹与网格边对齐的结构网格。
    solution = solve_linear_plane_stress(mesh, YOUNG, POISSON, THICKNESS, _tension_load(mesh), _minimal_constraints(mesh))  # 完成线弹性平面应力求解。
    dx = WIDTH / nx  # 计算当前局部网格尺度。
    tips = np.asarray(((half_crack, 0.0), (-half_crack, 0.0)), dtype=float)  # 定义两个裂尖坐标。
    distances = np.min(np.linalg.norm(solution.element_centers[:, None, :] - tips[None, :, :], axis=2), axis=1)  # 计算单元中心到最近裂尖的距离。
    near_mask = distances <= 3.5 * dx  # 定义跟随网格变化的裂尖近场区域以复现峰值趋势。
    result = {  # 组织模型每轮可见的基础证据。
        "nx": int(nx),  # 记录横纵网格划分数。
        "elements": int(len(mesh.elements)),  # 记录单元数量。
        "h_local_mm": float(dx),  # 记录裂尖附近特征尺寸。
        "half_crack_mm": float(half_crack),  # 记录半裂纹长度。
        "tip_peak_sigma_y_mpa": float(np.max(solution.element_stress[near_mask, 1])),  # 记录裂尖附近法向应力峰值。
        "remote_opening_mm": float(_remote_opening(mesh, solution)),  # 记录远场张开响应。
        "strain_energy_n_mm": float(solution.strain_energy),  # 记录总应变能。
        "energy_balance_relative": float(solution.energy_balance_rel),  # 记录内外功平衡误差。
        "mesh": mesh,  # 在执行层缓存网格但不会直接发送给模型。
        "solution": solution,  # 在执行层缓存解对象但不会直接发送给模型。
    }  # 完成基础结果组织。
    _SOLVE_CACHE[key] = result  # 把真实求解写入缓存。
    return result  # 返回基础求解结果。


def _public_result(result: dict[str, Any]) -> dict[str, Any]:  # 移除不可序列化的网格和解对象。
    return {key: value for key, value in result.items() if key not in {"mesh", "solution"}}  # 只保留模型可见的数值字段。


def _initial_evidence() -> dict[str, Any]:  # 构造不含断裂专用结论的初始证据包。
    history = [_public_result(_solve(nx)) for nx in INITIAL_LEVELS]  # 运行三档相同模型的真实网格序列。
    return {  # 返回用户问题、模型事实和初始网格证据。
        "user_question": "钢板有一条约40 mm贯穿裂纹，裂尖附近网格从5 mm细化到2.5 mm后最大应力一直升高。我还要继续加密吗？当前模型能不能用于判断裂纹问题，下一步网格具体怎么处理？",  # 提供自然语言工程问题。
        "model_facts": {"geometry": "200 mm × 200 mm中心裂纹钢板", "half_crack_mm": HALF_CRACK, "thickness_mm": THICKNESS, "material": {"young_mpa": YOUNG, "poisson": POISSON, "plastic_curve": None}, "loading": "上下边缘100 MPa对称均布拉应力", "analysis": "二维平面应力线弹性"},  # 提供已知模型事实并明确塑性曲线缺失。
        "initial_mesh_history": history,  # 提供峰值、远场位移、能量和平衡证据。
    }  # 完成初始证据包。


def _fixed_location_probe(arguments: dict[str, Any]) -> dict[str, Any]:  # 执行固定物理距离取样工具。
    distances = [float(value) for value in arguments.get("distances_mm", [2.5, 5.0, 10.0])]  # 读取或使用默认取样距离。
    if not distances or any(value <= 0.0 or value > 30.0 for value in distances):  # 验证距离范围。
        raise ValueError("distances_mm must contain values in (0, 30]")  # 拒绝无效取样距离。
    rows: list[dict[str, Any]] = []  # 初始化固定位置结果列表。
    for nx in sorted({key[0] for key in _SOLVE_CACHE if abs(key[1] - HALF_CRACK) < 1.0e-8}):  # 遍历已经求解的原裂纹网格。
        result = _solve(nx)  # 读取缓存中的网格和解。
        centers = result["solution"].element_centers  # 读取单元中心坐标。
        stress_y = result["solution"].element_stress[:, 1]  # 读取法向应力。
        samples: dict[str, float] = {}  # 初始化当前网格的固定距离样本。
        for distance in distances:  # 逐个处理物理取样距离。
            targets = np.asarray(((HALF_CRACK + distance, 0.0), (-HALF_CRACK - distance, 0.0)), dtype=float)  # 定义两个裂尖前方的对称目标点。
            nearest = [int(np.argmin(np.linalg.norm(centers - target, axis=1))) for target in targets]  # 找到最接近两个目标点的单元。
            samples[f"distance_{distance:g}_mm_mean_sigma_y_mpa"] = float(np.mean(stress_y[nearest]))  # 记录两个对称点的平均应力。
        rows.append({"nx": nx, "h_local_mm": WIDTH / nx, "samples": samples})  # 保存当前网格固定位置结果。
    return {"tool": "probe_fixed_physical_locations", "rows": rows}  # 返回完整固定位置证据。


def _region_average(arguments: dict[str, Any]) -> dict[str, Any]:  # 执行固定半径区域平均工具。
    radius = float(arguments.get("radius_mm", 5.0))  # 读取或使用默认物理半径。
    if radius <= 0.0 or radius > 30.0:  # 验证区域半径范围。
        raise ValueError("radius_mm must lie in (0, 30]")  # 拒绝无效区域半径。
    rows: list[dict[str, Any]] = []  # 初始化区域平均结果列表。
    tips = np.asarray(((HALF_CRACK, 0.0), (-HALF_CRACK, 0.0)), dtype=float)  # 定义两个裂尖位置。
    for nx in sorted({key[0] for key in _SOLVE_CACHE if abs(key[1] - HALF_CRACK) < 1.0e-8}):  # 遍历已求解网格。
        result = _solve(nx)  # 读取当前网格结果。
        centers = result["solution"].element_centers  # 读取单元中心坐标。
        distance = np.min(np.linalg.norm(centers[:, None, :] - tips[None, :, :], axis=2), axis=1)  # 计算到最近裂尖的物理距离。
        mask = distance <= radius  # 选择固定物理半径内的单元。
        rows.append({"nx": nx, "radius_mm": radius, "mean_sigma_y_mpa": float(np.mean(result["solution"].element_stress[mask, 1])), "element_samples": int(np.sum(mask))})  # 保存区域平均应力与样本数。
    return {"tool": "compare_region_average", "rows": rows}  # 返回固定区域平均证据。


def _geometry_energy(arguments: dict[str, Any]) -> dict[str, Any]:  # 执行小裂纹延长前后的能量比较。
    nx = int(arguments.get("nx", max(INITIAL_LEVELS)))  # 读取或使用当前最细网格。
    extension = float(arguments.get("extension_mm", WIDTH / nx))  # 读取或使用一个网格尺度的裂纹延长量。
    if extension <= 0.0 or extension > 5.0:  # 验证几何扰动幅度。
        raise ValueError("extension_mm must lie in (0, 5]")  # 拒绝过大的几何改变。
    base = _solve(nx, HALF_CRACK)  # 求解或读取原裂纹模型。
    extended = _solve(nx, HALF_CRACK + extension)  # 求解延长后的裂纹模型。
    added_surface = 2.0 * extension * THICKNESS  # 计算两个对称裂尖新增裂纹表面积。
    energy_change = float(extended["strain_energy_n_mm"] - base["strain_energy_n_mm"])  # 计算总应变能变化。
    return {"tool": "compare_nearby_geometry_energy", "nx": nx, "extension_mm": extension, "base_energy_n_mm": float(base["strain_energy_n_mm"]), "extended_energy_n_mm": float(extended["strain_energy_n_mm"]), "energy_change_n_mm": energy_change, "added_crack_surface_mm2": added_surface, "energy_change_per_added_surface_n_per_mm": energy_change / added_surface, "controlled_variables": ["外形尺寸", "材料参数", "远场应力", "边界约束", "网格划分数"]}  # 返回几何扰动能量证据与控制变量。


def _closed_form() -> dict[str, Any]:  # 计算有限宽中心裂纹的线弹性解析参照。
    half_width = WIDTH / 2.0  # 计算板的半宽。
    correction = sqrt(1.0 / cos(pi * HALF_CRACK / (2.0 * half_width)))  # 计算有限宽修正因子。
    k_reference = REMOTE_STRESS * sqrt(pi * HALF_CRACK) * correction  # 计算模式I应力强度参照。
    energy_reference = k_reference**2 / YOUNG  # 在平面应力假设下换算能量参照。
    return {"tool": "crosscheck_with_closed_form", "stress_intensity_reference_mpa_sqrt_mm": k_reference, "energy_reference_n_per_mm": energy_reference, "assumptions": ["中心贯穿裂纹", "线弹性", "均匀远场拉应力", "二维平面应力", "不含塑性区修正"], "warning": "该参照只能检查理想化模型量级，不能认证真实构件或替代材料非线性分析"}  # 返回解析参照与适用范围警告。


def _execute_action(action: dict[str, Any]) -> dict[str, Any]:  # 验证并执行模型选择的唯一动作。
    name = str(action.get("name", ""))  # 读取动作名称。
    arguments = action.get("arguments", {})  # 读取动作参数。
    if name not in ALLOWED_ACTIONS:  # 检查动作是否属于开放工具目录。
        raise ValueError(f"unknown action: {name}")  # 拒绝未注册动作。
    if not isinstance(arguments, dict):  # 检查参数必须为 JSON 对象。
        raise ValueError("action.arguments must be an object")  # 拒绝无效参数结构。
    if name == "refine_current_model":  # 处理同类网格加密动作。
        return {"tool": name, "result": _public_result(_solve(int(arguments.get("nx", 100))))}  # 返回新增真实求解证据。
    if name == "probe_fixed_physical_locations":  # 处理固定位置取样动作。
        return _fixed_location_probe(arguments)  # 返回固定位置趋势。
    if name == "compare_region_average":  # 处理固定区域平均动作。
        return _region_average(arguments)  # 返回区域平均趋势。
    if name == "compare_nearby_geometry_energy":  # 处理小几何扰动能量动作。
        return _geometry_energy(arguments)  # 返回能量变化证据。
    if name == "crosscheck_with_closed_form":  # 处理解析交叉检查动作。
        return _closed_form()  # 返回解析参照。
    if name == "request_material_curve":  # 处理材料数据请求动作。
        return {"tool": name, "status": "information_required", "requested": ["屈服强度", "真实应力-塑性应变曲线", "硬化模型", "断裂或损伤参数"], "reason": str(arguments.get("reason", "模型认为材料非线性可能影响工程决策"))}  # 记录缺失材料信息。
    if name == "request_load_or_boundary_detail":  # 处理载荷或边界信息请求动作。
        return {"tool": name, "status": "information_required", "question": str(arguments.get("question", "请补充真实载荷传递和边界约束"))}  # 记录需要用户补充的问题。
    return {"tool": "finish", "status": "finished"}  # 处理主动结束动作。


def _validate_decision(decision: Any) -> list[str]:  # 对模型 JSON 进行结构验证而不检查路线。
    errors: list[str] = []  # 初始化错误列表。
    if not isinstance(decision, dict):  # 检查顶层必须为对象。
        return ["response must be an object"]  # 返回顶层结构错误。
    hypotheses = decision.get("competing_hypotheses")  # 读取竞争假设。
    if not isinstance(hypotheses, list) or len(hypotheses) < 2:  # 检查至少两个竞争假设。
        errors.append("competing_hypotheses must contain at least two items")  # 记录假设数量错误。
    for key in ("evidence_refs", "uncertainties", "controlled_variables"):  # 遍历必须为数组的通用字段。
        if not isinstance(decision.get(key), list):  # 检查字段类型。
            errors.append(f"{key} must be an array")  # 记录字段类型错误。
    if not isinstance(decision.get("engineering_reason"), str):  # 检查公开工程理由。
        errors.append("engineering_reason must be a string")  # 记录理由字段错误。
    if not isinstance(decision.get("provisional_answer"), str):  # 检查暂定答复。
        errors.append("provisional_answer must be a string")  # 记录答复字段错误。
    action = decision.get("action")  # 读取动作对象。
    if not isinstance(action, dict) or action.get("name") not in ALLOWED_ACTIONS or not isinstance(action.get("arguments", {}), dict):  # 检查动作名称与参数结构。
        errors.append("action must name one registered tool and contain object arguments")  # 记录动作合同错误。
    return errors  # 返回纯结构验证结果。


def _call_model(client: OpenAI, model: str, system_prompt: str, packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:  # 请求一次开放决策并验证结构。
    user_prompt = "请根据以下实时证据选择唯一下一步动作。不得假设存在必选路线，也不得补造未提供事实。\n\n" + json.dumps(packet, ensure_ascii=False, indent=2)  # 组合当前完整证据包。
    last_error = "unknown"  # 初始化重试错误信息。
    for attempt in range(2):  # 最多允许一次结构重试。
        prompt = user_prompt if attempt == 0 else "上一次 JSON 结构无效，请只修复结构，不要改变证据事实。\n\n" + user_prompt  # 第二次只要求修复结构。
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], response_format={"type": "json_object"}, max_tokens=8000, reasoning_effort="high", extra_body={"thinking": {"type": "enabled"}}, temperature=0.2, stream=False)  # 调用真实 DeepSeek 并启用高推理预算。
        content = response.choices[0].message.content or ""  # 读取公开 JSON 内容。
        try:  # 尝试解析模型返回。
            decision = json.loads(content)  # 把 JSON 文本转换为对象。
        except json.JSONDecodeError as exc:  # 捕获非法 JSON。
            last_error = str(exc)  # 保存解析错误用于最终诊断。
            continue  # 进入结构重试。
        errors = _validate_decision(decision)  # 执行不含路线要求的结构验证。
        if errors:  # 检查是否存在合同错误。
            last_error = "; ".join(errors)  # 保存合同错误。
            continue  # 进入结构重试。
        metadata = {"requested_model": model, "response_model": response.model, "finish_reason": response.choices[0].finish_reason, "usage": response.usage.model_dump() if response.usage is not None else {}}  # 保存调用元数据但不保存隐藏推理。
        return decision, metadata  # 返回有效决策和审计元数据。
    raise RuntimeError(f"DeepSeek did not return a valid decision: {last_error}")  # 两次无效后终止本组实验。


def _generic_audit(trace: list[dict[str, Any]]) -> dict[str, Any]:  # 事后执行不要求固定 Skill 的通用审计。
    decisions = [row["decision"] for row in trace]  # 提取所有模型决策。
    actions = [row["decision"]["action"]["name"] for row in trace]  # 提取实际动作序列。
    return {"rounds": len(trace), "action_sequence": actions, "used_multiple_hypotheses_every_round": all(len(item.get("competing_hypotheses", [])) >= 2 for item in decisions), "cited_evidence_every_round": all(bool(item.get("evidence_refs")) for item in decisions), "retained_uncertainty_every_round": all(isinstance(item.get("uncertainties"), list) and bool(item.get("uncertainties")) for item in decisions), "declared_controlled_variables_every_round": all(isinstance(item.get("controlled_variables"), list) for item in decisions), "stopped_voluntarily": bool(actions and actions[-1] == "finish"), "route_specific_requirement_used": False}  # 返回通用可审计指标与实际动作序列。


def _run_mode(mode: str, client: OpenAI, model: str, max_rounds: int) -> dict[str, Any]:  # 运行一个 guided 或 bare 开放发现轨迹。
    principles = GUIDED_PRINCIPLES if mode == "guided" else BARE_PRINCIPLES  # 选择高层工程指南或通用对照指南。
    system_prompt = principles + OUTPUT_CONTRACT  # 组合运行时系统提示且不写入场景路线。
    evidence = _initial_evidence()  # 生成相同的初始真实有限元证据。
    trace: list[dict[str, Any]] = []  # 初始化逐轮审计轨迹。
    tool_results: list[dict[str, Any]] = []  # 初始化工具返回历史。
    for round_index in range(1, max_rounds + 1):  # 在固定预算内逐轮决策。
        packet = {"mode": mode, "round": round_index, "remaining_rounds": max_rounds - round_index + 1, "evidence": evidence, "previous_tool_results": tool_results, "available_tools": TOOL_DESCRIPTIONS}  # 向模型提供完整当前状态和开放工具目录。
        decision, metadata = _call_model(client, model, system_prompt, packet)  # 请求模型选择唯一下一步动作。
        tool_result = _execute_action(decision["action"])  # 执行模型自主选择的动作。
        trace.append({"round": round_index, "decision": decision, "tool_result": tool_result, "metadata": metadata})  # 保存公开理由、动作和真实工具反馈。
        tool_results.append(tool_result)  # 把工具反馈加入下一轮证据。
        if decision["action"]["name"] == "finish":  # 检查模型是否主动结束。
            break  # 冻结当前轨迹并停止继续驱动。
        if tool_result.get("status") == "information_required":  # 检查是否需要现实中无法自动补齐的信息。
            break  # 保留请求信息结论并停止自动循环。
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()  # 计算实际系统提示摘要以便审计。
    return {"mode": mode, "model": model, "system_prompt": system_prompt, "system_prompt_sha256": prompt_hash, "initial_evidence": evidence, "trace": trace, "audit": _generic_audit(trace), "final_provisional_answer": trace[-1]["decision"]["provisional_answer"] if trace else "未获得有效决策"}  # 返回完整可复现实验记录。


def main() -> int:  # 解析命令行并运行选定实验组。
    parser = argparse.ArgumentParser()  # 创建命令行解析器。
    parser.add_argument("--mode", choices=("guided", "bare", "both"), default="both")  # 选择工程指南组、通用对照组或两组。
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))  # 选择真实 DeepSeek 模型。
    parser.add_argument("--max-rounds", type=int, default=6)  # 设置最大决策轮数。
    parser.add_argument("--output", type=Path, default=Path("artifacts/deepseek_crack_open_discovery.json"))  # 设置轨迹输出路径。
    args = parser.parse_args()  # 读取命令行参数。
    if args.max_rounds < 1 or args.max_rounds > 10:  # 验证决策预算范围。
        raise ValueError("max-rounds must lie between 1 and 10")  # 拒绝不合理预算。
    api_key = os.environ.get("DEEPSEEK_API_KEY")  # 读取 GitHub Secret 注入的 API 密钥。
    if not api_key:  # 检查密钥是否存在。
        raise RuntimeError("DEEPSEEK_API_KEY is required for the live discovery experiment")  # 明确拒绝伪造模型输出。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")  # 创建 DeepSeek 兼容客户端。
    modes = ("guided", "bare") if args.mode == "both" else (args.mode,)  # 解析需要运行的实验组。
    results = [_run_mode(mode, client, args.model, args.max_rounds) for mode in modes]  # 顺序运行两组以共享相同求解缓存。
    output = {"experiment": "deepseek_crack_open_discovery", "runtime_route_requirements": [], "results": results}  # 组织顶层实验结果并显式记录无路线要求。
    args.output.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录。
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 写入完整 JSON 审计轨迹。
    print(json.dumps({"status": "completed", "output": str(args.output), "modes": list(modes), "actions": {result["mode"]: result["audit"]["action_sequence"] for result in results}}, ensure_ascii=False, indent=2))  # 在日志中输出简洁动作序列摘要。
    return 0  # 返回成功状态码。


if __name__ == "__main__":  # 仅在脚本直接执行时启动实验。
    raise SystemExit(main())  # 把主函数返回值交给操作系统。
