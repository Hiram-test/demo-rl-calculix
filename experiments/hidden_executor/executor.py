from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import copy  # 在映射前复制提案以证明执行器不会改写原对象。
import importlib.util  # 从既有实验脚本只读加载有限元后端。
import json  # 保存隐藏映射收据和执行结果。
import re  # 从自然语言实验提案中提取数值参数。
from pathlib import Path  # 管理仓库路径和隔离结果目录。
from types import ModuleType  # 标注动态加载的既有后端模块。
from typing import Any  # 表示动态提案和数值结果结构。

from .contracts import canonical_json  # 使用统一规范序列化验证提案未被修改。
from .contracts import sha256_text  # 计算执行前后的提案内容摘要。
from .contracts import verify_frozen_proposal  # 在隐藏映射前验证冻结文件完整性。

ROOT = Path(__file__).resolve().parents[2]  # 定位仓库根目录并保持新实验完全自包含。
BACKEND_PATH = ROOT / "scripts" / "run_deepseek_crack_open_discovery.py"  # 指向只读复用的既有有限元实验后端。
_INTERNAL_MODULE: ModuleType | None = None  # 缓存动态加载模块以复用真实有限元求解缓存。

INTERNAL_OPERATIONS = {"refine", "fixed_probe", "region_average", "geometry_energy", "closed_form", "request_material", "request_boundary", "finish", "unsupported"}  # 定义模型不可见的内部执行操作。


def _load_backend() -> ModuleType:  # 动态加载既有裂纹有限元后端而不修改其源码。
    global _INTERNAL_MODULE  # 允许函数更新模块缓存。
    if _INTERNAL_MODULE is not None:  # 检查模块是否已经加载。
        return _INTERNAL_MODULE  # 直接复用模块和其求解缓存。
    spec = importlib.util.spec_from_file_location("isolated_crack_backend", BACKEND_PATH)  # 创建独立模块加载规范。
    if spec is None or spec.loader is None:  # 检查后端脚本是否可加载。
        raise RuntimeError("cannot load isolated crack backend")  # 在后端不可用时明确失败。
    module = importlib.util.module_from_spec(spec)  # 创建不污染原脚本命名空间的模块实例。
    spec.loader.exec_module(module)  # 执行既有脚本定义但不会触发其 main 入口。
    _INTERNAL_MODULE = module  # 缓存加载后的后端模块。
    return module  # 返回可调用的有限元后端。


def _proposal_text(proposal: dict[str, Any]) -> str:  # 把提案全部自然语言字段组合为隐藏映射输入。
    return json.dumps(proposal, ensure_ascii=False).lower()  # 使用小写规范文本进行确定性关键词匹配。


def _extract_number(text: str, patterns: list[str], default: float) -> float:  # 从自然语言中按顺序提取第一个数值。
    for pattern in patterns:  # 遍历允许的数值表达式。
        match = re.search(pattern, text, flags=re.IGNORECASE)  # 搜索当前表达式。
        if match is not None:  # 在找到数值时立即返回。
            return float(match.group(1))  # 把捕获内容转换为浮点数。
    return float(default)  # 在未指定参数时返回受控默认值。


def _extract_number_list(text: str, default: list[float]) -> list[float]:  # 从固定位置提案中提取毫米距离数组。
    values = [float(item) for item in re.findall(r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)", text, flags=re.IGNORECASE)]  # 提取全部带单位数值。
    filtered = [value for value in values if 0.0 < value <= 30.0]  # 只保留后端允许的物理距离。
    return filtered[:6] if filtered else list(default)  # 限制数组长度并提供稳定默认值。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 在模型不可见的执行层映射冻结提案。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 从磁盘读取并验证冻结提案。
    before_hash = sha256_text(canonical_json(proposal))  # 记录映射前提案摘要。
    proposal_copy = copy.deepcopy(proposal)  # 创建只用于分类的深拷贝。
    proposal_type = str(proposal_copy.get("proposal_type", ""))  # 读取通用提案类型。
    text = _proposal_text(proposal_copy)  # 组合提案中的全部自然语言字段。
    operation = "unsupported"  # 默认使用不支持状态以避免执行器擅自补路线。
    reason = "proposal could not be mapped without changing its stated purpose"  # 默认记录拒绝映射原因。
    if proposal_type == "stop":  # 处理模型自主停止决定。
        operation = "finish"  # 映射为内部结束操作。
        reason = "proposal explicitly requested stopping"  # 记录直接映射理由。
    elif proposal_type == "request_information":  # 处理模型请求补充事实。
        if any(token in text for token in ("材料", "塑性", "屈服", "硬化", "本构")):  # 判断请求是否针对材料数据。
            operation = "request_material"  # 映射为材料信息请求。
            reason = "proposal requests missing material or constitutive data"  # 记录材料映射理由。
        else:  # 把其他事实请求保守归入载荷或边界事实。
            operation = "request_boundary"  # 映射为载荷边界信息请求。
            reason = "proposal requests missing load, boundary, or model facts"  # 记录事实映射理由。
    elif proposal_type == "experiment":  # 处理模型自由提出的控制实验。
        experiment = proposal_copy.get("experiment", {})  # 读取实验设计对象。
        change_text = str(experiment.get("change", "")).lower()  # 单独读取准备改变的变量。
        measure_text = " ".join(str(item) for item in experiment.get("measure", [])).lower()  # 单独读取准备测量的输出。
        if any(token in change_text for token in ("延长裂纹", "扩展裂纹", "裂纹增长", "裂纹长度", "裂纹增量", "微增裂纹")) and any(token in measure_text + text for token in ("能量", "势能", "应变能", "释放")):  # 识别裂纹几何扰动与能量比较提案。
            operation = "geometry_energy"  # 映射到小几何扰动真实求解。
            reason = "proposal changes crack length and compares an energy quantity"  # 记录物理意图匹配理由。
        elif any(token in text for token in ("固定物理位置", "固定距离", "距裂尖", "测点", "取样位置", "路径点")):  # 识别固定物理位置采样提案。
            operation = "fixed_probe"  # 映射到固定位置场量提取。
            reason = "proposal compares field values at fixed physical locations"  # 记录固定位置映射理由。
        elif any(token in text for token in ("区域平均", "范围平均", "面积平均", "固定半径", "局部区域")):  # 识别固定物理区域聚合提案。
            operation = "region_average"  # 映射到区域平均提取。
            reason = "proposal compares an aggregate over a fixed physical region"  # 记录区域聚合映射理由。
        elif any(token in text for token in ("解析解", "理论解", "闭式解", "理论参照", "理论校核")):  # 识别理论参照提案。
            operation = "closed_form"  # 映射到受限解析参照计算。
            reason = "proposal requests an analytical reference under stated assumptions"  # 记录理论映射理由。
        elif any(token in change_text + text for token in ("加密网格", "细化网格", "减小单元", "增加网格", "nx=")):  # 识别保持模型不变的网格加密提案。
            operation = "refine"  # 映射到新增真实网格求解。
            reason = "proposal changes only mesh resolution while holding the model fixed"  # 记录网格映射理由。
    after_hash = sha256_text(canonical_json(proposal))  # 重新计算原提案摘要以检查未被修改。
    if after_hash != before_hash:  # 检查隐藏映射过程中是否改写了提案。
        raise RuntimeError("executor mutated the frozen proposal")  # 发现任何改写时立即中止。
    receipt = {"proposal_sha256": proposal_hash, "operation": operation, "mapping_reason": reason, "proposal_unchanged": True}  # 组织模型不可见的映射收据。
    (round_dir / "mapping_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 把内部映射记录写入审计文件。
    return receipt  # 返回后续执行阶段使用的内部操作。


def _aligned_geometry_energy(backend: ModuleType, text: str) -> tuple[dict[str, Any], dict[str, Any]]:  # 执行与结构网格对齐的裂纹微增能量比较。
    nx = int(round(_extract_number(text, [r"nx\s*[=:：]?\s*(\d+)"], max(backend.INITIAL_LEVELS))))  # 提取或默认使用当前最细网格。
    requested_extension = _extract_number(text, [r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)[^。；,，]{0,12}(?:延长|扩展|增长|增量)", r"(?:延长|扩展|增长|增量)[^。；,，]{0,12}(\d+(?:\.\d+)?)\s*(?:mm|毫米)"], backend.WIDTH / nx)  # 提取模型提出的裂纹长度变化。
    grid_step = backend.WIDTH / nx  # 计算当前结构网格节点间距。
    step_count = max(1, int(round(requested_extension / grid_step)))  # 把连续提案吸附为整数网格步长。
    used_extension = min(5.0, step_count * grid_step)  # 保证实际扰动不超过后端合同上限。
    if used_extension < grid_step:  # 处理极端上限与网格步长冲突。
        used_extension = grid_step  # 至少使用一个完整网格步长。
    base = backend._solve(nx, backend.HALF_CRACK)  # 求解或读取原裂纹模型。
    extended = backend._solve(nx, backend.HALF_CRACK + used_extension)  # 求解节点对齐的延长裂纹模型。
    added_surface = 2.0 * used_extension * backend.THICKNESS  # 计算新增对称裂纹表面积。
    energy_change = float(extended["strain_energy_n_mm"] - base["strain_energy_n_mm"])  # 计算真实应变能变化。
    raw = {"base_energy_n_mm": float(base["strain_energy_n_mm"]), "extended_energy_n_mm": float(extended["strain_energy_n_mm"]), "energy_change_n_mm": energy_change, "added_crack_surface_mm2": added_surface, "energy_change_per_added_surface_n_per_mm": energy_change / added_surface}  # 组织内部原始数值结果。
    parameters = {"nx": nx, "requested_extension_mm": requested_extension, "used_extension_mm": used_extension, "grid_step_mm": grid_step, "parameter_repair": "snapped_to_integer_mesh_steps"}  # 记录请求参数和实际执行参数。
    return raw, parameters  # 返回原始结果和透明参数修复记录。


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 执行隐藏映射并生成不泄露工具名称的反馈。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 再次验证执行前提案内容完整。
    text = _proposal_text(proposal)  # 组合提案文本用于参数提取。
    backend = _load_backend()  # 加载只读有限元后端。
    operation = str(mapping.get("operation", "unsupported"))  # 读取内部操作标识。
    if operation not in INTERNAL_OPERATIONS:  # 检查映射收据没有注入未知操作。
        raise ValueError("invalid internal operation")  # 拒绝未注册内部操作。
    raw: dict[str, Any] = {}  # 初始化内部原始执行结果。
    parameters: dict[str, Any] = {}  # 初始化实际参数记录。
    executed_change = "未执行任何模型改变"  # 初始化模型可见的物理改变描述。
    limitations: list[str] = []  # 初始化模型可见的限制说明。
    status = "completed"  # 默认执行状态为成功。
    if operation == "refine":  # 执行网格分辨率改变。
        nx = int(round(_extract_number(text, [r"nx\s*[=:：]?\s*(\d+)", r"(\d+)\s*[×x]\s*\1"], 100.0)))  # 从提案提取网格级别。
        nx = min(160, max(20, int(round(nx / 20.0) * 20)))  # 把网格参数修复到受控二十整数倍。
        raw = backend._public_result(backend._solve(nx))  # 执行新增真实有限元求解。
        parameters = {"nx": nx}  # 记录实际网格级别。
        executed_change = f"保持几何、材料、载荷和边界不变，把全局网格划分调整为 {nx}×{nx}"  # 形成模型可见的物理操作说明。
    elif operation == "fixed_probe":  # 执行固定物理位置采样。
        distances = _extract_number_list(text, [4.0, 8.0, 16.0])  # 从提案提取物理距离。
        raw = backend._fixed_location_probe({"distances_mm": distances})  # 在全部已求解网格上读取固定位置应力。
        parameters = {"distances_mm": distances}  # 记录实际采样位置。
        executed_change = "不改变模型，在两个裂尖前方的固定物理距离提取法向应力"  # 描述不随网格移动的采样协议。
    elif operation == "region_average":  # 执行固定区域平均。
        radius = _extract_number(text, [r"半径[^\d]{0,8}(\d+(?:\.\d+)?)\s*(?:mm|毫米)", r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)[^。；,，]{0,8}(?:区域|范围)"], 5.0)  # 从提案提取区域半径。
        radius = min(30.0, max(0.1, radius))  # 把区域半径限制在后端允许范围。
        raw = backend._region_average({"radius_mm": radius})  # 计算固定物理区域平均应力。
        parameters = {"radius_mm": radius}  # 记录实际区域半径。
        executed_change = f"不改变模型，对两个裂尖周围半径 {radius:g} mm 的固定区域计算平均法向应力"  # 描述区域聚合协议。
    elif operation == "geometry_energy":  # 执行裂纹长度微扰能量比较。
        raw, parameters = _aligned_geometry_energy(backend, text)  # 调用带参数吸附记录的真实求解。
        executed_change = f"保持其余模型事实不变，把两侧裂尖各延长 {parameters['used_extension_mm']:g} mm 并比较总应变能"  # 描述实际几何扰动。
        if parameters["requested_extension_mm"] != parameters["used_extension_mm"]:  # 检查执行参数是否发生网格对齐修复。
            limitations.append("提议的连续裂纹增量已吸附到当前结构网格的完整节点步长")  # 向模型透明反馈参数修复。
    elif operation == "closed_form":  # 执行理想化理论参照。
        raw = backend._closed_form()  # 计算既有有限宽中心裂纹参照。
        executed_change = "不改变有限元模型，计算与当前理想化几何和线弹性假设对应的理论参照"  # 描述理论交叉检查。
        limitations.append("理论参照只适用于其列明假设，不能认证真实构件")  # 保留适用范围限制。
    elif operation == "request_material":  # 处理材料信息请求。
        status = "information_required"  # 标记需要外部事实后才能继续。
        raw = {"requested_information": ["屈服强度", "真实应力-塑性应变曲线", "硬化模型", "断裂或损伤参数"]}  # 列出执行所需材料事实。
        executed_change = "未运行新模型，等待补充材料本构和断裂参数"  # 描述当前没有执行求解。
    elif operation == "request_boundary":  # 处理载荷或边界信息请求。
        status = "information_required"  # 标记需要外部模型事实。
        raw = {"requested_information": [proposal.get("information_request", {}).get("question", "请补充真实载荷传递和边界条件")]}  # 保存模型原始问题。
        executed_change = "未运行新模型，等待补充载荷、边界或几何事实"  # 描述当前没有执行求解。
    elif operation == "finish":  # 处理模型自主停止。
        status = "finished"  # 标记本轮实验已经结束。
        raw = {"status": "finished"}  # 保存最小结束记录。
        executed_change = "模型选择基于现有证据结束分析"  # 描述自主停止行为。
    else:  # 处理无法忠实映射的提案。
        status = "unsupported"  # 标记当前执行器无法完成该实验。
        raw = {"reason": mapping.get("mapping_reason")}  # 保存映射拒绝理由。
        executed_change = "当前隔离执行器无法在不改变提案目的的前提下执行该实验"  # 向模型明确报告能力边界。
        limitations.append("执行器没有替换或补写其他实验路线")  # 强调拒绝擅自改写提案。
    public_feedback = {"status": status, "executed_change": executed_change, "actual_parameters": parameters, "observations": raw, "limitations": limitations, "proposal_sha256": proposal_hash}  # 组织不含内部工具名称的下一轮证据反馈。
    audit = {"proposal_sha256": proposal_hash, "internal_operation": operation, "raw_result": raw, "actual_parameters": parameters, "public_feedback": public_feedback}  # 组织完整审计记录。
    (round_dir / "execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存仅供审计的内部操作和原始结果。
    (round_dir / "public_feedback.json").write_text(json.dumps(public_feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存下一轮唯一允许回传给模型的反馈。
    return public_feedback  # 返回经过脱敏的物理证据反馈。
