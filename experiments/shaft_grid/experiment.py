from __future__ import annotations  # 启用现代类型注解并避免运行时前向引用问题。

import copy  # 在隐藏映射前复制冻结提案以证明不会改写原意。
import json  # 保存实验状态、映射收据、公开反馈和最终评分。
import re  # 从工程语言提案与最终答复中识别方向、网格和优化意图。
from pathlib import Path  # 管理隔离实验输出目录和真实求解缓存。
from typing import Any  # 标注模型生成的动态 JSON 结构。

from experiments.hidden_executor.contracts import canonical_json  # 复用 PR 十八的稳定 JSON 规范。
from experiments.hidden_executor.contracts import sha256_text  # 复用冻结提案摘要算法。
from experiments.hidden_executor.contracts import verify_frozen_proposal  # 在隐藏执行前重新验证提案完整性。
from experiments.shaft_grid.backend import MeshConfig  # 导入圆轴网格配置合同。
from experiments.shaft_grid.backend import VISIBLE_FORCE_N  # 导入工程问题可见轴向拉力。
from experiments.shaft_grid.backend import VISIBLE_TORQUE_NMM  # 导入工程问题可见扭矩。
from experiments.shaft_grid.backend import analytical_optimum  # 导入仅供隐藏评分与独立校核的解析参照。
from experiments.shaft_grid.backend import angle_sweep  # 导入同一位移场上的廉价方向扫描。
from experiments.shaft_grid.backend import compact_scan  # 导入模型可见的紧凑角度响应摘要。
from experiments.shaft_grid.backend import particle_search  # 导入真实有限元黑箱上的一维粒子群搜索。
from experiments.shaft_grid.backend import solve  # 导入真实 CalculiX 圆轴求解后端。

USER_QUESTION = (  # 定义只呈现工程结果诉求的用户问题。
    "我们在做一根圆钢轴的拉扭试验，轴的直径是80毫米，材料按普通钢材考虑。"  # 用常见工程语言交代对象和材料。
    "试验时轴上同时承受180千牛的拉力和2.0千牛米的扭矩，我们准备在轴中间表面画一小段斜线，"  # 交代普通组合载荷和表面标线。
    "通过相机观察受力以后这条线转过了多少角度。现在主要想知道这条线一开始画成多少度，"  # 直接提出需要的方向结果。
    "受力后的角度变化会比较明显，方便实际测量。我们用有限元试算过几次，"  # 说明信号越明显越便于测量。
    "但换了网格大小或者网格排列方向以后，算出来的推荐角度会有一些变化，"  # 自然暴露工程师已经遇到的数值不一致。
    "所以也不确定应该相信哪个结果。请帮我给出一个可以直接采用的角度，"  # 要求最终可用答案而非研究计划。
    "并说明网格大概要怎么划，才能避免这个角度只是网格造成的假结果。"  # 要求简明网格建议但不规定研究路线。
)  # 完成工程师可见的自然问题。

MODEL_FACTS = {  # 定义模型可见且不会泄露竞赛出处的基本事实。
    "geometry": "均匀实心圆轴，直径80 mm；只计算中部100 mm观测段",  # 说明真实有限元子模型边界。
    "material": "线弹性钢，E=210 GPa，泊松比0.30",  # 提供完成工程判断所需材料数据。
    "load": "轴向拉力180 kN，扭矩2.0 kN·m，加载方向固定",  # 提供当前可见工况。
    "mark": "表面短直线长度30 mm；标线不改变结构刚度",  # 说明标线是观测对象而非实体加强件。
    "result_definition": "报告加载前后标线方向的角度变化幅值",  # 固定工程量定义避免每轮漂移。
}  # 完成模型事实包。

INTERNAL_OPERATIONS = {"orientation_compare", "refinement_compare", "extraction_compare", "angle_optimize", "analytical_check", "information", "finish", "unsupported"}  # 定义模型不可见的执行操作集合。


def _default_state() -> dict[str, Any]:  # 创建独立实验的初始运行状态。
    return {"active_mesh": {"circumferential": 32, "radial": 5, "axial": 16, "helix_angle_deg": 0.0}, "active_extraction": "nearest_node", "completed_operations": [], "finite_element_cases": []}  # 返回默认中等正交网格和最近节点提取状态。


def _state_path(output_dir: Path) -> Path:  # 定位隐藏执行器的跨轮状态文件。
    return output_dir / "state.json"  # 把状态限制在当前隔离实验目录。


def load_state(output_dir: Path) -> dict[str, Any]:  # 读取或创建跨轮实验状态。
    path = _state_path(output_dir)  # 定位当前实验状态文件。
    if path.exists():  # 检查前轮是否已经保存状态。
        return json.loads(path.read_text(encoding="utf-8"))  # 返回前轮真实执行后状态。
    state = _default_state()  # 创建初始状态。
    save_state(output_dir, state)  # 在第一次执行前保存初始状态。
    return state  # 返回初始状态。


def save_state(output_dir: Path, state: dict[str, Any]) -> None:  # 原子化保存跨轮实验状态。
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保隔离输出目录存在。
    _state_path(output_dir).write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 写入可审计状态快照。


def _config_from_dict(value: dict[str, Any]) -> MeshConfig:  # 把 JSON 网格对象转换为强类型配置。
    return MeshConfig(circumferential=int(value["circumferential"]), radial=int(value["radial"]), axial=int(value["axial"]), helix_angle_deg=float(value["helix_angle_deg"]))  # 返回通过后端校验的网格配置。


def _record_case(state: dict[str, Any], config: MeshConfig, force_n: float, torque_nmm: float) -> None:  # 在状态中记录实际请求过的有限元工况。
    record = {"mesh": {"circumferential": config.circumferential, "radial": config.radial, "axial": config.axial, "helix_angle_deg": config.helix_angle_deg}, "force_n": force_n, "torque_nmm": torque_nmm}  # 组织稳定工况记录。
    if record not in state["finite_element_cases"]:  # 避免缓存命中时重复计算工况数。
        state["finite_element_cases"].append(record)  # 保存一个新的真实有限元工况。


def build_initial_evidence(output_dir: Path) -> dict[str, Any]:  # 生成模型第一次看到的真实网格不一致证据。
    cache_root = output_dir / "solver_cache"  # 定位本实验专属真实求解缓存。
    state = load_state(output_dir)  # 读取初始执行状态。
    configurations = [MeshConfig(16, 3, 8, 0.0), MeshConfig(32, 5, 16, 0.0), MeshConfig(32, 5, 16, 20.0)]  # 定义粗正交、中等正交和等规模斜交三组初始网格。
    labels = ["较粗且网格线沿轴向和圆周方向", "中等密度且网格线沿轴向和圆周方向", "与上一组单元数量相同但轴向网格线倾斜约20度"]  # 用工程语言标记三组配置。
    calculations: list[dict[str, Any]] = []  # 初始化模型可见的初步计算列表。
    for label, config in zip(labels, configurations):  # 遍历全部初始真实工况。
        result = solve(config, cache_root)  # 执行或读取真实 CalculiX 求解。
        _record_case(state, config, VISIBLE_FORCE_N, VISIBLE_TORQUE_NMM)  # 记录当前真实有限元工况。
        scan = angle_sweep(result, "nearest_node", 0.5)  # 使用常见最近节点方法扫描标线初始方向。
        summary = compact_scan(scan)  # 压缩完整方向曲线为工程师可读摘要。
        summary["description"] = label  # 加入当前网格的自然语言说明。
        calculations.append(summary)  # 保存一组初步计算证据。
    save_state(output_dir, state)  # 保存初始真实工况记录。
    return {"user_question": USER_QUESTION, "known_facts": MODEL_FACTS, "initial_calculations": calculations, "note": "三组计算使用相同几何、材料、载荷和标线长度；目前只改变网格大小或排列方向。"}  # 返回不含解析答案和工具目录的初始证据包。


def _proposal_text(proposal: dict[str, Any]) -> str:  # 把冻结提案中的自然语言合并为确定性映射文本。
    return json.dumps(proposal, ensure_ascii=False).lower()  # 使用小写 JSON 文本匹配工程意图。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 把模型冻结的工程提案忠实映射到隐藏执行能力。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 从磁盘重新读取并验证提案摘要。
    before_hash = sha256_text(canonical_json(proposal))  # 记录隐藏映射前提案哈希。
    proposal_copy = copy.deepcopy(proposal)  # 创建只读分类副本。
    proposal_type = str(proposal_copy.get("proposal_type", ""))  # 读取通用提案类型。
    text = _proposal_text(proposal_copy)  # 组合全部自然语言字段。
    operation = "unsupported"  # 默认诚实拒绝无法保持原意的提案。
    reason = "当前后端无法在不改变提案目的的情况下执行该实验"  # 保存默认拒绝原因。
    if proposal_type == "stop":  # 识别模型主动停止并给出工程建议的决定。
        operation = "finish"  # 映射为结束实验。
        reason = "模型明确认为证据已经足够形成工程建议"  # 记录忠实映射理由。
    elif proposal_type == "request_information":  # 识别模型请求补充工程事实。
        operation = "information"  # 映射为返回已有事实或明确缺失。
        reason = "模型请求与几何、载荷、材料或测量有关的事实"  # 记录信息请求映射理由。
    elif proposal_type == "experiment":  # 识别模型提出的唯一下一步控制实验。
        if any(token in text for token in ("插值", "形函数", "精确端点", "位移梯度", "变形梯度", "最近节点", "取点方法", "提取方法")):  # 优先识别结果提取协议对照。
            operation = "extraction_compare"  # 映射为同一位移场上的提取方法对照。
            reason = "提案要求保持物理模型不变并改变标线端点或方向的提取方式"  # 记录提取对照映射理由。
        elif any(token in text for token in ("旋转网格", "网格方向", "网格排列", "斜交", "正交", "贴合", "对齐", "螺旋角", "倾斜网格")):  # 识别等规模网格方向对照。
            operation = "orientation_compare"  # 映射为相同密度下的多方向网格比较。
            reason = "提案要求在保持网格数量近似不变时改变网格主方向"  # 记录方向对照映射理由。
        elif any(token in text for token in ("加密", "细化", "网格大小", "网格密度", "收敛", "更细", "单元尺寸")):  # 识别网格分辨率对照。
            operation = "refinement_compare"  # 映射为同方向下的多级细化比较。
            reason = "提案要求保持其他条件不变并改变网格分辨率"  # 记录细化对照映射理由。
        elif any(token in text for token in ("解析", "理论", "闭式", "材料力学", "手算", "独立校核")):  # 识别独立连续体理论校核。
            operation = "analytical_check"  # 映射为不使用网格的解析参照。
            reason = "提案要求使用独立理论模型校核有限元最优方向"  # 记录理论校核映射理由。
        elif any(token in text for token in ("粒子群", "pso", "优化", "搜索", "扫描角度", "遍历角度", "最大", "最佳角度")):  # 识别连续方向黑箱搜索。
            operation = "angle_optimize"  # 映射为当前网格与提取协议上的连续搜索。
            reason = "提案要求把标线初始方向作为连续变量寻找最大方向变化"  # 记录方向优化映射理由。
    after_hash = sha256_text(canonical_json(proposal))  # 重新计算原提案哈希以检查映射未改写内容。
    if after_hash != before_hash:  # 检查隐藏映射是否污染冻结提案。
        raise RuntimeError("hidden executor mutated the frozen proposal")  # 在发现改写时立即终止。
    receipt = {"proposal_sha256": proposal_hash, "operation": operation, "mapping_reason": reason, "proposal_unchanged": True}  # 组织模型不可见的映射收据。
    (round_dir / "mapping_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存隐藏映射审计文件。
    return receipt  # 返回真实执行阶段所需的内部操作。


def _public_scan(label: str, scan: dict[str, Any]) -> dict[str, Any]:  # 为模型生成一组不泄露内部函数名的工程观测。
    summary = compact_scan(scan)  # 压缩完整角度响应曲线。
    summary["description"] = label  # 加入实际改变的网格或提取说明。
    return summary  # 返回模型可见的紧凑观测。


def execute_mapping(output_dir: Path, round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 执行隐藏映射并只回传物理变化和数值观测。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在真实执行前再次验证冻结提案。
    operation = str(mapping.get("operation", "unsupported"))  # 读取模型不可见的内部操作。
    if operation not in INTERNAL_OPERATIONS:  # 检查映射未注入未知执行路径。
        raise ValueError("invalid internal shaft-grid operation")  # 拒绝未注册操作。
    state = load_state(output_dir)  # 读取当前跨轮实验状态。
    cache_root = output_dir / "solver_cache"  # 定位真实 CalculiX 求解缓存。
    active_config = _config_from_dict(state["active_mesh"])  # 恢复当前主网格配置。
    extraction = str(state["active_extraction"])  # 读取当前标线结果提取协议。
    observations: list[dict[str, Any]] = []  # 初始化模型可见数值观测列表。
    limitations: list[str] = []  # 初始化模型可见限制说明。
    executed_change = "没有改变有限元模型"  # 初始化实际物理或数值改变说明。
    status = "completed"  # 默认当前实验执行成功。
    internal_raw: dict[str, Any] = {}  # 初始化仅供审计的完整内部结果。
    if operation == "orientation_compare":  # 执行等规模正交与斜交网格方向比较。
        configs = [MeshConfig(32, 5, 16, 0.0), MeshConfig(32, 5, 16, 15.0), MeshConfig(32, 5, 16, 30.0)]  # 定义相同单元数量的三个表面网格方向。
        for config in configs:  # 遍历全部方向配置。
            result = solve(config, cache_root)  # 执行或读取当前方向真实有限元结果。
            _record_case(state, config, VISIBLE_FORCE_N, VISIBLE_TORQUE_NMM)  # 记录真实有限元工况。
            scan = angle_sweep(result, extraction, 0.5)  # 在同一网格上扫描标线方向。
            observations.append(_public_scan(f"网格密度不变，表面轴向网格线倾斜{config.helix_angle_deg:.0f}度", scan))  # 回传当前方向的工程结果。
        state["active_mesh"] = {"circumferential": 32, "radial": 5, "axial": 16, "helix_angle_deg": 15.0}  # 把中间方向保存为后续对照主网格。
        executed_change = "保持几何、材料、载荷、单元数量和标线提取方法不变，只改变表面网格主方向"  # 说明实际控制变量。
        internal_raw = {"configs": [item["mesh"] for item in observations]}  # 保存内部方向对照摘要。
    elif operation == "refinement_compare":  # 执行同一方向下的三档网格细化比较。
        helix = float(active_config.helix_angle_deg)  # 保持当前网格方向不变。
        configs = [MeshConfig(16, 3, 8, helix), MeshConfig(32, 5, 16, helix), MeshConfig(64, 8, 32, helix)]  # 定义粗中细三档真实实体网格。
        for config in configs:  # 遍历全部分辨率配置。
            result = solve(config, cache_root)  # 执行或读取当前分辨率真实有限元结果。
            _record_case(state, config, VISIBLE_FORCE_N, VISIBLE_TORQUE_NMM)  # 记录真实有限元工况。
            scan = angle_sweep(result, extraction, 0.5)  # 在当前网格上扫描标线方向。
            observations.append(_public_scan(f"网格方向保持{helix:.0f}度，圆周/径向/轴向划分为{config.circumferential}/{config.radial}/{config.axial}", scan))  # 回传当前分辨率工程结果。
        state["active_mesh"] = {"circumferential": 64, "radial": 8, "axial": 32, "helix_angle_deg": helix}  # 把最细网格保存为后续主网格。
        executed_change = "保持几何、材料、载荷、网格方向和提取方法不变，逐级细化圆周、径向和轴向网格"  # 说明实际控制变量。
        internal_raw = {"configs": [item["mesh"] for item in observations]}  # 保存内部细化对照摘要。
    elif operation == "extraction_compare":  # 在完全相同位移场上比较两种标线端点提取方式。
        result = solve(active_config, cache_root)  # 读取当前主网格真实有限元位移场。
        _record_case(state, active_config, VISIBLE_FORCE_N, VISIBLE_TORQUE_NMM)  # 记录当前真实有限元工况。
        nearest = angle_sweep(result, "nearest_node", 0.5)  # 使用最近节点端点方法扫描方向。
        interpolated = angle_sweep(result, "surface_interpolation", 0.5)  # 使用精确端点表面插值方法扫描方向。
        observations.append(_public_scan("把标线两端吸附到最近表面节点", nearest))  # 回传最近节点工程结果。
        observations.append(_public_scan("保持标线真实位置，在所在表面单元内插值两端位移", interpolated))  # 回传精确端点插值结果。
        state["active_extraction"] = "surface_interpolation"  # 把方向偏置更小的协议保存为后续主提取方法。
        executed_change = "有限元网格和位移场完全不变，只改变标线两端位移的读取方法"  # 说明该对照没有新增结构求解。
        limitations.append("该对照能够识别节点吸附误差，但仍需要通过旋转网格或继续细化检查插值结果是否稳定")  # 说明当前证据边界。
        internal_raw = {"nearest_node": nearest, "surface_interpolation": interpolated}  # 保存完整内部响应曲线。
    elif operation == "angle_optimize":  # 在当前真实位移场上执行连续黑箱方向搜索。
        result = solve(active_config, cache_root)  # 读取当前主网格真实有限元位移场。
        _record_case(state, active_config, VISIBLE_FORCE_N, VISIBLE_TORQUE_NMM)  # 记录当前真实有限元工况。
        search = particle_search(result, extraction)  # 使用一维粒子群搜索最大标线转角。
        observations.append({"mesh": search["mesh"], "extraction": extraction, "recommended_angle_deg": round(float(search["best_beta_deg"]), 4), "predicted_angle_change_deg": round(float(search["best_delta_beta_deg"]), 8), "direction_evaluations": int(search["evaluations"])})  # 回传优化得到的工程结果。
        executed_change = "保持有限元位移场不变，把标线初始方向作为连续变量搜索角度变化最大值"  # 说明方向搜索只增加廉价后处理。
        limitations.append("该结果只有在网格方向、网格密度和标线提取方法已经得到独立验证时才可直接采用")  # 防止把数值伪峰当作物理最优。
        internal_raw = search  # 保存完整粒子群轨迹供审计。
    elif operation == "analytical_check":  # 执行独立于有限元网格的连续体理论校核。
        reference = analytical_optimum(VISIBLE_FORCE_N, VISIBLE_TORQUE_NMM)  # 计算当前工程工况解析参照。
        observations.append({"assumptions": "均匀实心圆轴、小变形、线弹性、远离端部", "recommended_angle_deg": round(reference["beta_deg"], 4), "predicted_angle_change_deg": round(reference["delta_beta_deg"], 8)})  # 回传独立理论结果和适用条件。
        executed_change = "不改变有限元结果，增加一个独立连续体理论校核"  # 说明理论校核与网格无关。
        limitations.append("理论结果不能替代对实际网格方向、离散分辨率和结果提取方法的验证")  # 说明解析校核的工程边界。
        internal_raw = reference  # 保存高分辨率解析真值。
    elif operation == "information":  # 回答模型请求的已知工程事实。
        question = str(proposal.get("information_request", {}).get("question", ""))  # 读取冻结的信息请求文本。
        observations.append({"requested_information": question, "available_facts": MODEL_FACTS})  # 返回全部已知事实并避免猜测缺失数据。
        executed_change = "补充现有几何、材料、载荷和标线事实"  # 说明没有执行新求解。
        status = "information_provided"  # 标记本轮只补充事实。
    elif operation == "finish":  # 处理模型主动形成最终工程建议。
        executed_change = "模型选择停止新增计算并形成最终建议"  # 记录自主停止行为。
        status = "finished"  # 通知主循环结束。
    else:  # 处理当前后端无法忠实执行的实验。
        executed_change = "未改变模型，因为该提案无法在保持原目的的情况下执行"  # 公开说明诚实拒绝。
        limitations.append(str(mapping.get("mapping_reason", "实验不受支持")))  # 回传无法执行的工程限制。
        status = "unsupported"  # 标记本轮执行未完成。
    if operation not in {"unsupported", "information", "finish"}:  # 检查是否完成了一个真实数值或理论操作。
        state["completed_operations"].append(operation)  # 在隐藏状态中记录操作序列。
    save_state(output_dir, state)  # 保存本轮实际执行后的状态。
    audit = {"proposal_sha256": proposal_hash, "operation": operation, "executed_change": executed_change, "observations": observations, "limitations": limitations, "internal_raw": internal_raw, "state_after": state}  # 组织完整模型不可见执行审计。
    (round_dir / "execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存完整内部数值和状态。
    feedback = {"status": status, "executed_change": executed_change, "observations": observations, "limitations": limitations}  # 组织下一轮模型可见的脱敏反馈。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存公开物理反馈。
    return feedback  # 返回下一轮模型允许看到的证据。


def _extract_recommended_angle(public_history: list[dict[str, Any]]) -> float | None:  # 从模型公开答复中提取最后一个明确推荐角度。
    candidates: list[float] = []  # 初始化候选角度列表。
    for item in public_history:  # 按时间遍历全部模型公开轮次。
        proposal = item.get("proposal", {})  # 读取当前轮冻结提案。
        answer = str(proposal.get("provisional_answer", ""))  # 读取模型当前暂定或最终答复。
        preferred = re.findall(r"(?:推荐|建议|采用|取|约为|角度)[^0-9]{0,12}(\d+(?:\.\d+)?)\s*(?:°|度)", answer)  # 优先提取与推荐语义相邻的角度。
        fallback = re.findall(r"(\d+(?:\.\d+)?)\s*(?:°|度)", answer)  # 兼容模型只给数值角度的情况。
        selected = preferred if preferred else fallback  # 优先使用推荐语义候选。
        if selected:  # 检查当前答复是否包含角度。
            candidates.append(float(selected[-1]))  # 保存当前答复最后一个候选角度。
    return candidates[-1] if candidates else None  # 返回最后明确推荐角度或空值。


def score_experiment(output_dir: Path, public_history: list[dict[str, Any]]) -> dict[str, Any]:  # 使用隐藏解析真值和盲测工况评价完整建模能力。
    state = load_state(output_dir)  # 读取模型最终选择后的网格和提取状态。
    text = json.dumps(public_history, ensure_ascii=False).lower()  # 组合全部公开提案和反馈用于能力标记。
    recommendation = _extract_recommended_angle(public_history)  # 提取模型最终明确推荐方向。
    visible_truth = analytical_optimum(VISIBLE_FORCE_N, VISIBLE_TORQUE_NMM)  # 计算当前可见工况隐藏解析真值。
    visible_error = None if recommendation is None else abs(recommendation - visible_truth["beta_deg"])  # 计算最终建议与隐藏真值的角度误差。
    active_config = _config_from_dict(state["active_mesh"])  # 恢复模型最终主网格。
    extraction = str(state["active_extraction"])  # 恢复模型最终主提取协议。
    cache_root = output_dir / "solver_cache"  # 定位真实求解缓存用于盲测。
    blind_cases: list[dict[str, Any]] = []  # 初始化不同拉扭比例的隐藏通用性测试。
    for force_n in (120000.0, 240000.0):  # 使用两个模型未见过的轴向拉力比例。
        result = solve(active_config, cache_root, force_n=force_n, torque_nmm=VISIBLE_TORQUE_NMM)  # 用最终网格执行真实盲测有限元求解。
        predicted = particle_search(result, extraction, seed=20260731 + int(force_n))  # 用最终提取协议搜索盲测最佳方向。
        truth = analytical_optimum(force_n, VISIBLE_TORQUE_NMM)  # 计算该盲测工况解析真值。
        blind_cases.append({"force_n": force_n, "torque_nmm": VISIBLE_TORQUE_NMM, "predicted_beta_deg": predicted["best_beta_deg"], "truth_beta_deg": truth["beta_deg"], "absolute_error_deg": abs(predicted["best_beta_deg"] - truth["beta_deg"])})  # 保存一组盲测误差。
    capability_flags = {"identified_direction_variable": any(token in text for token in ("初始方向", "标线方向", "角度变量", "方向作为")), "used_continuous_search": any(token in text for token in ("粒子群", "pso", "连续", "搜索", "扫描")), "tested_mesh_orientation": any(token in text for token in ("网格方向", "旋转网格", "斜交", "正交", "贴合", "对齐")), "tested_refinement": any(token in text for token in ("细化", "加密", "收敛", "网格密度")), "questioned_extraction_protocol": any(token in text for token in ("插值", "形函数", "最近节点", "位移梯度", "提取方法")), "requested_independent_check": any(token in text for token in ("解析", "理论", "手算", "独立校核"))}  # 记录模型是否自行发现六个关键建模点。
    angle_pass = visible_error is not None and visible_error <= 1.0  # 要求最终工程角度在一度内命中隐藏真值。
    blind_pass = all(item["absolute_error_deg"] <= 1.0 for item in blind_cases)  # 要求最终数值方法在两组盲测中保持一度精度。
    discovery_score = sum(1 for value in capability_flags.values() if value)  # 统计模型主动发现的关键能力数量。
    return {"visible_case": {"model_recommendation_deg": recommendation, "truth_deg": visible_truth["beta_deg"], "absolute_error_deg": visible_error, "pass": angle_pass}, "final_numerical_method": {"mesh": state["active_mesh"], "extraction": extraction}, "blind_generalization": blind_cases, "blind_pass": blind_pass, "capability_flags": capability_flags, "discovery_score_out_of_6": discovery_score, "overall_pass": bool(angle_pass and blind_pass and capability_flags["tested_mesh_orientation"] and capability_flags["tested_refinement"] and capability_flags["questioned_extraction_protocol"])}  # 返回完整隐藏评分报告。
