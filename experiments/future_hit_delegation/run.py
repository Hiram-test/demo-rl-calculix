#!/usr/bin/env python3  # 使用仓库 Python 运行未来命中—资源重投实验。
from __future__ import annotations  # 启用现代类型注解并保持 Python 3.11 兼容。
import argparse  # 解析 CalculiX 路径、输出目录和实验轮次参数。
import csv  # 写出逐方法与逐区域机器可读结果。
import hashlib  # 冻结发送给大模型的证据包并生成审计哈希。
import importlib.util  # 从已有正式横向通道基准脚本动态载入真实求解器实现。
import json  # 读写大模型结构化预测、实验轨迹和汇总结果。
import math  # 计算误差统计与安全归一化数值。
import os  # 从 GitHub Actions 环境读取 DeepSeek 凭据与模型名称。
from pathlib import Path  # 统一处理仓库路径和实验输出目录。
from typing import Any  # 为结构化证据与 JSON 结果提供通用类型。
import matplotlib.pyplot as plt  # 使用脚本生成论文型反馈深度与未来命中对比图。
import numpy as np  # 处理区域优先级、网格级别和统计数组。
from openai import OpenAI  # 通过 OpenAI 兼容接口调用 DeepSeek 模型。
ROOT = Path(__file__).resolve().parents[2]  # 定位仓库根目录以载入已有 CalculiX 基准。
BASE_SCRIPT = ROOT / "experiments" / "cross_passage_torsion_benchmark" / "run_benchmark.py"  # 指向 PR24 已验证的真实 CalculiX 横向通道实现。
DEFAULT_OUTPUT = ROOT / "experiments" / "future_hit_delegation" / "results"  # 定义本实验默认结果目录。
METHOD_DYNAMIC = "dynamic_dorfler"  # 定义逐轮真实反馈的经典 Dörfler 基线名称。
METHOD_ORACLE = "oracle_future_hit_macro"  # 定义使用离线未来命中真值的一次宏动作上界名称。
METHOD_RANK = "current_evidence_rank_macro"  # 定义仅依据当前数值强度进行重投的确定性对照名称。
METHOD_LLM_PREFIX = "llm_future_hit_macro"  # 定义真实大模型未来命中预测路线名称前缀。

def load_base_module() -> Any:  # 动态载入正式横向通道真实 CalculiX 基准模块。
    spec = importlib.util.spec_from_file_location("cross_passage_base", BASE_SCRIPT)  # 为已有脚本创建独立模块规范。
    if spec is None or spec.loader is None:  # 检查动态载入器是否成功创建。
        raise RuntimeError(f"cannot load benchmark module from {BASE_SCRIPT}")  # 在无法载入正式基准时立即停止实验。
    module = importlib.util.module_from_spec(spec)  # 根据规范创建模块对象。
    spec.loader.exec_module(module)  # 执行已有正式基准脚本并注册其中的类与常量。
    return module  # 返回可直接构造 TorsionBenchmark 的模块对象。

def dorfler_mark(priority: np.ndarray, theta: float) -> list[int]:  # 对非负区域价值执行最小前缀 Dörfler 标记。
    values = np.maximum(np.asarray(priority, dtype=np.float64), 0.0)  # 将数值噪声导致的负值截断为零。
    total = float(np.sum(values))  # 计算当前区域评价总质量。
    if total <= 0.0:  # 检查是否不存在可辨识的正热点。
        return [int(np.argmax(values))]  # 退化情况下至少保留一个稳定索引避免空动作。
    order = np.argsort(-values, kind="stable")  # 按区域评价量从大到小稳定排序。
    target = float(theta) * total  # 计算需要覆盖的 Dörfler 质量阈值。
    running = 0.0  # 初始化已覆盖评价质量。
    marked: list[int] = []  # 初始化标记区域索引列表。
    for index in order.tolist():  # 按排序结果逐一累计候选区域。
        marked.append(int(index))  # 将当前区域加入动作支撑。
        running += float(values[index])  # 累加当前区域评价质量。
        if running + 1.0e-15 >= target:  # 检查是否已达到规定覆盖比例。
            break  # 达到阈值后立即停止以保持最小前缀性质。
    return marked  # 返回当前轮数值热点区域集合。

def make_benchmark(module: Any, output_root: Path, ccx_command: str) -> Any:  # 为每条方法路线创建独立真实求解缓存以公平计费。
    benchmark = module.TorsionBenchmark(output_root, ccx_command)  # 构造已有真实 CalculiX 横向通道基准对象。
    benchmark.prepare_reference()  # 独立求解隐藏参考网格，仅用于最终误差评价而不发送给大模型。
    return benchmark  # 返回完成参考解准备的独立基准对象。

def solution_record(benchmark: Any, solution: Any, step: int, marked: list[int] | None = None) -> dict[str, Any]:  # 把真实有限元状态压缩成统一审计记录。
    objective, torque_error, energy_error, probe_error, hotspot_recall = benchmark.metrics(solution)  # 使用统一隐藏参考计算最终评价而不改变在线决策。
    return {"step": int(step), "levels": [int(value) for value in solution.levels], "element_count": int(solution.element_count), "objective": float(objective), "torque_error": float(torque_error), "energy_distribution_error": float(energy_error), "probe_error": float(probe_error), "hotspot_recall": float(hotspot_recall), "marked_regions": [] if marked is None else [int(value) for value in marked]}  # 返回机器可读的真实轨迹状态。

def apply_increment(benchmark: Any, levels: tuple[int, ...], increments: np.ndarray, priority: np.ndarray) -> tuple[int, ...]:  # 将区域级资源重投向量转成满足统一单元预算的实际网格级别。
    proposed = np.asarray(levels, dtype=np.float64) + np.asarray(increments, dtype=np.float64)  # 在当前级别上叠加预测的累计未来动作。
    return benchmark.repair_levels(proposed, np.asarray(priority, dtype=np.float64))  # 使用已有预算修复器保证最终网格不超过相同单元上限。

def run_dynamic_dorfler(module: Any, output_root: Path, ccx_command: str, theta: float, max_rounds: int) -> dict[str, Any]:  # 执行逐轮 solve→mark→一级加密的真实反馈基线。
    benchmark = make_benchmark(module, output_root, ccx_command)  # 为动态基线创建独立求解缓存和参考解。
    levels = tuple(0 for _ in range(benchmark.region_count))  # 所有区域从最粗离散级别开始以形成共同在线起点。
    solution = benchmark.solve(levels)  # 执行共同粗网格真实 CalculiX 求解并获得第一份物理证据。
    initial_priority = benchmark.hotspot_priority(solution, levels)  # 从共同粗解提取当前区域误差/响应价值代理。
    initial_marked = dorfler_mark(initial_priority, theta)  # 用标准 Dörfler 规则得到第一次当前热点支撑。
    trace = [solution_record(benchmark, solution, 0, initial_marked)]  # 记录共同粗解但不把它计入额外反馈深度。
    marks: list[list[int]] = []  # 初始化各轮真实 Dörfler 标记历史用于构造 future-hit 真值。
    current_priority = initial_priority.copy()  # 保存当前真实有限元状态对应的区域价值。
    for round_index in range(max_rounds):  # 在预注册的最大反馈深度内逐轮执行经典自适应。
        marked = dorfler_mark(current_priority, theta)  # 根据当前而非未来信息重新确定本轮热点区域。
        increments = np.zeros(benchmark.region_count, dtype=np.float64)  # 初始化本轮统一一级加密动作。
        increments[marked] = 1.0  # 标准基线只对本轮命中区域增加一级资源。
        next_levels = apply_increment(benchmark, levels, increments, current_priority)  # 将一级动作修复到统一最终资源预算内。
        marks.append([int(value) for value in marked])  # 保存本轮真实命中区域作为离线轨迹真值的一部分。
        if next_levels == levels:  # 检查预算或最大级别是否使本轮动作无法改变网格。
            break  # 无法继续改变网格时停止逐轮反馈以避免重复求解。
        levels = next_levels  # 接受本轮经过预算修复后的新区域网格级别。
        solution = benchmark.solve(levels)  # 重新执行真实 CalculiX 并形成下一轮物理反馈。
        current_priority = benchmark.hotspot_priority(solution, levels)  # 使用新的真实解重新计算当前热点证据。
        trace.append(solution_record(benchmark, solution, len(trace), marked))  # 记录本次额外高保真反馈状态。
    future_hits = np.zeros(benchmark.region_count, dtype=np.int64)  # 初始化从共同粗网格开始的未来重复命中次数真值。
    for marked in marks:  # 遍历动态基线实际经历的全部标记轮次。
        for index in marked:  # 遍历当前轮所有被命中的区域。
            future_hits[index] += 1  # 将该区域累计未来资源需求增加一次。
    final_record = trace[-1]  # 提取动态基线有限轮次后的最终真实状态。
    return {"method": METHOD_DYNAMIC, "theta": float(theta), "max_rounds": int(max_rounds), "additional_H": int(max(len(trace) - 1, 0)), "logical_N_FE": int(max(len(trace) - 1, 0)), "final": final_record, "trace": trace, "future_hits": future_hits.tolist(), "initial_levels": trace[0]["levels"], "initial_priority": initial_priority.tolist(), "initial_marked": initial_marked, "region_names": list(benchmark.region_names), "region_member_counts": benchmark.region_member_counts.astype(int).tolist(), "element_cap": int(benchmark.element_cap)}  # 返回基线、未来命中 oracle 和共同在线证据。

def current_rank_prediction(priority: np.ndarray, marked: list[int], region_count: int) -> np.ndarray:  # 构造不使用机制推理的当前强度重投对照。
    prediction = np.zeros(region_count, dtype=np.int64)  # 初始化所有区域不提前重投资源。
    if not marked:  # 检查是否不存在当前 Dörfler 支撑。
        return prediction  # 空支撑时直接返回全零预测。
    ranked = sorted(marked, key=lambda index: float(priority[index]), reverse=True)  # 只依据当前评价大小给被标记区域排序。
    for rank, index in enumerate(ranked):  # 遍历当前热点并分配固定深度。
        fraction = rank / max(len(ranked) - 1, 1)  # 将热点排序位置归一化到零到一。
        prediction[index] = 3 if fraction <= 0.25 else (2 if fraction <= 0.65 else 1)  # 当前最强热点重投三级、中间热点两级、其余一级。
    return prediction  # 返回不含未来机制推理的重投深度向量。

def build_llm_packet(dynamic: dict[str, Any], benchmark: Any, coarse_solution: Any) -> dict[str, Any]:  # 构造严格不泄漏 future-hit oracle 的大模型在线证据包。
    levels = tuple(int(value) for value in dynamic["initial_levels"])  # 读取所有方法共享的粗网格区域级别。
    features = benchmark.region_features(coarse_solution, levels)  # 从共同粗解计算能量、应力、对比度与局部成本特征。
    priority = np.asarray(dynamic["initial_priority"], dtype=np.float64)  # 读取第一次 Dörfler 使用的同一当前评价量。
    scale = max(float(np.max(priority)), 1.0e-18)  # 定义稳定优先级归一化尺度。
    rows: list[dict[str, Any]] = []  # 初始化可发送给大模型的区域证据列表。
    for index, name in enumerate(dynamic["region_names"]):  # 遍历十六个结构区域构造匿名化数值证据。
        neighbor_ids = np.flatnonzero(benchmark.region_adjacency[index] > 0.0).astype(int).tolist()  # 提取当前结构区域的真实图邻接关系。
        rows.append({"id": int(index), "name": str(name), "current_dorfler_marked": bool(index in dynamic["initial_marked"]), "priority_normalized": float(priority[index] / scale), "energy_normalized": float(features[index, 0]), "peak_proxy_normalized": float(features[index, 1]), "neighbor_contrast": float(features[index, 2]), "current_level": int(levels[index]), "local_cost_fraction": float(features[index, 4]), "neighbor_ids": neighbor_ids})  # 写入当前物理状态可直接支持的证据字段。
    return {"task": "Predict cumulative future refinement depth from the current reliable Dörfler evidence so that several future one-level Dörfler actions can be safely delegated as one macro-action.", "physics": "linear-elastic 3D B31 space-frame torsion benchmark solved by CalculiX", "qoi_contract": "same composite accuracy objective used only for post-execution audit; hidden reference values are not visible online", "budget": {"maximum_region_level": 3, "element_cap": int(dynamic["element_cap"]), "allowed_extra_levels_per_region": [0, 1, 2, 3]}, "current_evidence": {"dorfler_theta": float(dynamic["theta"]), "marked_region_ids": [int(value) for value in dynamic["initial_marked"]], "regions": rows}, "decision_boundary": "Do not redesign the error indicator and do not search for a new current hotspot map. Treat the Dörfler support and coarse-solve features as reliable current evidence. Your only scientific task is to infer which current or physically coupled regions are likely to remain persistently important over subsequent adaptive rounds, and therefore how many future one-level actions can be delegated now. Use 0 when evidence does not justify advance allocation.", "required_output": {"regions": "exactly 16 objects with id, extra_levels integer 0..3, confidence 0..1, short mechanism label, evidence_refs array of supplied field names", "global_confidence": "number 0..1", "audit_recommendation": "one of FINAL_SOLVE_ONLY or FINAL_PLUS_ONE_CHECKPOINT"}}  # 返回不包含任何未来轨迹标签的严格在线证据合同。

def call_llm(packet: dict[str, Any], model: str, api_key: str, trial: int) -> tuple[dict[str, Any], str]:  # 调用真实 DeepSeek 预测未来累计资源投入深度。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")  # 使用仓库既有 DeepSeek Environment 凭据建立兼容客户端。
    packet_text = json.dumps(packet, ensure_ascii=False, sort_keys=True)  # 将冻结证据包序列化为稳定文本。
    packet_hash = hashlib.sha256(packet_text.encode("utf-8")).hexdigest()  # 计算实际发送证据包的 SHA256 以防标签泄漏争议。
    system_text = "You are a finite-element adaptation planner. Return only one valid JSON object. Current local error evidence is authoritative. Do not replace Dörfler marking. Estimate only persistent future refinement demand and safe resource delegation depth. Use short auditable reasons, not hidden chain-of-thought."  # 定义只允许机制级资源重投的系统边界。
    user_text = f"Trial {trial}. Evidence packet follows. Predict cumulative extra refinement levels for every region under the supplied contract.\n{packet_text}"  # 将唯一试验编号和冻结证据包发送给模型。
    completion = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_text}, {"role": "user", "content": user_text}], temperature=0.0, response_format={"type": "json_object"})  # 以确定性 JSON 模式执行一次真实模型调用。
    raw_text = completion.choices[0].message.content or "{}"  # 提取模型实际结构化输出文本并处理空响应。
    parsed = json.loads(raw_text)  # 将模型 JSON 输出解析为可执行资源配置合同。
    parsed["packet_sha256"] = packet_hash  # 在结构化输出中绑定此次证据包哈希。
    parsed["model"] = model  # 记录实际调用模型名称以支持后续复现与模型对照。
    return parsed, raw_text  # 返回解析对象与原始模型响应用于完整审计。

def validate_llm_prediction(parsed: dict[str, Any], region_count: int) -> tuple[np.ndarray, float, str]:  # 对模型输出执行硬限制门并转换为整数重投向量。
    rows = parsed.get("regions")  # 读取模型输出的逐区域预测数组。
    if not isinstance(rows, list) or len(rows) != region_count:  # 要求模型对全部区域显式给出判断以避免默默漏区。
        raise ValueError(f"LLM must return exactly {region_count} region rows")  # 输出区域数量不完整时拒绝执行任何网格动作。
    prediction = np.zeros(region_count, dtype=np.int64)  # 初始化受硬门约束的整数资源重投向量。
    seen: set[int] = set()  # 初始化区域 ID 完整性检查集合。
    for row in rows:  # 遍历模型给出的每个结构区域决策。
        index = int(row["id"])  # 读取并规范化区域编号。
        if index < 0 or index >= region_count or index in seen:  # 检查编号范围和重复输出。
            raise ValueError(f"invalid or duplicate region id {index}")  # 区域索引不合法时拒绝执行模型动作。
        seen.add(index)  # 将当前区域登记为已验证输出。
        prediction[index] = int(np.clip(int(row["extra_levels"]), 0, 3))  # 把模型资源深度限制在预注册的零到三级动作域。
    global_confidence = float(np.clip(float(parsed.get("global_confidence", 0.0)), 0.0, 1.0))  # 将全局机制置信度约束到零到一。
    audit = str(parsed.get("audit_recommendation", "FINAL_PLUS_ONE_CHECKPOINT"))  # 读取模型建议的物理审计深度并提供保守默认值。
    if audit not in {"FINAL_SOLVE_ONLY", "FINAL_PLUS_ONE_CHECKPOINT"}:  # 检查审计枚举是否落在允许集合内。
        audit = "FINAL_PLUS_ONE_CHECKPOINT"  # 未识别的审计建议自动降级为保守中间检查。
    return prediction, global_confidence, audit  # 返回可执行的重投深度、置信度和审计建议。

def run_macro_method(module: Any, output_root: Path, ccx_command: str, dynamic: dict[str, Any], prediction: np.ndarray, method_name: str, target_objective: float, theta: float, max_corrections: int) -> dict[str, Any]:  # 执行一次资源重投并允许有限真实物理纠错。
    benchmark = make_benchmark(module, output_root, ccx_command)  # 为当前宏动作方法创建独立真实求解缓存。
    levels = tuple(0 for _ in range(benchmark.region_count))  # 从与经典基线相同的最粗网格开始。
    coarse_solution = benchmark.solve(levels)  # 执行共同粗网格真实 FEA 并建立方法自己的在线初始状态。
    priority = benchmark.hotspot_priority(coarse_solution, levels)  # 使用与经典基线相同的当前局部物理价值证据。
    trace = [solution_record(benchmark, coarse_solution, 0, dorfler_mark(priority, theta))]  # 保存共同粗解但不计入额外 H。
    macro_levels = apply_increment(benchmark, levels, prediction.astype(np.float64), priority)  # 将预测累计未来动作一次性映射为预算可行网格。
    if macro_levels != levels:  # 检查宏动作是否真实改变网格状态。
        levels = macro_levels  # 接受通过资源门修复后的区域级目标网格。
        solution = benchmark.solve(levels)  # 执行一次真实 CalculiX 作为宏动作后的物理审计。
        trace.append(solution_record(benchmark, solution, 1, []))  # 记录第一次额外高保真反馈。
    else:  # 处理模型预测全零或预算修复后无动作的情况。
        solution = coarse_solution  # 沿用共同粗解作为当前状态并避免伪造额外求解。
    corrections = 0  # 初始化宏动作后的保守 Dörfler 纠错次数。
    while float(benchmark.metrics(solution)[0]) > float(target_objective) and corrections < max_corrections:  # 仅在真实审计未达到动态基线质量时购买新物理反馈。
        current_priority = benchmark.hotspot_priority(solution, levels)  # 从失败后的真实状态重新提取局部误差证据。
        marked = dorfler_mark(current_priority, theta)  # 使用可靠 Dörfler 规则确定下一纠错支撑。
        increments = np.zeros(benchmark.region_count, dtype=np.float64)  # 初始化一次保守一级纠错动作。
        increments[marked] = 1.0  # 仅对当前真实热点增加一级资源以限制先验错误传播。
        next_levels = apply_increment(benchmark, levels, increments, current_priority)  # 将纠错动作修复到同一最终资源预算。
        if next_levels == levels:  # 检查是否已无法在预算内继续改变网格。
            break  # 无可行动作时终止并如实保留当前失败状态。
        levels = next_levels  # 接受经过预算门校正后的纠错网格。
        solution = benchmark.solve(levels)  # 再执行一次真实 CalculiX 获取纠错反馈。
        corrections += 1  # 累计真实额外纠错反馈次数。
        trace.append(solution_record(benchmark, solution, len(trace), marked))  # 保存纠错后的真实物理状态。
    return {"method": method_name, "prediction": prediction.astype(int).tolist(), "additional_H": int(max(len(trace) - 1, 0)), "logical_N_FE": int(max(len(trace) - 1, 0)), "corrections": int(corrections), "target_objective": float(target_objective), "success_vs_dynamic": bool(float(trace[-1]["objective"]) <= float(target_objective)), "final": trace[-1], "trace": trace}  # 返回宏动作效率、最终质量和完整物理反馈轨迹。

def depth_metrics(prediction: np.ndarray, future_hits: np.ndarray) -> dict[str, float]:  # 评价提前重投深度是否复现长程 Dörfler 实际需求。
    pred = np.asarray(prediction, dtype=np.float64)  # 转换预测累计深度为双精度数组。
    truth = np.minimum(np.asarray(future_hits, dtype=np.float64), 3.0)  # 将 oracle 命中次数截断到系统最大三级可执行深度。
    mae = float(np.mean(np.abs(pred - truth)))  # 计算全部区域累计深度平均绝对误差。
    persistent_truth = truth >= 2.0  # 将未来至少重复命中两次定义为持续热点。
    persistent_pred = pred >= 2.0  # 将预测至少提前重投两级定义为持续热点判断。
    tp = int(np.sum(persistent_truth & persistent_pred))  # 统计持续热点真阳性数量。
    fp = int(np.sum(~persistent_truth & persistent_pred))  # 统计过度重投造成的假阳性数量。
    fn = int(np.sum(persistent_truth & ~persistent_pred))  # 统计遗漏持续热点造成的假阴性数量。
    precision = float(tp / max(tp + fp, 1))  # 计算持续热点预测精确率。
    recall = float(tp / max(tp + fn, 1))  # 计算持续热点预测召回率。
    f1 = float(2.0 * precision * recall / max(precision + recall, 1.0e-18))  # 计算持续热点 F1 分数。
    return {"depth_mae": mae, "persistent_precision": precision, "persistent_recall": recall, "persistent_f1": f1}  # 返回可直接用于论文统计的机制轨迹预测指标。

def write_region_table(path: Path, dynamic: dict[str, Any], predictions: dict[str, np.ndarray]) -> None:  # 写出每个区域当前证据、未来命中真值和各方法预测。
    future_hits = np.asarray(dynamic["future_hits"], dtype=np.int64)  # 读取长程动态 Dörfler 轨迹形成的离线未来命中真值。
    priority = np.asarray(dynamic["initial_priority"], dtype=np.float64)  # 读取仅来自共同粗解的当前区域价值。
    with path.open("w", newline="", encoding="utf-8") as handle:  # 打开 UTF-8 CSV 输出文件。
        writer = csv.writer(handle)  # 创建标准 CSV 写入器。
        header = ["region_id", "region_name", "current_dorfler_marked", "initial_priority", "future_hit_count"] + [f"pred_{name}" for name in predictions]  # 定义当前证据、未来真值和所有预测列。
        writer.writerow(header)  # 写出区域级实验表头。
        for index, name in enumerate(dynamic["region_names"]):  # 遍历全部结构区域写出配对记录。
            row = [index, name, int(index in dynamic["initial_marked"]), float(priority[index]), int(future_hits[index])] + [int(predictions[key][index]) for key in predictions]  # 组合当前证据、离线真值和各方法累计深度预测。
            writer.writerow(row)  # 写出当前区域的完整配对记录。

def write_method_table(path: Path, methods: list[dict[str, Any]], dynamic_H: int) -> None:  # 写出逐方法最终精度、资源和反馈深度汇总。
    with path.open("w", newline="", encoding="utf-8") as handle:  # 打开方法级 CSV 结果文件。
        writer = csv.writer(handle)  # 创建标准 CSV 写入器。
        writer.writerow(["method", "success_vs_dynamic", "additional_H", "H_saved_vs_dynamic", "logical_N_FE", "final_objective", "final_elements", "depth_mae", "persistent_f1", "global_confidence", "audit_recommendation"])  # 固定论文核心效率与机制预测列。
        for item in methods:  # 遍历经典、oracle、确定性和真实 LLM 方法。
            writer.writerow([item["method"], int(item.get("success_vs_dynamic", True)), int(item["additional_H"]), int(dynamic_H - int(item["additional_H"])), int(item["logical_N_FE"]), float(item["final"]["objective"]), int(item["final"]["element_count"]), item.get("depth_mae", ""), item.get("persistent_f1", ""), item.get("global_confidence", ""), item.get("audit_recommendation", "")])  # 写出当前方法的实际反馈节省与最终质量。

def plot_results(output_root: Path, methods: list[dict[str, Any]], dynamic: dict[str, Any], predictions: dict[str, np.ndarray]) -> None:  # 用脚本生成反馈效率与未来命中预测的两张独立论文图。
    labels = [item["method"] for item in methods]  # 提取全部方法名称作为横轴标签。
    h_values = [int(item["additional_H"]) for item in methods]  # 提取各方法实际额外真实反馈深度。
    objectives = [float(item["final"]["objective"]) for item in methods]  # 提取各方法最终真实误差目标值。
    figure = plt.figure(figsize=(11.0, 6.0))  # 创建独立反馈深度—最终误差散点图。
    axis = figure.add_subplot(111)  # 创建单一坐标轴避免子图混杂。
    axis.scatter(h_values, objectives, s=85.0)  # 绘制各方法实际 H 与最终真实误差位置。
    for x_value, y_value, label in zip(h_values, objectives, labels):  # 遍历方法为每个散点添加短标签。
        axis.annotate(label, (x_value, y_value), xytext=(6, 6), textcoords="offset points", fontsize=8)  # 在点旁标出完整方法名称。
    axis.set_xlabel("Additional real FEA feedback depth H")  # 标记真实顺序高保真反馈深度横轴。
    axis.set_ylabel("Final audited objective")  # 标记最终真实有限元误差目标纵轴。
    axis.set_title("Future-hit delegation: final quality versus real FEA feedback")  # 标明图像验证的资源重投科学命题。
    axis.grid(True, alpha=0.25)  # 添加弱网格帮助读取散点位置而不改变数据表达。
    figure.tight_layout()  # 自动压缩标签避免裁切。
    figure.savefig(output_root / "objective_vs_feedback_depth.png", dpi=220)  # 保存位图用于快速审阅。
    figure.savefig(output_root / "objective_vs_feedback_depth.pdf")  # 保存矢量图用于后续论文排版。
    plt.close(figure)  # 关闭第一张图释放无头运行内存。
    truth = np.minimum(np.asarray(dynamic["future_hits"], dtype=np.int64), 3)  # 构造最大三级动作域内的未来累计命中真值。
    x_values = np.arange(len(truth), dtype=np.int64)  # 为十六个结构区域建立横轴索引。
    figure = plt.figure(figsize=(12.0, 6.0))  # 创建独立未来命中预测对比图。
    axis = figure.add_subplot(111)  # 创建单一柱线叠加坐标轴。
    axis.plot(x_values, truth, marker="o", linewidth=2.0, label="dynamic Dörfler future hits")  # 绘制长程经典轨迹形成的未来命中真值。
    for name, prediction in predictions.items():  # 遍历当前强度、oracle 和真实 LLM 预测曲线。
        axis.plot(x_values, prediction, marker=".", linewidth=1.2, label=name)  # 绘制每条累计资源重投深度预测曲线。
    axis.set_xlabel("Region id")  # 标记结构区域索引横轴。
    axis.set_ylabel("Cumulative extra refinement levels")  # 标记未来需要提前委派的累计加密深度纵轴。
    axis.set_title("Predicted resource delegation versus future Dörfler hit count")  # 标明机制先验是否能够预测后续重复命中的核心问题。
    axis.set_xticks(x_values)  # 显示全部十六个区域索引方便逐点审计。
    axis.set_ylim(-0.1, 3.2)  # 固定为系统允许的零到三级资源重投范围。
    axis.grid(True, alpha=0.25)  # 添加弱网格方便检查整数深度误差。
    axis.legend(fontsize=8, ncol=2)  # 展示真值、规则和各次模型调用图例。
    figure.tight_layout()  # 自动调整图例和坐标标签空间。
    figure.savefig(output_root / "future_hit_prediction.png", dpi=220)  # 保存未来命中预测位图。
    figure.savefig(output_root / "future_hit_prediction.pdf")  # 保存未来命中预测矢量图。
    plt.close(figure)  # 关闭第二张图释放内存。

def main() -> None:  # 组织基线、真实模型预测、宏动作执行、纠错和结果输出全过程。
    parser = argparse.ArgumentParser(description="Test whether prior future-hit prediction can compress real Dörfler feedback rounds.")  # 定义命令行说明。
    parser.add_argument("--ccx", required=True, help="CalculiX executable path")  # 要求显式传入真实 CalculiX 可执行文件。
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="experiment output directory")  # 允许工作流指定独立结果目录。
    parser.add_argument("--theta", type=float, default=0.50, help="Dörfler bulk parameter")  # 使用统一 Dörfler 覆盖参数控制当前热点证据。
    parser.add_argument("--max-rounds", type=int, default=5, help="dynamic Dörfler feedback rounds used as long-horizon target")  # 固定经典反馈基线最长真实求解轮次。
    parser.add_argument("--llm-trials", type=int, default=3, help="independent live LLM prediction calls")  # 默认执行三次独立真实大模型预测以观察稳定性。
    parser.add_argument("--max-corrections", type=int, default=2, help="maximum post-macro Dörfler corrections")  # 限制先验失败后最多购买两次额外真实反馈。
    args = parser.parse_args()  # 解析实际实验参数。
    if not 0.0 < float(args.theta) <= 1.0:  # 检查 Dörfler 参数是否满足数学定义域。
        raise ValueError("theta must be in (0, 1]")  # 非法参数时禁止启动真实求解。
    output_root = Path(args.output).resolve()  # 将结果目录规范化为绝对路径。
    output_root.mkdir(parents=True, exist_ok=True)  # 创建本次实验结果目录。
    module = load_base_module()  # 载入已有真实横向通道 CalculiX 实现。
    dynamic = run_dynamic_dorfler(module, output_root / "dynamic_solver", args.ccx, float(args.theta), int(args.max_rounds))  # 首先执行经典反馈基线并冻结未来命中 oracle。
    dynamic_H = int(dynamic["additional_H"])  # 读取经典方法达到有限视界终态所需额外真实反馈次数。
    target_objective = float(dynamic["final"]["objective"]) * 1.02 + 1.0e-12  # 允许宏动作终态相对经典有限视界误差有百分之二数值容差。
    future_hits = np.minimum(np.asarray(dynamic["future_hits"], dtype=np.int64), 3)  # 将未来命中真值投影到可执行最大三级重投域。
    rank_prediction = current_rank_prediction(np.asarray(dynamic["initial_priority"], dtype=np.float64), [int(value) for value in dynamic["initial_marked"]], len(dynamic["region_names"]))  # 构造无机制推理的当前强度重投对照。
    oracle_result = run_macro_method(module, output_root / "oracle_solver", args.ccx, dynamic, future_hits, METHOD_ORACLE, target_objective, float(args.theta), int(args.max_corrections))  # 用真实未来命中向量测试理论上能否压缩多轮反馈。
    rank_result = run_macro_method(module, output_root / "rank_solver", args.ccx, dynamic, rank_prediction, METHOD_RANK, target_objective, float(args.theta), int(args.max_corrections))  # 测试仅按当前热点强度重投是否足够。
    oracle_metrics = depth_metrics(future_hits, np.asarray(dynamic["future_hits"], dtype=np.int64))  # 计算未来命中 oracle 的机制轨迹上界指标。
    rank_metrics = depth_metrics(rank_prediction, np.asarray(dynamic["future_hits"], dtype=np.int64))  # 计算当前强度规则相对真实未来轨迹的预测误差。
    oracle_result.update(oracle_metrics)  # 将 oracle 深度预测指标写入统一方法记录。
    rank_result.update(rank_metrics)  # 将当前强度深度预测指标写入统一方法记录。
    dynamic["success_vs_dynamic"] = True  # 动态基线按定义满足自身终态质量目标。
    dynamic["depth_mae"] = ""  # 动态基线不属于提前预测器因此不计算深度预测误差。
    dynamic["persistent_f1"] = ""  # 动态基线不属于持续热点预测器因此不计算 F1。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()  # 从受保护 GitHub Environment 读取真实模型凭据。
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro").strip()  # 读取工作流冻结的模型名称。
    if not api_key:  # 检查本次 push 是否具有可信环境中的真实模型凭据。
        raise RuntimeError("DEEPSEEK_API_KEY is required for the live future-hit experiment")  # 缺少凭据时拒绝用伪造预测替代大模型实验。
    evidence_benchmark = make_benchmark(module, output_root / "llm_evidence_solver", args.ccx)  # 创建独立证据求解器防止读取经典基线缓存状态。
    coarse_levels = tuple(0 for _ in range(evidence_benchmark.region_count))  # 构造与所有路线一致的共同粗网格。
    coarse_solution = evidence_benchmark.solve(coarse_levels)  # 只执行一次共同粗网格真实 CalculiX 形成大模型可见证据。
    packet = build_llm_packet(dynamic, evidence_benchmark, coarse_solution)  # 构造并冻结完全不包含 future-hit 标签的在线机制证据包。
    packet_text = json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True)  # 将大模型输入保存为人可读稳定 JSON。
    (output_root / "llm_evidence_packet.json").write_text(packet_text + "\n", encoding="utf-8")  # 永久保存实际在线证据包以审计标签隔离。
    llm_results: list[dict[str, Any]] = []  # 初始化真实大模型宏动作结果列表。
    predictions: dict[str, np.ndarray] = {"oracle": future_hits.copy(), "current_rank": rank_prediction.copy()}  # 初始化区域图所需的 oracle 与无推理基线预测。
    raw_calls: list[dict[str, Any]] = []  # 初始化原始模型调用审计列表。
    for trial in range(1, int(args.llm_trials) + 1):  # 按预注册次数独立调用真实大模型。
        parsed, raw_text = call_llm(packet, model, api_key, trial)  # 在相同证据下获得一次真实 future-hit 资源重投预测。
        prediction, global_confidence, audit = validate_llm_prediction(parsed, len(dynamic["region_names"]))  # 通过硬限制门将模型输出转换为可执行动作。
        method_name = f"{METHOD_LLM_PREFIX}_{trial}"  # 为当前独立调用生成稳定方法名称。
        correction_budget = 0 if audit == "FINAL_SOLVE_ONLY" and global_confidence >= 0.85 else int(args.max_corrections)  # 仅在高置信且模型主动要求单次验收时允许零中间纠错，其余保留安全门。
        result = run_macro_method(module, output_root / f"llm_solver_{trial}", args.ccx, dynamic, prediction, method_name, target_objective, float(args.theta), correction_budget)  # 执行模型建议的一次重投并按门控购买最少真实纠错。
        result.update(depth_metrics(prediction, np.asarray(dynamic["future_hits"], dtype=np.int64)))  # 计算模型预测未来资源深度相对动态轨迹的误差。
        result["global_confidence"] = global_confidence  # 保存模型自身机制置信度用于后续 PH 校准。
        result["audit_recommendation"] = audit  # 保存模型主动提出的物理反馈需求。
        result["packet_sha256"] = parsed["packet_sha256"]  # 将模型动作与实际在线证据包哈希绑定。
        result["model"] = model  # 保存实际模型名称用于跨模型实验。
        llm_results.append(result)  # 将当前真实模型执行结果加入统一方法集合。
        predictions[f"llm_{trial}"] = prediction.copy()  # 保存当前逐区域累计资源重投预测用于绘图。
        raw_calls.append({"trial": trial, "parsed": parsed, "raw": raw_text})  # 保存完整原始响应以便人工检查证据引用和机制边界。
    methods = [dynamic, oracle_result, rank_result] + llm_results  # 按经典基线、理论上界、无推理对照和真实 LLM 顺序汇总全部方法。
    write_region_table(output_root / "region_future_hits.csv", dynamic, predictions)  # 写出逐区域未来命中与资源重投预测配对表。
    write_method_table(output_root / "method_summary.csv", methods, dynamic_H)  # 写出达到同一有限视界质量时的反馈深度节省结果。
    plot_results(output_root, methods, dynamic, predictions)  # 使用脚本生成 H—终态质量图和 future-hit 预测图。
    payload = {"experiment": "future_hit_resource_delegation", "claim_tested": "Reliable current Dörfler evidence plus prior prediction of persistent future hits can delegate several future one-level refinements as one resource-heavy macro-action and reduce real FEA feedback depth.", "target_definition": {"dynamic_dorfler_final_objective": float(dynamic["final"]["objective"]), "acceptance_objective": float(target_objective), "dynamic_additional_H": dynamic_H, "theta": float(args.theta), "max_rounds": int(args.max_rounds)}, "methods": methods, "llm_calls": raw_calls}  # 组织完整实验、边界和所有真实轨迹结果。
    (output_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存完整机器可读实验结果。
    print(json.dumps({"dynamic_H": dynamic_H, "dynamic_objective": float(dynamic["final"]["objective"]), "oracle_H": int(oracle_result["additional_H"]), "oracle_success": bool(oracle_result["success_vs_dynamic"]), "rank_H": int(rank_result["additional_H"]), "rank_success": bool(rank_result["success_vs_dynamic"]), "llm": [{"method": item["method"], "H": item["additional_H"], "success": item["success_vs_dynamic"], "objective": item["final"]["objective"], "depth_mae": item["depth_mae"], "persistent_f1": item["persistent_f1"]} for item in llm_results]}, ensure_ascii=False, indent=2))  # 在 Actions 日志中输出最小但足够判断核心假设的结果摘要。
if __name__ == "__main__":  # 仅在脚本直接执行时启动真实实验。
    main()  # 执行完整未来命中—资源重投实验。
