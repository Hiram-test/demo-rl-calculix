#!/usr/bin/env python3  # 使用稀疏机制委派接口快速测试真实大模型的未来资源重投能力。
from __future__ import annotations  # 启用现代类型注解并保持 Python 3.11 兼容。
import argparse  # 解析真实 CalculiX 路径和独立输出目录。
import json  # 保存紧凑证据、原始模型响应和真实物理执行结果。
import os  # 从 GitHub 受保护 Environment 读取真实 DeepSeek 凭据和模型名称。
import sys  # 将仓库根目录加入 Python 模块搜索路径。
from pathlib import Path  # 跨平台定位仓库与实验输出目录。
from typing import Any  # 为模型 JSON 稀疏动作提供通用类型注解。
import numpy as np  # 处理区域优先级、邻接关系和资源重投向量。
from openai import OpenAI  # 通过 OpenAI 兼容接口调用真实 DeepSeek 服务。
ROOT = Path(__file__).resolve().parents[2]  # 从当前实验目录返回仓库根目录。
PROTOCOL_VERSION = "sparse-delegation-v1"  # 冻结稀疏机制委派接口版本并触发已注册 Actions 工作流。
sys.path.insert(0, str(ROOT))  # 允许稳定导入已有真实 CalculiX 横向通道基准与 future-hit 工具函数。
from experiments.cross_passage_torsion_benchmark import run_benchmark as base_module  # 导入 PR24 已验证的真实 CalculiX 横向通道实现。
from experiments.future_hit_delegation import run as core  # 导入动态 Dörfler、宏动作执行和轨迹评价函数。

def clean_json(text: str) -> dict[str, Any]:  # 从模型可能附带 Markdown 围栏的输出中提取 JSON 对象。
    cleaned = text.strip()  # 去除响应首尾空白字符。
    if cleaned.startswith("```json"):  # 检查 JSON 专用 Markdown 围栏。
        cleaned = cleaned[7:].strip()  # 删除开头的 json 围栏标记。
    elif cleaned.startswith("```"):  # 检查普通 Markdown 围栏。
        cleaned = cleaned[3:].strip()  # 删除普通开头围栏标记。
    if cleaned.endswith("```"):  # 检查尾部 Markdown 围栏是否存在。
        cleaned = cleaned[:-3].strip()  # 删除尾部围栏并保留纯 JSON。
    return json.loads(cleaned)  # 将清理后的文本解析为结构化稀疏委派对象。

def build_compact_packet(dynamic: dict[str, Any], benchmark: Any, coarse_solution: Any) -> dict[str, Any]:  # 仅提供当前 Dörfler 热点及其一跳耦合区的紧凑机制证据。
    levels = tuple(int(value) for value in dynamic["initial_levels"])  # 读取所有方法共享的粗网格区域级别。
    features = benchmark.region_features(coarse_solution, levels)  # 从共同粗解提取能量、峰值、邻域对比和局部成本特征。
    priority = np.asarray(dynamic["initial_priority"], dtype=np.float64)  # 读取第一次 Dörfler 使用的同一当前局部价值。
    scale = max(float(np.max(priority)), 1.0e-18)  # 定义稳定的当前优先级归一化尺度。
    marked = {int(value) for value in dynamic["initial_marked"]}  # 将第一次可靠 Dörfler 支撑转换为集合便于构造耦合候选。
    candidate = set(marked)  # 未来资源委派至少考虑当前可靠热点本身。
    for index in list(marked):  # 遍历当前可靠热点寻找一跳物理耦合邻区。
        candidate.update(np.flatnonzero(benchmark.region_adjacency[index] > 0.0).astype(int).tolist())  # 将所有一跳相邻区域加入机制耦合候选集合。
    rows: list[dict[str, Any]] = []  # 初始化发送给模型的紧凑区域证据数组。
    for index in sorted(candidate):  # 只遍历当前热点与其一跳邻区避免强迫模型处理无关完整网格。
        rows.append({"id": int(index), "name": str(dynamic["region_names"][index]), "marked_now": bool(index in marked), "q": round(float(priority[index] / scale), 4), "energy": round(float(features[index, 0]), 4), "peak": round(float(features[index, 1]), 4), "contrast": round(float(features[index, 2]), 4), "cost": round(float(features[index, 4]), 4), "neighbors": np.flatnonzero(benchmark.region_adjacency[index] > 0.0).astype(int).tolist()})  # 提供足够判断持续性和耦合的当前物理证据。
    return {"goal": "Using reliable current Dörfler evidence, delegate multiple future one-level refinements now only where persistence is justified, so real FEA feedback rounds can be reduced.", "rules": ["Do not replace current Dörfler marking.", "Choose only regions that deserve advance allocation because they are likely to stay important or become immediately exposed by coupling.", "depth means cumulative extra mesh levels to allocate now: 1, 2, or 3.", "Omit uncertain regions; omitted regions receive depth 0 by the deterministic compiler.", "Do not output explanations outside JSON."], "current_marked": sorted(marked), "candidate_regions": rows, "output_schema": {"delegations": [{"id": "candidate region integer", "depth": "1|2|3", "confidence": "0..1", "mechanism": "short phrase", "evidence": ["q|energy|peak|contrast|neighbors|marked_now"]}], "global_confidence": "0..1"}}  # 返回稀疏而可编译的机制资源委派合同。

def compile_sparse(parsed: dict[str, Any], region_count: int, candidate_ids: set[int]) -> tuple[np.ndarray, float]:  # 将模型少量机制委派建议确定性扩展为完整区域动作向量。
    prediction = np.zeros(region_count, dtype=np.int64)  # 所有未被模型明确委派的区域默认不提前投入资源。
    rows = parsed.get("delegations", [])  # 读取模型稀疏资源委派列表并允许合法空列表代表主动保守。
    if not isinstance(rows, list):  # 检查稀疏动作顶层类型是否符合执行合同。
        raise ValueError("delegations must be a list")  # 非列表输出无法由确定性 compiler 执行因此拒绝。
    seen: set[int] = set()  # 初始化重复区域检查集合防止同一区域被多次冲突委派。
    for row in rows:  # 遍历模型主动选择的少量未来持续热点或耦合区。
        index = int(row["id"])  # 读取结构区域编号。
        if index not in candidate_ids or index in seen:  # 限制模型只能在当前热点及其一跳候选域中进行资源委派。
            raise ValueError(f"invalid sparse delegation region {index}")  # 超出候选域或重复委派时拒绝执行。
        seen.add(index)  # 登记当前区域已通过执行门。
        prediction[index] = int(np.clip(int(row["depth"]), 1, 3))  # 将资源重投深度硬限制到一至三级。
    confidence = float(np.clip(float(parsed.get("global_confidence", 0.0)), 0.0, 1.0))  # 将模型全局机制置信度规范到零到一。
    return prediction, confidence  # 返回完整十六维可执行动作与模型置信度。

def main() -> None:  # 执行动态 Dörfler、一次稀疏真实模型委派、确定性编译和真实 CalculiX 审计。
    parser = argparse.ArgumentParser(description="Sparse live LLM mechanism-delegation experiment.")  # 定义命令行实验说明。
    parser.add_argument("--ccx", required=True, help="CalculiX executable path")  # 要求显式传入真实 CalculiX 求解器路径。
    parser.add_argument("--output", required=True, help="output directory")  # 要求显式指定独立结果目录。
    parser.add_argument("--theta", type=float, default=0.50, help="Dörfler bulk parameter")  # 冻结当前可靠热点证据的 Dörfler 参数。
    parser.add_argument("--max-rounds", type=int, default=5, help="dynamic Dörfler horizon")  # 冻结形成未来命中真值的经典有限视界。
    args = parser.parse_args()  # 解析真实实验参数。
    output_root = Path(args.output).resolve()  # 将结果目录规范化为绝对路径。
    output_root.mkdir(parents=True, exist_ok=True)  # 创建本次稀疏真实模型实验结果目录。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()  # 从受保护 Environment 读取真实 API 凭据。
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro").strip()  # 读取冻结真实模型名称。
    if not api_key:  # 检查受保护凭据是否真实可用。
        raise RuntimeError("DEEPSEEK_API_KEY is required")  # 缺少凭据时禁止使用任何规则模拟 LLM 输出。
    dynamic = core.run_dynamic_dorfler(base_module, output_root / "dynamic_solver", args.ccx, float(args.theta), int(args.max_rounds))  # 执行真实动态 Dörfler 并记录未来重复命中真值。
    dynamic_h = int(dynamic["additional_H"])  # 读取动态基线相对共同粗解的真实反馈深度。
    target_objective = float(dynamic["final"]["objective"]) * 1.02 + 1.0e-12  # 使用动态终态误差加百分之二容差定义统一成功门。
    evidence_benchmark = core.make_benchmark(base_module, output_root / "evidence_solver", args.ccx)  # 创建缓存隔离的共同粗解证据求解器。
    coarse_levels = tuple(0 for _ in range(evidence_benchmark.region_count))  # 构造全部区域最粗网格的共同在线起点。
    coarse_solution = evidence_benchmark.solve(coarse_levels)  # 执行一次真实 CalculiX 获得模型可见的当前物理证据。
    packet = build_compact_packet(dynamic, evidence_benchmark, coarse_solution)  # 将当前 Dörfler 支撑及其耦合邻区压缩为稀疏机制证据合同。
    (output_root / "compact_evidence_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存实际模型输入供标签隔离和机制证据审计。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=55.0, max_retries=0)  # 建立五十五秒硬超时且禁止自动重试的真实模型客户端。
    system_text = "You are a finite-element resource-delegation planner. Current Dörfler hotspots are reliable. Select only a sparse set of regions for advance multi-level allocation based on persistence or immediate physical coupling. Return valid JSON only, with keys delegations and global_confidence."  # 明确模型只承担人工式跨轮次资源委派而不承担逐元素 marking。
    completion = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_text}, {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}], temperature=0.0, max_tokens=1800)  # 使用紧凑提示执行一次真实稀疏机制委派调用。
    raw_text = completion.choices[0].message.content or "{}"  # 提取模型原始响应并保留空对象作为可审计失败状态。
    (output_root / "raw_llm_response.txt").write_text(raw_text + "\n", encoding="utf-8")  # 永久保存未改写的真实模型输出。
    parsed = clean_json(raw_text)  # 将模型返回文本解析为稀疏机制委派结构。
    candidate_ids = {int(row["id"]) for row in packet["candidate_regions"]}  # 提取允许模型委派资源的预注册候选区域集合。
    prediction, confidence = compile_sparse(parsed, evidence_benchmark.region_count, candidate_ids)  # 使用确定性 compiler 将稀疏 LLM 建议扩展为完整十六维动作。
    result = core.run_macro_method(base_module, output_root / "llm_solver", args.ccx, dynamic, prediction, "llm_sparse_delegate", target_objective, float(args.theta), 2)  # 执行一次宏动作并在失败时最多购买两轮真实 Dörfler 纠错。
    result.update(core.depth_metrics(prediction, np.asarray(dynamic["future_hits"], dtype=np.int64)))  # 评价资源深度预测相对真实未来命中轨迹的误差与持续热点 F1。
    payload = {"model": model, "dynamic_H": dynamic_h, "dynamic_objective": dynamic["final"]["objective"], "target_objective": target_objective, "initial_marked": dynamic["initial_marked"], "future_hits": dynamic["future_hits"], "prediction": prediction.astype(int).tolist(), "global_confidence": confidence, "compiled_from_sparse": True, "llm_result": result}  # 组织完整模型、预测、反馈压缩和真实物理终态结果。
    (output_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存机器可读真实模型实验结果。
    print(json.dumps({"model": model, "dynamic_H": dynamic_h, "dynamic_objective": dynamic["final"]["objective"], "llm_H": result["additional_H"], "H_saved": dynamic_h - int(result["additional_H"]), "success": result["success_vs_dynamic"], "llm_objective": result["final"]["objective"], "depth_mae": result["depth_mae"], "persistent_f1": result["persistent_f1"], "global_confidence": confidence, "future_hits": dynamic["future_hits"], "prediction": prediction.astype(int).tolist(), "raw_response": raw_text}, ensure_ascii=False, indent=2))  # 在 Actions 日志输出核心真实模型行为和数值结果便于即时判断。
if __name__ == "__main__":  # 仅在脚本被工作流直接执行时启动实验。
    main()  # 执行完整稀疏机制委派—确定性编译—真实审计过程。
