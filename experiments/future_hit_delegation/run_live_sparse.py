#!/usr/bin/env python3  # 使用自然语言机制委派接口测试真实大模型的未来资源重投能力。
from __future__ import annotations  # 启用现代类型注解并保持 Python 3.11 兼容。
import argparse  # 解析真实 CalculiX 路径、输出目录和有限视界参数。
import json  # 保存证据包、API 元数据、解析结果和真实物理执行结果。
import os  # 从受保护 GitHub Environment 读取真实 DeepSeek 凭据和模型名称。
import re  # 从自然语言或紧凑文本中确定性提取 region、depth 和 confidence。
import sys  # 将仓库根目录加入 Python 模块搜索路径。
from pathlib import Path  # 跨平台定位仓库与实验输出目录。
from typing import Any  # 为模型响应、证据包和解析记录提供通用类型注解。
import numpy as np  # 处理区域优先级、邻接关系、网格级别和资源重投向量。
from openai import OpenAI  # 通过 OpenAI 兼容接口调用真实 DeepSeek 服务。
ROOT = Path(__file__).resolve().parents[2]  # 从当前实验目录返回仓库根目录。
PROTOCOL_VERSION = "sparse-mechanism-delegation-v2"  # 冻结修复后的自然语言机制委派协议版本。
sys.path.insert(0, str(ROOT))  # 允许稳定导入已有真实 CalculiX 横向通道基准与 future-hit 工具函数。
from experiments.cross_passage_torsion_benchmark import run_benchmark as base_module  # 导入 PR24 已验证的真实 CalculiX 横向通道实现。
from experiments.future_hit_delegation import run as core  # 导入动态 Dörfler、宏动作执行和轨迹评价函数。

def build_compact_packet(dynamic: dict[str, Any], benchmark: Any, coarse_solution: Any) -> dict[str, Any]:  # 构造理论结构信息与共同粗网格证据融合的紧凑机制证据包。
    levels = tuple(int(value) for value in dynamic["initial_levels"])  # 读取所有路线共享的粗网格区域级别。
    features = benchmark.region_features(coarse_solution, levels)  # 从共同粗解提取能量、峰值、邻域对比和局部成本特征。
    priority = np.asarray(dynamic["initial_priority"], dtype=np.float64)  # 读取第一次 Dörfler 使用的同一当前局部价值。
    scale = max(float(np.max(priority)), 1.0e-18)  # 定义稳定的当前优先级归一化尺度。
    marked = {int(value) for value in dynamic["initial_marked"]}  # 将第一次可靠 Dörfler 支撑转换为集合。
    candidate = set(marked)  # 未来资源委派至少考虑当前可靠热点本身。
    for index in list(marked):  # 遍历当前可靠热点寻找一跳结构耦合邻区。
        candidate.update(np.flatnonzero(benchmark.region_adjacency[index] > 0.0).astype(int).tolist())  # 将一跳相邻区域加入机制候选域。
    rows: list[dict[str, Any]] = []  # 初始化模型可见的候选区域证据数组。
    for index in sorted(candidate):  # 只遍历当前热点及其一跳邻区避免无关全域搜索。
        name = str(dynamic["region_names"][index])  # 读取区域结构名称以保留有限的工程语义。
        if index <= 4:  # 检查当前区域是否属于五个沿跨度布置的弦杆面板。
            role = f"longitudinal chords in panel {index + 1} from fixed end toward twisted end"  # 明确弦杆区域在传力体系中的空间角色。
        elif index <= 10:  # 检查当前区域是否属于六个横向框架站点。
            role = f"transverse frame at station {index - 4}, station 1 fixed and station 6 twisted"  # 明确横框区域与边界的相对位置。
        else:  # 处理最后五个沿跨度布置的空间斜撑面板。
            role = f"X bracing in panel {index - 10} from fixed end toward twisted end"  # 明确斜撑区域承担扭转载荷传递的结构角色。
        rows.append({"id": int(index), "name": name, "structural_role": role, "marked_now": bool(index in marked), "q": round(float(priority[index] / scale), 4), "energy": round(float(features[index, 0]), 4), "peak": round(float(features[index, 1]), 4), "contrast": round(float(features[index, 2]), 4), "cost": round(float(features[index, 4]), 4), "neighbors": np.flatnonzero(benchmark.region_adjacency[index] > 0.0).astype(int).tolist()})  # 写入可用于持续性判断的当前物理证据。
    physics = {"model": "six-station 3D box-like space frame with B31 beams", "boundary": "station 1 is fully fixed", "loading": "station 6 receives a prescribed small twist about the longitudinal span axis", "material": "linear elastic steel", "adaptation": "each selected region can receive one to three additional subdivision levels", "mechanism_hint": "torsional load is transmitted through coupled chords, transverse frames and X bracing; persistent regions should remain important after local discretization error is reduced"}  # 提供不包含未来命中标签的结构力学上下文。
    return {"protocol": PROTOCOL_VERSION, "goal": "Predict only which candidate regions justify advance multi-level resource allocation because their present error is likely to persist across later Dörfler rounds.", "physics_context": physics, "current_marked": sorted(marked), "candidate_regions": rows, "decision_rules": ["Treat current Dörfler marking as reliable present-state evidence.", "Do not invent a new global hotspot map.", "Use structural load-transfer reasoning together with coarse-grid q, energy, peak and contrast evidence.", "A depth of 1, 2 or 3 means how many future one-level refinement actions can be safely delegated now.", "If no advance delegation is justified, explicitly answer ABSTAIN rather than returning an empty object."]}  # 返回不泄漏 future-hit oracle 的机制证据合同。

def extract_plain_delegations(text: str, candidate_ids: set[int]) -> tuple[list[dict[str, Any]], bool]:  # 从自由文本中确定性提取少量区域与累计重投深度。
    cleaned = text.strip()  # 去除响应首尾空白以稳定判断接口状态。
    if not cleaned or cleaned in {"{}", "[]"}:  # 将空字符串和空容器明确判定为接口失败而非保守科学决策。
        return [], False  # 返回无委派并标记接口无效。
    if re.search(r"\bABSTAIN\b", cleaned, flags=re.IGNORECASE):  # 检查模型是否明确作出可审计的保守弃权。
        return [], True  # 显式弃权属于有效机制判断而非接口故障。
    patterns = [r"REGION\s*[:#]?\s*(\d+)\s*[,;| ]+\s*DEPTH\s*[:=]?\s*([123])(?:\s*[,;| ]+\s*CONF(?:IDENCE)?\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?))?", r"\b(\d+)\s*[:=]\s*([123])(?:\s*[@,]\s*(0(?:\.\d+)?|1(?:\.0+)?))?", r"region\s+(\d+).*?(?:depth|levels?)\s*(?:of|=|:)?\s*([123])(?:.*?(?:confidence|conf)\s*(?:=|:)?\s*(0(?:\.\d+)?|1(?:\.0+)?))?"]  # 定义从严格行格式到自然语言的三级容错模式。
    records: dict[int, dict[str, Any]] = {}  # 使用区域编号去重并保留最先出现的有效委派。
    for pattern in patterns:  # 依次尝试不同表达形式以降低格式约束对科学实验的干扰。
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL):  # 搜索当前模式命中的所有区域委派表达。
            index = int(match.group(1))  # 读取候选结构区域编号。
            depth = int(match.group(2))  # 读取模型建议的累计额外加密层数。
            confidence = float(match.group(3)) if match.lastindex is not None and match.lastindex >= 3 and match.group(3) is not None else 0.60  # 在缺少置信度时采用中性默认值而不猜高置信。
            if index in candidate_ids and index not in records:  # 只接受预注册候选域中的首个合法委派。
                records[index] = {"id": index, "depth": depth, "confidence": float(np.clip(confidence, 0.0, 1.0))}  # 保存确定性规范化后的稀疏动作。
        if records:  # 检查当前较严格模式是否已经得到至少一个有效动作。
            break  # 已提取有效稀疏委派时停止更宽松的正则扫描。
    return list(records.values()), bool(records)  # 仅当存在明确委派或显式 ABSTAIN 时认为接口有效。

def call_mechanism_planner(client: OpenAI, model: str, packet: dict[str, Any], output_root: Path) -> tuple[list[dict[str, Any]], str, int, bool]:  # 通过最多两次简洁自然语言调用获得可解析机制委派。
    candidate_ids = {int(row["id"]) for row in packet["candidate_regions"]}  # 提取预注册允许委派的候选区域集合。
    candidate_table = "\n".join(f"region {row['id']} | {row['structural_role']} | marked={row['marked_now']} | q={row['q']} | energy={row['energy']} | peak={row['peak']} | contrast={row['contrast']} | neighbors={row['neighbors']}" for row in packet["candidate_regions"])  # 将结构化证据压缩成模型容易阅读的逐区域文本表。
    context_text = f"Structure: {packet['physics_context']['model']}. Boundary: {packet['physics_context']['boundary']}. Loading: {packet['physics_context']['loading']}. Mechanism cue: {packet['physics_context']['mechanism_hint']}."  # 将核心理论上下文转换为简洁自然语言。
    first_prompt = f"{context_text}\nCurrent Dörfler-marked regions are {packet['current_marked']}. Current evidence:\n{candidate_table}\nDecide which few candidate regions are likely to remain important for multiple later refinement rounds. Do not redesign the current hotspot detector. For each region you choose, write one short line exactly like REGION 13 DEPTH 3 CONF 0.8. DEPTH is cumulative future one-level refinements to delegate now and must be 1, 2, or 3. Add at most one short sentence of mechanism summary after the lines. If none is justified, write ABSTAIN. Do not use JSON or braces."  # 第一尝试采用人类式机制判断加极轻格式锚点。
    retry_prompt = f"{context_text}\nCandidates are only {sorted(candidate_ids)}. Current marked regions are {packet['current_marked']}. Choose persistent regions using load path plus current numerical evidence. Reply ONLY with semicolon-separated pairs such as 13:3;14:3;15:2 where left is region id and right is depth 1-3. If you cannot justify any, reply ABSTAIN. Never reply with JSON, {{}} or []."  # 第二尝试进一步降低格式复杂度并明确禁止空 JSON。
    prompts = [first_prompt, retry_prompt]  # 固定最多两次接口尝试以控制模型成本和实验可复现性。
    raw_responses: list[str] = []  # 初始化原始模型响应列表供完整接口审计。
    for attempt_index, prompt in enumerate(prompts, start=1):  # 顺序执行自然语言机制判断与一次格式降级重试。
        completion = client.chat.completions.create(model=model, messages=[{"role": "system", "content": "Act as a finite-element adaptation expert. Make a concise, auditable resource-delegation decision from the supplied mechanics and coarse-grid evidence. Do not output chain-of-thought."}, {"role": "user", "content": prompt}], temperature=0.0, max_tokens=700)  # 调用真实模型并只要求短决策摘要而非完整控制向量。
        raw_text = completion.choices[0].message.content or ""  # 读取模型最终公开回答并禁止把空响应替换成伪造动作。
        raw_responses.append(raw_text)  # 保存当前真实响应以区分接口故障、弃权和有效委派。
        (output_root / f"raw_llm_response_attempt_{attempt_index}.txt").write_text(raw_text + "\n", encoding="utf-8")  # 将每次未改写响应冻结到结果目录。
        metadata = {"attempt": attempt_index, "model": model, "finish_reason": str(completion.choices[0].finish_reason), "usage": completion.usage.model_dump() if completion.usage is not None else None, "content_length": len(raw_text)}  # 只保存非推理型 API 元数据用于诊断输出异常。
        (output_root / f"api_metadata_attempt_{attempt_index}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存结束原因、token 用量和内容长度而不依赖隐藏推理字段。
        delegations, valid = extract_plain_delegations(raw_text, candidate_ids)  # 使用确定性解析器提取少量 region/depth 决策。
        if valid:  # 检查当前输出是否形成有效委派或显式弃权。
            return delegations, raw_text, attempt_index, True  # 成功后立即返回并避免额外模型调用。
    return [], "\n---RETRY---\n".join(raw_responses), len(prompts), False  # 两次均无有效公开决策时明确返回接口失败。

def compile_sparse(delegations: list[dict[str, Any]], region_count: int, candidate_ids: set[int]) -> tuple[np.ndarray, float]:  # 将少量机制委派确定性扩展为完整区域资源动作向量。
    prediction = np.zeros(region_count, dtype=np.int64)  # 所有未被明确委派的区域默认不提前投入资源。
    confidence_values: list[float] = []  # 初始化有效委派置信度列表用于生成全局审计值。
    seen: set[int] = set()  # 初始化重复区域检查集合。
    for row in delegations:  # 遍历确定性解析器得到的少量持续热点委派。
        index = int(row["id"])  # 读取结构区域编号。
        if index not in candidate_ids or index in seen:  # 限制动作只能进入当前热点及其耦合候选域且禁止冲突重复。
            raise ValueError(f"invalid sparse delegation region {index}")  # 非法动作在网格执行前硬拒绝。
        seen.add(index)  # 登记当前区域已经通过确定性执行门。
        prediction[index] = int(np.clip(int(row["depth"]), 1, 3))  # 将累计重投深度硬限制到一至三级。
        confidence_values.append(float(np.clip(float(row.get("confidence", 0.60)), 0.0, 1.0)))  # 保存经过范围门约束的区域置信度。
    global_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0  # 以明确委派的平均置信度形成可审计全局值。
    return prediction, global_confidence  # 返回完整十六维动作和数值审计置信度。

def main() -> None:  # 执行真实动态 Dörfler、自然语言机制委派、确定性编译与真实 CalculiX 审计。
    parser = argparse.ArgumentParser(description="Live LLM future-hit mechanism delegation with deterministic compilation.")  # 定义命令行实验说明。
    parser.add_argument("--ccx", required=True, help="CalculiX executable path")  # 要求显式传入真实 CalculiX 求解器路径。
    parser.add_argument("--output", required=True, help="output directory")  # 要求显式指定独立结果目录。
    parser.add_argument("--theta", type=float, default=0.50, help="Dörfler bulk parameter")  # 冻结当前可靠热点证据的 Dörfler 参数。
    parser.add_argument("--max-rounds", type=int, default=5, help="dynamic Dörfler horizon")  # 冻结形成未来命中真值的经典有限视界。
    args = parser.parse_args()  # 解析真实实验参数。
    output_root = Path(args.output).resolve()  # 将结果目录规范化为绝对路径。
    output_root.mkdir(parents=True, exist_ok=True)  # 创建本次真实模型实验结果目录。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()  # 从受保护 Environment 读取真实 API 凭据。
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro").strip()  # 读取冻结真实模型名称。
    if not api_key:  # 检查受保护凭据是否真实可用。
        raise RuntimeError("DEEPSEEK_API_KEY is required")  # 缺少凭据时禁止用规则伪装真实 LLM 输出。
    dynamic = core.run_dynamic_dorfler(base_module, output_root / "dynamic_solver", args.ccx, float(args.theta), int(args.max_rounds))  # 执行真实动态 Dörfler 并记录 future-hit 离线真值。
    dynamic_h = int(dynamic["additional_H"])  # 读取动态基线相对共同粗解的真实反馈深度。
    target_objective = float(dynamic["final"]["objective"]) * 1.02 + 1.0e-12  # 使用动态终态误差加百分之二容差定义统一终态质量门。
    evidence_benchmark = core.make_benchmark(base_module, output_root / "evidence_solver", args.ccx)  # 创建缓存隔离的共同粗解证据求解器。
    coarse_levels = tuple(0 for _ in range(evidence_benchmark.region_count))  # 构造全部区域最粗网格的共同在线起点。
    coarse_solution = evidence_benchmark.solve(coarse_levels)  # 执行一次真实 CalculiX 获得 LLM 可见的当前物理证据。
    packet = build_compact_packet(dynamic, evidence_benchmark, coarse_solution)  # 将理论结构信息与粗网格 Dörfler 证据融合成机制判断输入。
    (output_root / "compact_evidence_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存实际模型输入以证明无 future-hit 标签泄漏。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=55.0, max_retries=0)  # 建立五十五秒硬超时且禁止隐藏自动重试的真实模型客户端。
    delegations, raw_text, attempt_count, interface_valid = call_mechanism_planner(client, model, packet, output_root)  # 让模型只做少量人工式持续性判断并允许一次轻格式重试。
    candidate_ids = {int(row["id"]) for row in packet["candidate_regions"]}  # 提取确定性 compiler 允许执行的候选区域集合。
    if not interface_valid:  # 检查两次公开模型回答是否仍未形成有效决策或明确弃权。
        failure_payload = {"protocol": PROTOCOL_VERSION, "model": model, "interface_valid": False, "attempt_count": attempt_count, "raw_response": raw_text, "dynamic_H": dynamic_h, "dynamic_objective": dynamic["final"]["objective"], "future_hits": dynamic["future_hits"], "initial_marked": dynamic["initial_marked"]}  # 组织纯接口故障证据并禁止把它误解释为全零算法动作。
        (output_root / "interface_failure.json").write_text(json.dumps(failure_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 冻结真实接口故障用于后续客户端诊断。
        raise RuntimeError("live LLM returned no parseable delegation and did not explicitly ABSTAIN")  # 直接失败使 CI 明确区分接口问题与科学性能问题。
    prediction, confidence = compile_sparse(delegations, evidence_benchmark.region_count, candidate_ids)  # 将有效稀疏委派确定性编译成完整十六维网格动作。
    result = core.run_macro_method(base_module, output_root / "llm_solver", args.ccx, dynamic, prediction, "llm_sparse_mechanism_delegate_v2", target_objective, float(args.theta), 2)  # 执行宏动作并在终态未通过时最多购买两轮真实 Dörfler 纠错。
    result.update(core.depth_metrics(prediction, np.asarray(dynamic["future_hits"], dtype=np.int64)))  # 评价累计深度相对真实 future-hit 轨迹的 MAE 与持续热点 F1。
    payload = {"protocol": PROTOCOL_VERSION, "model": model, "interface_valid": True, "attempt_count": attempt_count, "delegations": delegations, "dynamic_H": dynamic_h, "dynamic_objective": dynamic["final"]["objective"], "target_objective": target_objective, "initial_marked": dynamic["initial_marked"], "future_hits": dynamic["future_hits"], "prediction": prediction.astype(int).tolist(), "global_confidence": confidence, "compiled_from_sparse": True, "raw_response": raw_text, "llm_result": result}  # 组织机制判断、确定性编译和真实物理终态的完整结果。
    (output_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存机器可读真实模型实验结果。
    summary = {"protocol": PROTOCOL_VERSION, "model": model, "interface_valid": True, "attempt_count": attempt_count, "delegations": delegations, "dynamic_H": dynamic_h, "dynamic_objective": dynamic["final"]["objective"], "llm_H": result["additional_H"], "H_saved": dynamic_h - int(result["additional_H"]), "success": result["success_vs_dynamic"], "llm_objective": result["final"]["objective"], "depth_mae": result["depth_mae"], "persistent_f1": result["persistent_f1"], "global_confidence": confidence, "future_hits": dynamic["future_hits"], "prediction": prediction.astype(int).tolist()}  # 构造 Actions 日志中的短科学摘要并省略模型自由文本。
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # 输出只含可审计决策和数值结果的运行摘要。
if __name__ == "__main__":  # 检查脚本是否由 Actions 或命令行直接执行。
    main()  # 启动修复后的完整真实 LLM 机制委派实验。
