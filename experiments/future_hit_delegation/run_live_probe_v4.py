#!/usr/bin/env python3  # 使用一次真实 Dörfler 探针后的响应证据测试 LLM 剩余轨迹资源委派能力。
from __future__ import annotations  # 启用现代类型注解并保持 Python 3.11 兼容。
import argparse  # 解析真实 CalculiX 路径、输出目录和有限视界参数。
import json  # 保存探针前后证据、API 公开统计、委派结果和真实物理轨迹。
import os  # 从受保护 GitHub Environment 读取 DeepSeek 凭据和模型名称。
import sys  # 将仓库根目录加入 Python 模块搜索路径。
from pathlib import Path  # 跨平台定位仓库和实验结果目录。
from typing import Any  # 为 API 响应和证据对象提供通用类型注解。
import numpy as np  # 处理局部指标、网格级别、future-hit 和统计评价。
from openai import OpenAI  # 通过 OpenAI 兼容接口调用真实 DeepSeek 推理模型。
ROOT = Path(__file__).resolve().parents[2]  # 从当前实验目录返回仓库根目录。
PROTOCOL_VERSION = "one-probe-persistence-delegation-v4"  # 冻结一次物理探针后预测剩余累计深度的实验协议。
sys.path.insert(0, str(ROOT))  # 允许稳定导入已有真实 CalculiX 基准与资源委派工具。
from experiments.cross_passage_torsion_benchmark import run_benchmark as base_module  # 导入 PR24 已验证的真实 CalculiX 横向通道实现。
from experiments.future_hit_delegation import run as core  # 导入 Dörfler marking、预算修复、轨迹记录和深度评价函数。
from experiments.future_hit_delegation import run_live_sparse as helper  # 复用自然语言稀疏动作 parser 和 deterministic compiler。
from experiments.future_hit_delegation import run_live_sparse_v3 as reasoning_helper  # 复用 reasoning-aware 公开 token 统计函数。

def structural_role(index: int) -> str:  # 将十六个固定区域编号映射为可用于机制推理的结构角色。
    if index <= 4:  # 检查是否属于五个纵向弦杆面板。
        return f"longitudinal chords in panel {index + 1} from fixed end toward twisted end"  # 返回弦杆面板的跨度位置和构造角色。
    if index <= 10:  # 检查是否属于六个横向框架站点。
        return f"transverse frame at station {index - 4}, station 1 fixed and station 6 twisted"  # 返回横框站点及其与边界条件的关系。
    return f"X bracing in panel {index - 10} from fixed end toward twisted end"  # 返回空间斜撑面板及其沿扭转载荷路径的位置。

def make_probe_evidence(dynamic: dict[str, Any], benchmark: Any, coarse_solution: Any, probe_solution: Any, coarse_levels: tuple[int, ...], probe_levels: tuple[int, ...], theta: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[int], list[int]]:  # 构造一次真实加密前后的区域响应证据。
    q0 = benchmark.hotspot_priority(coarse_solution, coarse_levels)  # 计算共同粗网格的当前区域价值证据。
    q1 = benchmark.hotspot_priority(probe_solution, probe_levels)  # 计算一次标准 Dörfler 加密后的新区域价值证据。
    m0 = core.dorfler_mark(q0, theta)  # 在粗网格上得到可靠初始 Dörfler 支撑。
    m1 = core.dorfler_mark(q1, theta)  # 在探针网格上得到一次真实反馈后的新 Dörfler 支撑。
    scale0 = max(float(np.max(q0)), 1.0e-18)  # 定义粗网格区域价值归一化尺度。
    scale1 = max(float(np.max(q1)), 1.0e-18)  # 定义探针后区域价值归一化尺度。
    candidate = set(m0).union(m1)  # 将两次真实 Dörfler 支撑并入持续性候选集合。
    for index in list(candidate):  # 为已命中区域补充一跳耦合邻区以允许热点迁移判断。
        candidate.update(np.flatnonzero(benchmark.region_adjacency[index] > 0.0).astype(int).tolist())  # 将结构邻接区域加入候选集合而不读取未来标签。
    rows: list[dict[str, Any]] = []  # 初始化逐区域探针响应证据。
    for index in sorted(candidate):  # 遍历当前、探针后热点及其一跳耦合区。
        before = float(q0[index] / scale0)  # 计算探针前归一化区域价值。
        after = float(q1[index] / scale1)  # 计算探针后归一化区域价值。
        retention = after / max(before, 1.0e-6)  # 计算一次真实局部资源投入后的相对持续性代理。
        rows.append({"id": int(index), "role": structural_role(index), "marked_before": bool(index in m0), "marked_after_probe": bool(index in m1), "q_before": round(before, 4), "q_after": round(after, 4), "retention_ratio": round(float(retention), 4), "probe_level": int(probe_levels[index]), "neighbors": np.flatnonzero(benchmark.region_adjacency[index] > 0.0).astype(int).tolist()})  # 保存直接反映局部误差响应与持续性的物理探针证据。
    packet = {"protocol": PROTOCOL_VERSION, "physics": "six-station linear-elastic 3D B31 space frame; station 1 fixed; station 6 prescribed torsional twist", "decision_state": "one real Dörfler refinement has already been executed and solved", "goal": "predict only the remaining cumulative refinement depth that can now be delegated before the next real FEA", "current_marked_after_probe": m1, "regions": rows, "rules": ["Use q_before→q_after response and structural load-transfer role together.", "A region that remains marked and retains high normalized q after receiving refinement is evidence for persistent future demand.", "Do not repeat the already executed probe level in the predicted remaining depth.", "Choose only a few regions with remaining depth 1, 2 or 3; uncertain regions receive zero by deterministic compilation."]}  # 返回不包含长程 future-hit 标签的一次探针机制合同。
    return packet, q0, q1, m0, m1  # 返回证据包和可用于执行审计的两轮真实局部指标。

def build_prompt(packet: dict[str, Any]) -> str:  # 将探针响应证据转换为简洁的人工专家式剩余深度判断任务。
    table = "\n".join(f"REGION {row['id']}: {row['role']}; marked_before={row['marked_before']}; marked_after={row['marked_after_probe']}; q_before={row['q_before']}; q_after={row['q_after']}; retention={row['retention_ratio']}; probe_level={row['probe_level']}; neighbors={row['neighbors']}" for row in packet["regions"])  # 编译逐区域探针前后响应表。
    return f"A real FE solve has already been performed after one standard Dörfler refinement. Current marked regions after that probe are {packet['current_marked_after_probe']}. The structure is a six-station linear-elastic space frame under prescribed end twist, with station 1 fixed and station 6 twisted. The table shows how each candidate responded to the first real refinement:\n{table}\nUse mechanics plus the observed response to predict which FEW regions will still require repeated refinement in later rounds. For each chosen region output remaining cumulative depth, excluding the probe level already executed. Final answer only: one line per choice like REGION 13 DEPTH 2 CONF 0.9. If no remaining multi-round allocation is justified, answer ABSTAIN. Do not output JSON or hidden reasoning."  # 要求模型利用真实干预响应而非只按当前强度排序。

def call_llm(client: OpenAI, model: str, prompt: str, candidate_ids: set[int], output_root: Path) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:  # 使用 reasoning-aware 预算获取最终公开剩余资源委派。
    budgets = [4096, 8192]  # 复用已验证能够避免全部 token 被 reasoning 吃满的两级 completion 预算。
    audit_rows: list[dict[str, Any]] = []  # 初始化公开 API token 和结束原因记录。
    raw_answers: list[str] = []  # 初始化最终公开模型答案列表。
    for attempt, budget in enumerate(budgets, start=1):  # 最多执行两次真实模型调用并仅在推理截断时升级预算。
        completion = client.chat.completions.create(model=model, messages=[{"role": "system", "content": "Act as a finite-element adaptation expert. Return only the requested concise final region/depth decisions."}, {"role": "user", "content": prompt}], temperature=0.0, max_tokens=budget)  # 调用真实推理模型并允许充分内部推理后输出短最终答案。
        raw = completion.choices[0].message.content or ""  # 只读取最终公开 answer 字段而不读取隐藏推理正文。
        usage = reasoning_helper.public_usage(completion)  # 读取公开 reasoning-token 统计用于判断是否发生长度截断。
        finish = str(completion.choices[0].finish_reason)  # 保存公开完成原因。
        audit = {"attempt": attempt, "max_tokens": budget, "finish_reason": finish, "content_length": len(raw), **usage}  # 构造本次模型调用的公开接口审计记录。
        audit_rows.append(audit)  # 保存本次 API 调用统计。
        raw_answers.append(raw)  # 保存本次最终公开答案用于复核。
        (output_root / f"raw_final_answer_attempt_{attempt}.txt").write_text(raw + "\n", encoding="utf-8")  # 冻结最终公开答案且不保存隐藏推理文本。
        (output_root / f"api_usage_attempt_{attempt}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存公开 token 和完成原因。
        delegations, valid = helper.extract_plain_delegations(raw, candidate_ids)  # 使用确定性 parser 提取少量区域与剩余深度。
        if valid:  # 检查模型是否形成合法委派或明确弃权。
            return delegations, raw, audit_rows  # 成功后立即返回避免额外模型调用。
        reasoning_exhausted = finish == "length" and usage["reasoning_tokens"] >= int(0.80 * budget)  # 判断失败是否来自 reasoning-token 截断。
        if not reasoning_exhausted:  # 检查是否已经不是预算不足问题。
            break  # 普通不可解析回答不继续盲目增加 token。
    raise RuntimeError(f"probe-conditioned LLM produced no parseable final answer after {len(audit_rows)} attempts")  # 两级预算均失败时明确标记接口故障。

def execute_from_probe(benchmark: Any, dynamic: dict[str, Any], probe_levels: tuple[int, ...], probe_solution: Any, prediction: np.ndarray, theta: float, target_objective: float) -> dict[str, Any]:  # 从已经购买的一次真实探针状态执行剩余宏动作和至多一次保守纠错。
    levels = probe_levels  # 将一次标准 Dörfler 探针后的网格设为当前在线状态。
    solution = probe_solution  # 将探针真实 FEA 解设为当前物理状态。
    trace = [core.solution_record(benchmark, solution, 1, core.dorfler_mark(benchmark.hotspot_priority(solution, levels), theta))]  # 记录第一层额外物理反馈状态。
    priority = benchmark.hotspot_priority(solution, levels)  # 计算当前探针后真实区域价值用于预算修复。
    macro_levels = core.apply_increment(benchmark, levels, prediction.astype(np.float64), priority)  # 将 LLM 剩余累计深度编译成统一预算内网格。
    if macro_levels != levels:  # 检查模型是否确实委派了额外未来资源。
        levels = macro_levels  # 接受 deterministic budget gate 修复后的宏动作网格。
        solution = benchmark.solve(levels)  # 执行第二次额外真实 FEA 作为宏动作终态审计。
        trace.append(core.solution_record(benchmark, solution, 2, []))  # 记录第二层物理反馈状态。
    corrections = 0  # 初始化宏动作后的额外保守纠错次数。
    if float(benchmark.metrics(solution)[0]) > target_objective:  # 检查 H=2 宏动作是否达到共同动态 Dörfler 终态质量门。
        priority = benchmark.hotspot_priority(solution, levels)  # 从失败后的真实状态重新获取当前可靠局部证据。
        marked = core.dorfler_mark(priority, theta)  # 使用标准 Dörfler 选择一次安全纠错支撑。
        increments = np.zeros(benchmark.region_count, dtype=np.float64)  # 初始化一次一级安全纠错动作。
        increments[marked] = 1.0  # 仅对真实当前热点追加一级资源。
        next_levels = core.apply_increment(benchmark, levels, increments, priority)  # 将安全纠错修复到统一资源上限。
        if next_levels != levels:  # 检查预算内是否仍存在可执行纠错动作。
            levels = next_levels  # 接受安全纠错后的新网格级别。
            solution = benchmark.solve(levels)  # 执行第三次额外真实 FEA 检查系统能否恢复。
            corrections = 1  # 记录本次发生了一次回退式纠错。
            trace.append(core.solution_record(benchmark, solution, 3, marked))  # 保存 H=3 回退后的真实状态。
    return {"additional_H": int(trace[-1]["step"]), "corrections": corrections, "success_vs_dynamic": bool(float(trace[-1]["objective"]) <= target_objective), "final": trace[-1], "trace": trace}  # 返回探针条件化路线的真实反馈深度与终态质量。

def main() -> None:  # 执行长程 Dörfler 参照、一次真实探针、LLM 剩余深度委派和物理审计。
    parser = argparse.ArgumentParser(description="One-probe LLM persistence/depth delegation experiment.")  # 定义命令行实验说明。
    parser.add_argument("--ccx", required=True, help="CalculiX executable path")  # 要求显式传入真实 CalculiX 求解器。
    parser.add_argument("--output", required=True, help="output directory")  # 要求显式指定独立实验结果目录。
    parser.add_argument("--theta", type=float, default=0.50, help="Dörfler bulk parameter")  # 冻结所有路线共享的 Dörfler bulk 参数。
    parser.add_argument("--max-rounds", type=int, default=5, help="dynamic Dörfler horizon")  # 冻结 long-horizon future-hit 参照视界。
    args = parser.parse_args()  # 解析实验参数。
    output_root = Path(args.output).resolve()  # 将结果目录规范化为绝对路径。
    output_root.mkdir(parents=True, exist_ok=True)  # 创建本轮实验结果目录。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()  # 读取受保护真实模型凭据。
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro").strip()  # 读取冻结推理模型名称。
    if not api_key:  # 检查真实模型凭据是否存在。
        raise RuntimeError("DEEPSEEK_API_KEY is required")  # 缺少凭据时禁止以规则模拟 LLM 判断。
    dynamic = core.run_dynamic_dorfler(base_module, output_root / "reference_dynamic", args.ccx, float(args.theta), int(args.max_rounds))  # 执行真实动态 Dörfler 获取共同终态目标与离线 future-hit 标签。
    target_objective = float(dynamic["final"]["objective"]) * 1.02 + 1.0e-12  # 冻结动态终态加百分之二容差作为共同质量门。
    benchmark = core.make_benchmark(base_module, output_root / "online_solver", args.ccx)  # 创建本路线独立缓存的真实 CalculiX 在线求解器。
    coarse_levels = tuple(0 for _ in range(benchmark.region_count))  # 构造所有区域最粗网格的共同起点。
    coarse_solution = benchmark.solve(coarse_levels)  # 执行共同粗网格真实 FEA 形成当前局部误差证据。
    q0 = benchmark.hotspot_priority(coarse_solution, coarse_levels)  # 计算粗网格当前区域价值。
    m0 = core.dorfler_mark(q0, float(args.theta))  # 使用标准 Dörfler 选出第一轮真实热点。
    probe_increment = np.zeros(benchmark.region_count, dtype=np.float64)  # 初始化一次标准一级探针动作。
    probe_increment[m0] = 1.0  # 对第一次真实 Dörfler 支撑统一追加一级网格资源。
    probe_levels = core.apply_increment(benchmark, coarse_levels, probe_increment, q0)  # 将探针动作修复到相同资源预算。
    probe_solution = benchmark.solve(probe_levels)  # 执行第一层额外真实 FEA 获得干预后的局部响应。
    packet, q0_check, q1, m0_check, m1 = make_probe_evidence(dynamic, benchmark, coarse_solution, probe_solution, coarse_levels, probe_levels, float(args.theta))  # 构造探针前后机制持续性证据。
    (output_root / "probe_evidence_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存实际发送给模型的无 future-hit 标签物理证据。
    prompt = build_prompt(packet)  # 编译人工专家式剩余轨迹委派问题。
    (output_root / "public_prompt.txt").write_text(prompt + "\n", encoding="utf-8")  # 保存真实模型公开任务提示供方法复核。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=120.0, max_retries=0)  # 使用 reasoning-aware 较长超时且禁止隐藏自动重试。
    candidate_ids = {int(row["id"]) for row in packet["regions"]}  # 提取确定性 compiler 允许提前配置的区域集合。
    delegations, raw_answer, api_audit = call_llm(client, model, prompt, candidate_ids, output_root)  # 获取利用真实响应证据形成的少量剩余深度委派。
    prediction, confidence = helper.compile_sparse(delegations, benchmark.region_count, candidate_ids)  # 将稀疏机制判断确定性编译成完整区域剩余动作。
    execution = execute_from_probe(benchmark, dynamic, probe_levels, probe_solution, prediction, float(args.theta), target_objective)  # 从已购买探针状态执行剩余宏动作和至多一次安全回退。
    remaining_truth = np.maximum(np.asarray(dynamic["future_hits"], dtype=np.int64) - np.asarray([1 if index in m0 else 0 for index in range(benchmark.region_count)], dtype=np.int64), 0)  # 从完整 future-hit 标签中扣除已经真实执行的第一轮探针以形成离线剩余深度真值。
    depth_eval = core.depth_metrics(prediction, remaining_truth)  # 评价模型剩余累计深度相对离线剩余轨迹真值的误差。
    payload = {"protocol": PROTOCOL_VERSION, "model": model, "api_audit": api_audit, "delegations": delegations, "global_confidence": confidence, "dynamic_H": int(dynamic["additional_H"]), "dynamic_objective": float(dynamic["final"]["objective"]), "target_objective": target_objective, "initial_marked": m0, "marked_after_probe": m1, "future_hits_full": dynamic["future_hits"], "remaining_future_hits_after_probe": remaining_truth.astype(int).tolist(), "prediction_remaining_depth": prediction.astype(int).tolist(), "probe_levels": list(probe_levels), "execution": execution, "depth_metrics": depth_eval, "raw_final_answer": raw_answer}  # 组织探针响应、机制判断、真实执行和离线评价的完整结果。
    (output_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存可用于后续 P-H 校准和多算例聚合的机器结果。
    summary = {"protocol": PROTOCOL_VERSION, "model": model, "api_audit": api_audit, "delegations": delegations, "dynamic_H": int(dynamic["additional_H"]), "probe_route_H": int(execution["additional_H"]), "H_saved": int(dynamic["additional_H"]) - int(execution["additional_H"]), "success": bool(execution["success_vs_dynamic"]), "dynamic_objective": float(dynamic["final"]["objective"]), "probe_route_objective": float(execution["final"]["objective"]), "marked_before": m0, "marked_after_probe": m1, "remaining_future_hits": remaining_truth.astype(int).tolist(), "prediction": prediction.astype(int).tolist(), "depth_mae": float(depth_eval["depth_mae"]), "persistent_f1": float(depth_eval["persistent_f1"])}  # 构造只包含公开委派和数值指标的运行摘要。
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # 输出可直接读取的科学实验结果摘要。
if __name__ == "__main__":  # 检查脚本是否由 GitHub Actions 或命令行直接执行。
    main()  # 启动一次物理探针条件化的 future-hit 资源委派实验。
