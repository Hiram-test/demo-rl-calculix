#!/usr/bin/env python3  # 使用推理预算自适应接口执行真实大模型 future-hit 资源委派实验。
from __future__ import annotations  # 启用现代类型注解并保持 Python 3.11 兼容。
import argparse  # 解析真实 CalculiX 路径、输出目录和有限视界参数。
import json  # 保存证据包、API 元数据、稀疏委派和真实物理结果。
import os  # 从受保护 GitHub Environment 读取 DeepSeek 凭据和模型名称。
import sys  # 将仓库根目录加入 Python 模块搜索路径。
from pathlib import Path  # 跨平台定位仓库与独立结果目录。
from typing import Any  # 为模型返回对象和结构化记录提供通用类型注解。
import numpy as np  # 处理 future-hit 真值、区域动作和数值评价。
from openai import OpenAI  # 通过 OpenAI 兼容接口调用真实 DeepSeek 服务。
ROOT = Path(__file__).resolve().parents[2]  # 从实验脚本目录返回仓库根目录。
PROTOCOL_VERSION = "sparse-mechanism-delegation-v3-reasoning-aware"  # 冻结修复 reasoning token 截断问题后的协议版本。
sys.path.insert(0, str(ROOT))  # 允许稳定导入已有真实 CalculiX 基准与机制委派辅助函数。
from experiments.cross_passage_torsion_benchmark import run_benchmark as base_module  # 导入 PR24 已验证的真实 CalculiX 横向通道实现。
from experiments.future_hit_delegation import run as core  # 导入动态 Dörfler、宏动作执行和深度评价函数。
from experiments.future_hit_delegation import run_live_sparse as helper  # 复用已经验证的证据包、自然语言解析器和确定性 compiler。

def public_usage(completion: Any) -> dict[str, Any]:  # 提取公开 token 计数用于区分推理预算耗尽和普通格式故障。
    usage = completion.usage.model_dump() if completion.usage is not None else {}  # 读取 SDK 提供的公开用量对象并处理缺失情况。
    details = usage.get("completion_tokens_details") or {}  # 提取 completion token 的公开分类统计。
    return {"completion_tokens": int(usage.get("completion_tokens") or 0), "reasoning_tokens": int(details.get("reasoning_tokens") or 0), "prompt_tokens": int(usage.get("prompt_tokens") or 0), "total_tokens": int(usage.get("total_tokens") or 0)}  # 返回不含任何推理文本的安全统计字段。

def build_prompt(packet: dict[str, Any]) -> str:  # 将结构力学上下文与当前 Dörfler 证据编译成简短人工专家式判断任务。
    rows = packet["candidate_regions"]  # 读取当前热点及一跳耦合区域的候选证据。
    table = "\n".join(f"REGION {row['id']}: {row['structural_role']}; marked={row['marked_now']}; q={row['q']}; energy={row['energy']}; peak={row['peak']}; contrast={row['contrast']}; neighbors={row['neighbors']}" for row in rows)  # 将候选区转换为紧凑可读表格。
    context = packet["physics_context"]  # 读取不含 future-hit 标签的结构力学背景。
    return f"You are deciding advance mesh-resource delegation for a linear-elastic six-station 3D space frame. Station 1 is fixed and station 6 is prescribed a small twist. Torsion is carried by coupled chords, transverse frames and X bracing. Current Dörfler marking is reliable present-state evidence and must not be replaced. Current marked regions: {packet['current_marked']}.\n{table}\nInfer which FEW candidate regions are likely to remain important over multiple later Dörfler rounds, using both load-transfer mechanics and the coarse-grid evidence. For each region worth advance investment, give cumulative extra depth 1, 2, or 3. Final answer must contain only lines like REGION 13 DEPTH 3 CONF 0.8. If no advance investment is justified, answer exactly ABSTAIN. Do not output JSON. Do not explain hidden reasoning."  # 明确只要求最终可审计委派摘要而不要求完整数值控制器。

def call_with_reasoning_budget(client: OpenAI, model: str, prompt: str, candidate_ids: set[int], output_root: Path) -> tuple[list[dict[str, Any]], str, int, bool, list[dict[str, Any]]]:  # 在推理模型耗尽 token 时自动增加最终回答预算。
    token_budgets = [4096, 8192]  # 第一轮提供足够推理空间，只有被长度截断时才使用更高上限重试。
    audit_rows: list[dict[str, Any]] = []  # 初始化公开 API 运行统计记录。
    raw_responses: list[str] = []  # 初始化最终公开模型答案记录并禁止读取隐藏推理正文。
    for attempt_index, token_budget in enumerate(token_budgets, start=1):  # 按预注册预算依次执行最多两次真实模型调用。
        completion = client.chat.completions.create(model=model, messages=[{"role": "system", "content": "Act as a finite-element adaptation expert. Return only the concise final resource-delegation answer requested by the user prompt."}, {"role": "user", "content": prompt}], temperature=0.0, max_tokens=token_budget)  # 调用真实推理模型并为最终答案保留充分 completion 预算。
        raw_text = completion.choices[0].message.content or ""  # 只读取模型最终公开 answer 字段并忽略任何内部推理正文。
        usage = public_usage(completion)  # 读取 reasoning-token 数量用于诊断是否发生预算截断。
        finish_reason = str(completion.choices[0].finish_reason)  # 记录模型公开结束原因以区分正常完成和长度截断。
        audit = {"attempt": attempt_index, "max_tokens": token_budget, "finish_reason": finish_reason, "content_length": len(raw_text), **usage}  # 组织本次调用的公开接口诊断字段。
        audit_rows.append(audit)  # 保存本次公开 token 和完成状态供结果审计。
        raw_responses.append(raw_text)  # 保存本次最终公开回答供确定性 parser 和后续复核。
        (output_root / f"raw_final_answer_attempt_{attempt_index}.txt").write_text(raw_text + "\n", encoding="utf-8")  # 冻结本次最终公开答案且不保存隐藏推理文本。
        (output_root / f"api_usage_attempt_{attempt_index}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存公开 token 统计和结束原因。
        delegations, valid = helper.extract_plain_delegations(raw_text, candidate_ids)  # 使用确定性规则从最终公开答案提取少量 region/depth 委派。
        if valid:  # 检查模型是否给出至少一个合法委派或明确 ABSTAIN。
            return delegations, raw_text, attempt_index, True, audit_rows  # 成功后立即返回并避免浪费第二次真实模型调用。
        reasoning_exhausted = finish_reason == "length" and usage["reasoning_tokens"] >= int(0.80 * token_budget)  # 判断 completion 预算是否主要被推理 token 耗尽。
        if not reasoning_exhausted:  # 检查当前失败是否已经不是 reasoning-token 截断问题。
            break  # 对普通空答或不可解析终答不盲目增加模型预算。
    return [], "\n---ATTEMPT---\n".join(raw_responses), len(audit_rows), False, audit_rows  # 所有允许尝试均失败时明确返回接口无效状态。

def main() -> None:  # 执行真实动态 Dörfler、推理预算自适应机制委派、确定性编译和 CalculiX 审计。
    parser = argparse.ArgumentParser(description="Reasoning-aware live LLM future-hit delegation experiment.")  # 定义命令行实验说明。
    parser.add_argument("--ccx", required=True, help="CalculiX executable path")  # 要求显式传入真实 CalculiX 求解器路径。
    parser.add_argument("--output", required=True, help="output directory")  # 要求显式指定本轮独立结果目录。
    parser.add_argument("--theta", type=float, default=0.50, help="Dörfler bulk parameter")  # 冻结当前可靠热点证据的 Dörfler 参数。
    parser.add_argument("--max-rounds", type=int, default=5, help="dynamic Dörfler horizon")  # 冻结形成 future-hit 真值的经典有限视界。
    args = parser.parse_args()  # 解析真实实验参数。
    output_root = Path(args.output).resolve()  # 将输出路径规范化为绝对目录。
    output_root.mkdir(parents=True, exist_ok=True)  # 创建本轮真实模型与求解器证据目录。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()  # 从受保护 Environment 读取真实 API 凭据。
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro").strip()  # 读取冻结真实模型名称。
    if not api_key:  # 检查受保护凭据是否存在。
        raise RuntimeError("DEEPSEEK_API_KEY is required")  # 缺少真实模型凭据时禁止使用任何模拟输出替代。
    dynamic = core.run_dynamic_dorfler(base_module, output_root / "dynamic_solver", args.ccx, float(args.theta), int(args.max_rounds))  # 执行真实动态 Dörfler 并记录未来重复命中真值。
    dynamic_h = int(dynamic["additional_H"])  # 读取动态 Dörfler 相对共同粗解的顺序反馈深度。
    target_objective = float(dynamic["final"]["objective"]) * 1.02 + 1.0e-12  # 以动态终态加百分之二容差冻结共同质量门。
    evidence_benchmark = core.make_benchmark(base_module, output_root / "evidence_solver", args.ccx)  # 创建缓存隔离的粗网格证据求解器。
    coarse_levels = tuple(0 for _ in range(evidence_benchmark.region_count))  # 构造所有方法共享的最粗区域级网格。
    coarse_solution = evidence_benchmark.solve(coarse_levels)  # 执行一次真实 CalculiX 获得当前物理证据。
    packet = helper.build_compact_packet(dynamic, evidence_benchmark, coarse_solution)  # 复用无 oracle 泄漏的理论结构与粗网格证据合同。
    packet["protocol"] = PROTOCOL_VERSION  # 将证据包版本更新为 reasoning-aware 实验协议。
    (output_root / "compact_evidence_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存实际发送给模型的无标签证据包。
    prompt = build_prompt(packet)  # 将结构化证据编译成人工专家式持续热点判断问题。
    (output_root / "public_prompt.txt").write_text(prompt + "\n", encoding="utf-8")  # 保存模型实际收到的公开最终任务提示以便复现。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=120.0, max_retries=0)  # 为推理模型提供更长单次响应时间并禁止隐藏自动重试。
    candidate_ids = {int(row["id"]) for row in packet["candidate_regions"]}  # 提取确定性 compiler 允许提前配置的候选区域集合。
    delegations, raw_text, attempt_count, interface_valid, api_audit = call_with_reasoning_budget(client, model, prompt, candidate_ids, output_root)  # 获取具有充分推理预算的最终公开资源委派答案。
    if not interface_valid:  # 检查模型在 reasoning-aware 预算下是否仍无可执行最终答案。
        failure = {"protocol": PROTOCOL_VERSION, "model": model, "interface_valid": False, "attempt_count": attempt_count, "api_audit": api_audit, "dynamic_H": dynamic_h, "dynamic_objective": dynamic["final"]["objective"], "future_hits": dynamic["future_hits"], "initial_marked": dynamic["initial_marked"], "raw_final_answer": raw_text}  # 构造严格区分接口故障和科学失败的诊断对象。
        (output_root / "interface_failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 冻结公开接口故障证据用于后续模型客户端诊断。
        raise RuntimeError("reasoning-aware live LLM produced no parseable final delegation")  # 直接失败并禁止把接口空答当成全零算法决策。
    prediction, confidence = helper.compile_sparse(delegations, evidence_benchmark.region_count, candidate_ids)  # 将模型少量委派确定性编译成完整十六维网格动作。
    result = core.run_macro_method(base_module, output_root / "llm_solver", args.ccx, dynamic, prediction, "llm_sparse_mechanism_delegate_v3", target_objective, float(args.theta), 2)  # 执行一次资源重投并在失败时最多购买两轮真实 Dörfler 纠错。
    result.update(core.depth_metrics(prediction, np.asarray(dynamic["future_hits"], dtype=np.int64)))  # 计算累计深度 MAE 和持续热点 F1 以直接评价机制委派质量。
    payload = {"protocol": PROTOCOL_VERSION, "model": model, "interface_valid": True, "attempt_count": attempt_count, "api_audit": api_audit, "delegations": delegations, "dynamic_H": dynamic_h, "dynamic_objective": dynamic["final"]["objective"], "target_objective": target_objective, "initial_marked": dynamic["initial_marked"], "future_hits": dynamic["future_hits"], "prediction": prediction.astype(int).tolist(), "global_confidence": confidence, "raw_final_answer": raw_text, "llm_result": result}  # 组织接口、机制委派、资源动作和真实物理审计的完整机器结果。
    (output_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存可直接进入后续 P-H 校准的数据对象。
    summary = {"protocol": PROTOCOL_VERSION, "model": model, "api_audit": api_audit, "delegations": delegations, "dynamic_H": dynamic_h, "llm_H": int(result["additional_H"]), "H_saved": dynamic_h - int(result["additional_H"]), "success": bool(result["success_vs_dynamic"]), "dynamic_objective": float(dynamic["final"]["objective"]), "llm_objective": float(result["final"]["objective"]), "depth_mae": float(result["depth_mae"]), "persistent_f1": float(result["persistent_f1"]), "future_hits": dynamic["future_hits"], "prediction": prediction.astype(int).tolist()}  # 构造只含公开决策与数值结果的 Actions 摘要。
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # 输出不包含隐藏推理正文的科学结果摘要。
if __name__ == "__main__":  # 检查脚本是否由 GitHub Actions 或命令行直接执行。
    main()  # 启动修复 reasoning token 截断后的真实机制委派实验。
