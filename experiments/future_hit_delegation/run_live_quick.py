#!/usr/bin/env python3  # 使用一次真实大模型调用快速测试 future-hit 资源重投而避免长时间 API 阻塞。
from __future__ import annotations  # 启用现代类型注解并保持 Python 3.11 兼容。
import argparse  # 解析真实 CalculiX 路径和独立结果目录。
import json  # 保存模型证据包、原始响应和真实执行结果。
import os  # 从受保护 GitHub Environment 读取 DeepSeek 凭据和模型名。
import sys  # 将仓库根目录加入 Python 模块搜索路径。
from pathlib import Path  # 跨平台定位仓库和实验输出目录。
import numpy as np  # 处理 future-hit 深度和区域级资源重投向量。
from openai import OpenAI  # 通过 OpenAI 兼容客户端调用真实 DeepSeek API。
ROOT = Path(__file__).resolve().parents[2]  # 从当前实验目录返回仓库根目录。
sys.path.insert(0, str(ROOT))  # 允许稳定导入已有横向通道真实 CalculiX 基准和 future-hit 工具函数。
from experiments.cross_passage_torsion_benchmark import run_benchmark as base_module  # 导入 PR24 已验证的真实 CalculiX 横向通道实现。
from experiments.future_hit_delegation import run as core  # 导入 Dörfler、证据包、宏动作与物理审计函数。

def parse_json_response(text: str) -> dict:  # 从模型可能带 Markdown 围栏的响应中提取单一 JSON 对象。
    cleaned = text.strip()  # 去除模型响应首尾空白字符。
    if cleaned.startswith("```json"):  # 检查模型是否使用 JSON Markdown 围栏。
        cleaned = cleaned[len("```json"):].strip()  # 移除开头 JSON 围栏标记。
    elif cleaned.startswith("```"):  # 检查模型是否使用普通 Markdown 围栏。
        cleaned = cleaned[len("```"):].strip()  # 移除普通开头围栏标记。
    if cleaned.endswith("```"):  # 检查模型响应尾部是否仍有 Markdown 围栏。
        cleaned = cleaned[:-3].strip()  # 移除尾部围栏并保留纯 JSON 文本。
    return json.loads(cleaned)  # 将清理后的文本解析为结构化模型动作。

def main() -> None:  # 执行动态基线、一次真实大模型预测、宏动作和最多两轮 Dörfler 纠错。
    parser = argparse.ArgumentParser(description="Quick live LLM future-hit delegation test with a hard API timeout.")  # 定义快速真实模型实验命令行说明。
    parser.add_argument("--ccx", required=True, help="CalculiX executable path")  # 要求显式提供真实 CalculiX 求解器路径。
    parser.add_argument("--output", required=True, help="output directory")  # 要求指定与其他实验隔离的结果目录。
    parser.add_argument("--theta", type=float, default=0.50, help="Dörfler bulk parameter")  # 冻结共同当前热点证据的 Dörfler 阈值。
    parser.add_argument("--max-rounds", type=int, default=5, help="dynamic Dörfler horizon")  # 冻结形成 future-hit 真值和终态目标的经典视界。
    args = parser.parse_args()  # 解析本次真实实验参数。
    output_root = Path(args.output).resolve()  # 将结果路径规范化为绝对路径。
    output_root.mkdir(parents=True, exist_ok=True)  # 创建独立快速真实模型结果目录。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()  # 从受保护 Environment 读取真实模型凭据。
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro").strip()  # 读取冻结真实模型名称。
    if not api_key:  # 检查真实模型凭据是否成功注入。
        raise RuntimeError("DEEPSEEK_API_KEY is required")  # 缺少凭据时拒绝使用任何伪预测替代真实实验。
    dynamic = core.run_dynamic_dorfler(base_module, output_root / "dynamic_solver", args.ccx, float(args.theta), int(args.max_rounds))  # 执行经典逐轮 Dörfler 并获得未来重复命中离线真值。
    dynamic_h = int(dynamic["additional_H"])  # 读取经典方法相对共同粗解的真实反馈深度。
    target_objective = float(dynamic["final"]["objective"]) * 1.02 + 1.0e-12  # 将经典终态误差加百分之二容差作为统一成功门。
    evidence_benchmark = core.make_benchmark(base_module, output_root / "evidence_solver", args.ccx)  # 创建与动态基线缓存隔离的共同粗解证据求解器。
    coarse_levels = tuple(0 for _ in range(evidence_benchmark.region_count))  # 构造所有区域最粗离散的共同在线起点。
    coarse_solution = evidence_benchmark.solve(coarse_levels)  # 执行一次真实 CalculiX 得到大模型可见的当前物理证据。
    packet = core.build_llm_packet(dynamic, evidence_benchmark, coarse_solution)  # 构造严格不包含 future-hit 标签的机制资源重投证据包。
    packet_text = json.dumps(packet, ensure_ascii=False, sort_keys=True)  # 将实际在线证据冻结为稳定 JSON 文本。
    (output_root / "evidence_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存真实发送证据包供标签隔离审计。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=45.0, max_retries=0)  # 建立带四十五秒硬超时且禁止自动重试的真实 DeepSeek 客户端。
    system_text = "You are a finite-element adaptation planner. The current Dörfler support is authoritative current evidence. Do not redraw current hotspots. Predict only cumulative future refinement depth for all 16 regions so repeated future one-level refinements can be delegated now. Return one JSON object with exactly the required fields and short auditable reasons."  # 将大模型能力严格限制在未来持续性和跨轮次资源重投。
    user_text = "Use the supplied coarse-solve and Dörfler evidence to predict persistent future refinement demand. Return exactly 16 region objects and no prose outside JSON. Evidence packet:\n" + packet_text  # 发送完整结构化当前证据但绝不发送动态 future-hit 真值。
    completion = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_text}, {"role": "user", "content": user_text}], temperature=0.0, max_tokens=3000)  # 执行一次真实模型调用并设置明确输出上限。
    raw_text = completion.choices[0].message.content or "{}"  # 读取模型实际输出并对空响应使用可审计空对象。
    (output_root / "raw_llm_response.txt").write_text(raw_text + "\n", encoding="utf-8")  # 保存原始模型响应以便检查机制证据和格式错误。
    parsed = parse_json_response(raw_text)  # 将真实模型输出转换为结构化区域资源配置。
    prediction, global_confidence, audit = core.validate_llm_prediction(parsed, evidence_benchmark.region_count)  # 通过硬限制门得到零到三级整数资源重投深度。
    result = core.run_macro_method(base_module, output_root / "llm_solver", args.ccx, dynamic, prediction, "llm_future_hit_quick", target_objective, float(args.theta), 2)  # 一次执行模型重投并在真实验收失败时最多购买两轮 Dörfler 纠错。
    result.update(core.depth_metrics(prediction, np.asarray(dynamic["future_hits"], dtype=np.int64)))  # 计算模型预测相对长程 Dörfler 实际 future-hit 深度的误差和持续热点 F1。
    result["global_confidence"] = global_confidence  # 保存模型自身机制置信度用于后续 PH 校准。
    result["audit_recommendation"] = audit  # 保存模型主动请求的物理审计深度。
    payload = {"model": model, "dynamic_H": dynamic_h, "dynamic_objective": dynamic["final"]["objective"], "target_objective": target_objective, "future_hits": dynamic["future_hits"], "initial_marked": dynamic["initial_marked"], "prediction": prediction.astype(int).tolist(), "llm_result": result}  # 组织足以检验轮次压缩和机制预测质量的完整结果。
    (output_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存机器可读真实模型实验结果。
    print(json.dumps({"model": model, "dynamic_H": dynamic_h, "dynamic_objective": dynamic["final"]["objective"], "llm_H": result["additional_H"], "H_saved": dynamic_h - int(result["additional_H"]), "success": result["success_vs_dynamic"], "llm_objective": result["final"]["objective"], "depth_mae": result["depth_mae"], "persistent_f1": result["persistent_f1"], "global_confidence": global_confidence, "future_hits": dynamic["future_hits"], "prediction": prediction.astype(int).tolist()}, ensure_ascii=False, indent=2))  # 在 Actions 日志直接输出核心轮次、终态质量和 future-hit 预测指标。
if __name__ == "__main__":  # 仅在脚本被 Actions 直接运行时启动真实实验。
    main()  # 执行单次真实大模型 future-hit 资源重投试验。
