#!/usr/bin/env python3  # 使用真实 CalculiX 快速验证 future-hit 资源重投是否存在可压缩上界。
from __future__ import annotations  # 启用现代类型注解并保持 Python 3.11 兼容。
import argparse  # 解析真实 CalculiX 路径与独立结果目录。
import csv  # 写出经典、oracle 和当前强度方法的核心反馈效率表。
import json  # 保存完整机器可读数值轨迹和区域 future-hit 真值。
import sys  # 将仓库根目录加入 Python 模块搜索路径。
from pathlib import Path  # 跨平台定位仓库与输出目录。
import numpy as np  # 处理区域未来命中和资源重投深度向量。
ROOT = Path(__file__).resolve().parents[2]  # 从当前实验目录返回仓库根目录。
sys.path.insert(0, str(ROOT))  # 允许稳定导入已有真实 CalculiX 横向通道基准和实验函数。
from experiments.cross_passage_torsion_benchmark import run_benchmark as base_module  # 导入 PR24 已验证的真实 CalculiX 基准。
from experiments.future_hit_delegation import run as core  # 导入当前 future-hit 标记、重投和审计实现。

def main() -> None:  # 执行经典长程轨迹与两个一次性资源重投对照。
    parser = argparse.ArgumentParser(description="Numerical upper-bound test for future-hit refinement delegation.")  # 定义快速实验命令行说明。
    parser.add_argument("--ccx", required=True, help="CalculiX executable path")  # 要求显式提供真实 CalculiX 可执行文件路径。
    parser.add_argument("--output", required=True, help="output directory")  # 要求显式指定与 live LLM 结果隔离的输出目录。
    parser.add_argument("--theta", type=float, default=0.50, help="Dörfler bulk parameter")  # 冻结标准 Dörfler 覆盖比例。
    parser.add_argument("--max-rounds", type=int, default=5, help="long-horizon dynamic Dörfler rounds")  # 冻结用于形成 future-hit 真值的经典反馈视界。
    parser.add_argument("--max-corrections", type=int, default=2, help="post-macro correction rounds")  # 限制宏动作失败后最多购买两次真实反馈。
    args = parser.parse_args()  # 解析本次真实数值实验参数。
    output_root = Path(args.output).resolve()  # 将独立结果目录规范化为绝对路径。
    output_root.mkdir(parents=True, exist_ok=True)  # 创建本次数值上界实验输出目录。
    dynamic = core.run_dynamic_dorfler(base_module, output_root / "dynamic_solver", args.ccx, float(args.theta), int(args.max_rounds))  # 执行逐轮真实 Dörfler 并形成未来重复命中真值。
    dynamic_h = int(dynamic["additional_H"])  # 读取经典路线相对共同粗解的真实反馈深度。
    target_objective = float(dynamic["final"]["objective"]) * 1.02 + 1.0e-12  # 将经典有限视界终态误差加百分之二容差作为共同成功标准。
    future_hits = np.minimum(np.asarray(dynamic["future_hits"], dtype=np.int64), 3)  # 将长程实际命中次数映射到系统允许的最大三级一次性重投动作。
    rank_prediction = core.current_rank_prediction(np.asarray(dynamic["initial_priority"], dtype=np.float64), [int(value) for value in dynamic["initial_marked"]], len(dynamic["region_names"]))  # 构造只看当前指标大小的无机制重投对照。
    oracle = core.run_macro_method(base_module, output_root / "oracle_solver", args.ccx, dynamic, future_hits, core.METHOD_ORACLE, target_objective, float(args.theta), int(args.max_corrections))  # 使用真实未来命中深度测试理论可压缩上界。
    rank = core.run_macro_method(base_module, output_root / "rank_solver", args.ccx, dynamic, rank_prediction, core.METHOD_RANK, target_objective, float(args.theta), int(args.max_corrections))  # 使用当前指标排序深度测试无机制先验重投效果。
    oracle.update(core.depth_metrics(future_hits, np.asarray(dynamic["future_hits"], dtype=np.int64)))  # 计算 oracle 的区域累计深度和持续热点指标。
    rank.update(core.depth_metrics(rank_prediction, np.asarray(dynamic["future_hits"], dtype=np.int64)))  # 计算当前强度规则的 future-hit 预测误差。
    rows = [dynamic, oracle, rank]  # 按经典反馈、理论上界和当前强度对照顺序组织结果。
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:  # 打开快速实验核心结果表。
        writer = csv.writer(handle)  # 创建标准 CSV 写入器。
        writer.writerow(["method", "additional_H", "H_saved", "success_vs_dynamic", "final_objective", "final_elements", "depth_mae", "persistent_f1"])  # 固定核心反馈压缩、最终质量和未来命中预测列。
        for item in rows:  # 遍历三条数值路线写出配对结果。
            writer.writerow([item["method"], int(item["additional_H"]), int(dynamic_h - int(item["additional_H"])), int(item.get("success_vs_dynamic", True)), float(item["final"]["objective"]), int(item["final"]["element_count"]), item.get("depth_mae", ""), item.get("persistent_f1", "")])  # 写出当前路线相对经典方法实际节省的真实 FEA 轮次。
    payload = {"dynamic": dynamic, "oracle": oracle, "rank": rank, "target_objective": target_objective, "future_hits_clipped": future_hits.astype(int).tolist(), "rank_prediction": rank_prediction.astype(int).tolist()}  # 组织完整轨迹、真值和两种重投动作以支持后续论文审计。
    (output_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存完整机器可读实验结果。
    print(json.dumps({"dynamic_H": dynamic_h, "dynamic_objective": dynamic["final"]["objective"], "oracle_H": oracle["additional_H"], "oracle_saved_H": dynamic_h - int(oracle["additional_H"]), "oracle_success": oracle["success_vs_dynamic"], "oracle_objective": oracle["final"]["objective"], "rank_H": rank["additional_H"], "rank_saved_H": dynamic_h - int(rank["additional_H"]), "rank_success": rank["success_vs_dynamic"], "rank_objective": rank["final"]["objective"], "future_hits": future_hits.astype(int).tolist(), "rank_prediction": rank_prediction.astype(int).tolist()}, ensure_ascii=False, indent=2))  # 在 Actions 日志直接打印足以判断“未来命中能否压缩轮次”的核心结果。
if __name__ == "__main__":  # 仅在脚本作为主程序运行时启动真实数值实验。
    main()  # 执行经典与宏动作上界比较。
