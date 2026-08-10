#!/usr/bin/env python3  # 使用当前 Python 解释器构造评估 PH 与实际执行 PH 的差距表
from __future__ import annotations  # 启用前向类型注解以简化函数声明
import argparse  # 解析多个实验目录、输出路径与归一化参数
import csv  # 读取和写出机器可审计的 CSV 结果
import json  # 读取验证方法卡、实验配置和路由记录
import math  # 计算 PH 欧氏差距和稳定数值变换
from collections import defaultdict  # 按方法聚合逐算例 PH 预测和执行结果
from pathlib import Path  # 使用跨平台路径对象访问实验目录
from typing import Any  # 为通用 CSV 和 JSON 字典提供类型注解
import numpy as np  # 执行均值、分位数和确定性 bootstrap 统计
METHOD_FAMILY = {"DM_D": "afem", "LM_D": "afem", "DM_PSO": "pso", "LM_PSO": "pso", "DM_DQN": "rl", "LM_DQN": "rl", "SL_ONE_SHOT": "supervised", "LLM_PH_ROUTER": "llm"}  # 定义论文图使用的方法家族
METHOD_LABEL = {"DM_D": "Dörfler partition + Dörfler", "DM_PSO": "Dörfler partition + PSO", "DM_DQN": "Dörfler partition + DQN", "LM_D": "LLM mechanism partition + Dörfler", "LM_PSO": "LLM mechanism partition + PSO", "LM_DQN": "LLM mechanism partition + DQN", "SL_ONE_SHOT": "One-shot supervised", "LLM_PH_ROUTER": "Dynamic P–H routing LLM"}  # 定义适合图注的完整方法名称
CASE_FIELDS = ("run_seed", "case_id", "mechanism_family", "method", "method_label", "method_family", "route_initial", "assessed_p", "executed_p", "delta_p", "assessed_h_raw", "executed_h_raw", "assessed_h", "executed_h", "delta_h", "gap_l2", "success", "fallback_used", "n_fe", "work_phys")  # 定义逐算例输出列顺序
AGG_FIELDS = ("method", "method_label", "method_family", "n_cases", "assessed_p_mean", "executed_p_mean", "delta_p_mean", "assessed_h_mean", "executed_h_mean", "delta_h_mean", "gap_l2_mean", "assessed_p_ci_low", "assessed_p_ci_high", "executed_p_ci_low", "executed_p_ci_high", "assessed_h_ci_low", "assessed_h_ci_high", "executed_h_ci_low", "executed_h_ci_high", "success_rate", "fallback_rate", "mean_n_fe", "mean_work_phys")  # 定义聚合输出列顺序

def read_csv(path: Path) -> list[dict[str, str]]:  # 读取一个 UTF-8 CSV 文件并返回字典记录
    with path.open("r", encoding="utf-8", newline="") as handle:  # 打开输入文件并保持原始换行处理
        return list(csv.DictReader(handle))  # 将全部行加载为字典列表

def read_json(path: Path) -> Any:  # 读取一个 UTF-8 JSON 文件
    with path.open("r", encoding="utf-8") as handle:  # 打开 JSON 文件
        return json.load(handle)  # 解析并返回 JSON 对象

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:  # 将字典记录按固定列顺序写入 CSV
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录并允许父目录不存在
    with path.open("w", encoding="utf-8", newline="") as handle:  # 打开目标文件并禁止额外空行
        writer = csv.DictWriter(handle, fieldnames=fieldnames)  # 建立固定列顺序的字典写入器
        writer.writeheader()  # 写出 CSV 表头
        for row in rows:  # 遍历全部待写记录
            writer.writerow({name: row.get(name, "") for name in fieldnames})  # 按固定列顺序写出当前记录

def as_bool(value: Any) -> bool:  # 将 CSV 中的布尔字符串稳定转换为 Python 布尔值
    return str(value).strip().lower() in {"true", "1", "yes", "y"}  # 返回常见真值字符串的判断结果

def clamp01(value: float) -> float:  # 将任意浮点数裁剪到零到一范围
    return float(min(max(value, 0.0), 1.0))  # 返回裁剪后的稳定浮点值

def normalize_h(raw_h: float, h_cap: float) -> float:  # 将顺序物理反馈深度映射到论文图零到一纵轴
    if h_cap <= 0.0:  # 检查归一化上限是否非法
        raise ValueError("h_cap must be positive")  # 在上限非法时停止执行
    return clamp01(raw_h / h_cap)  # 使用预注册最大反馈深度完成线性归一化

def load_case_family_map(run_dir: Path) -> dict[str, str]:  # 从算例清单构造 case_id 到机制家族的映射
    manifest_rows = read_csv(run_dir / "case_manifest.csv")  # 读取当前实验目录的完整算例清单
    return {str(row["case_id"]): str(row["family"]) for row in manifest_rows}  # 返回唯一任务编号到机制家族名称的映射

def load_seed(run_dir: Path) -> int:  # 从实验配置读取当前算法随机种子
    config_path = run_dir / "experiment_config.json"  # 定义实验配置文件路径
    if not config_path.exists():  # 检查实验目录是否缺少配置文件
        return 0  # 在缺少配置时使用零作为未知种子占位
    config = read_json(config_path)  # 读取实验配置 JSON
    return int(config.get("seed", 0))  # 返回记录的算法随机种子

def assessed_card(cards: dict[str, Any], family: str, method: str) -> tuple[float, float]:  # 从独立验证方法卡读取执行前 P 与 H 评估
    if family not in cards:  # 检查方法卡是否缺少当前机制家族
        raise KeyError(f"missing family card: {family}")  # 在缺少机制家族卡片时拒绝静默回退
    if method not in cards[family]:  # 检查当前机制家族是否缺少候选路线
        raise KeyError(f"missing method card: {family}/{method}")  # 在缺少候选路线卡片时停止执行
    card = cards[family][method]  # 读取当前机制家族与路线的方法卡
    return float(card["median_p_effective"]), float(card["median_h"])  # 返回验证集预测的 P 与原始 H

def weighted_router_p(router_row: dict[str, str], selected_row: dict[str, str], fallback_row: dict[str, str] | None) -> float:  # 根据实际路线段物理工作重新计算路由执行后的 P
    selected_p = float(selected_row["p_effective"])  # 读取初始路线在当前任务上的实际有效 P
    if not as_bool(router_row.get("fallback_used", False)):  # 检查动态系统是否未触发保守回退
        return clamp01(selected_p)  # 未回退时直接使用初始路线的实际 P
    if fallback_row is None:  # 检查回退结果是否异常缺失
        return clamp01(float(router_row.get("p_effective", selected_p)))  # 在缺少回退记录时使用路由行保存值并保持可执行
    selected_work = max(float(selected_row.get("work_phys", 0.0)), 0.0)  # 读取初始路线实际消耗的物理工作
    fallback_work = max(float(fallback_row.get("work_phys", 0.0)), 0.0)  # 读取保守回退路线实际消耗的物理工作
    total_work = selected_work + fallback_work  # 计算完整执行链的物理工作总量
    if total_work <= 0.0:  # 检查两段路线是否都没有有效工作记录
        selected_h = max(float(selected_row.get("h_feedback", 0.0)), 0.0)  # 使用初始路线反馈深度作为备用权重
        fallback_h = max(float(fallback_row.get("h_feedback", 0.0)), 0.0)  # 使用回退路线反馈深度作为备用权重
        total_h = selected_h + fallback_h  # 计算备用反馈权重总和
        if total_h <= 0.0:  # 检查备用反馈权重是否仍然为零
            return clamp01(selected_p)  # 在没有任何成本证据时保留初始路线实际 P
        return clamp01((selected_p * selected_h + float(fallback_row["p_effective"]) * fallback_h) / total_h)  # 按反馈深度加权两段实际 P
    return clamp01((selected_p * selected_work + float(fallback_row["p_effective"]) * fallback_work) / total_work)  # 按真实物理工作加权两段实际 P

def fixed_result_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:  # 构造固定路线逐算例结果的快速查找表
    return {(str(row["case_id"]), str(row["method"])): row for row in rows}  # 使用任务编号与方法名称作为联合键

def make_case_record(run_seed: int, case_id: str, family: str, method: str, route_initial: str, assessed_p: float, assessed_h_raw: float, executed_p: float, executed_h_raw: float, success: bool, fallback_used: bool, n_fe: float, work_phys: float, h_cap: float) -> dict[str, Any]:  # 构造一个完整逐算例 PH 差距记录
    assessed_h = normalize_h(assessed_h_raw, h_cap)  # 归一化执行前预计顺序反馈深度
    executed_h = normalize_h(executed_h_raw, h_cap)  # 归一化实际执行顺序反馈深度
    delta_p = float(executed_p - assessed_p)  # 计算实际 P 相对评估 P 的有符号偏差
    delta_h = float(executed_h - assessed_h)  # 计算实际 H 相对评估 H 的有符号偏差
    gap_l2 = float(math.sqrt(delta_p * delta_p + delta_h * delta_h))  # 计算 PH 平面中的欧氏差距
    return {"run_seed": run_seed, "case_id": case_id, "mechanism_family": family, "method": method, "method_label": METHOD_LABEL.get(method, method), "method_family": METHOD_FAMILY.get(method, "other"), "route_initial": route_initial, "assessed_p": assessed_p, "executed_p": executed_p, "delta_p": delta_p, "assessed_h_raw": assessed_h_raw, "executed_h_raw": executed_h_raw, "assessed_h": assessed_h, "executed_h": executed_h, "delta_h": delta_h, "gap_l2": gap_l2, "success": success, "fallback_used": fallback_used, "n_fe": n_fe, "work_phys": work_phys}  # 返回统一逐算例 PH 差距记录

def build_run_records(run_dir: Path, h_cap: float) -> list[dict[str, Any]]:  # 从一个实验目录构造固定路线和动态路由的逐算例 PH 差距记录
    run_seed = load_seed(run_dir)  # 读取当前实验随机种子
    family_map = load_case_family_map(run_dir)  # 读取任务到机制家族的映射
    cards = read_json(run_dir / "validation_method_cards.json")  # 读取仅由独立验证集构造的方法卡片
    fixed_rows = read_csv(run_dir / "test_method_results.csv")  # 读取七条固定路线的真实测试执行结果
    fixed_index = fixed_result_index(fixed_rows)  # 构造固定路线结果查找表
    records: list[dict[str, Any]] = []  # 初始化当前实验目录的 PH 差距记录
    for row in fixed_rows:  # 遍历全部固定路线逐算例结果
        case_id = str(row["case_id"])  # 读取当前任务编号
        method = str(row["method"])  # 读取当前固定路线名称
        family = family_map[case_id]  # 查询当前任务所属机制家族
        assessed_p, assessed_h_raw = assessed_card(cards, family, method)  # 从验证方法卡读取执行前 PH 评估
        record = make_case_record(run_seed, case_id, family, method, method, assessed_p, assessed_h_raw, float(row["p_effective"]), float(row["h_feedback"]), as_bool(row["success"]), False, float(row["n_fe"]), float(row["work_phys"]), h_cap)  # 构造固定路线 PH 差距记录
        records.append(record)  # 保存当前固定路线记录
    router_path = run_dir / "router_results.csv"  # 定义动态路由真实执行结果路径
    if router_path.exists():  # 检查当前实验是否包含动态路由结果
        router_rows = read_csv(router_path)  # 读取动态路由逐算例真实执行结果
        for row in router_rows:  # 遍历全部动态路由任务
            case_id = str(row["case_id"])  # 读取当前任务编号
            family = family_map[case_id]  # 查询当前任务所属机制家族
            selected = str(row.get("route_initial", "DM_D"))  # 读取路由器执行前选择的固定路线
            assessed_p, assessed_h_raw = assessed_card(cards, family, selected)  # 使用首选路线方法卡作为执行前 PH 评估
            selected_row = fixed_index[(case_id, selected)]  # 读取初始路线在当前任务上的实际固定执行结果
            fallback_row = fixed_index.get((case_id, "DM_D")) if as_bool(row.get("fallback_used", False)) else None  # 在发生回退时读取保守动态 Dörfler 结果
            executed_p = weighted_router_p(row, selected_row, fallback_row)  # 根据实际执行链重新计算动态系统 P
            record = make_case_record(run_seed, case_id, family, "LLM_PH_ROUTER", selected, assessed_p, assessed_h_raw, executed_p, float(row["h_feedback"]), as_bool(row["success"]), as_bool(row.get("fallback_used", False)), float(row["n_fe"]), float(row["work_phys"]), h_cap)  # 构造动态路由 PH 差距记录
            records.append(record)  # 保存当前动态路由记录
    return records  # 返回当前实验目录全部逐算例记录

def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:  # 使用配对样本的确定性 bootstrap 估计均值置信区间
    clean = values[np.isfinite(values)]  # 删除非有限数值并保留有效观测
    if clean.size == 0:  # 检查当前指标是否没有有效观测
        return float("nan"), float("nan")  # 在无观测时返回空置信区间
    if clean.size == 1 or samples <= 0:  # 检查是否只有一个观测或关闭 bootstrap
        value = float(clean[0])  # 读取唯一有效观测
        return value, value  # 返回退化为单点的置信区间
    indices = rng.integers(0, clean.size, size=(samples, clean.size))  # 生成固定数量的有放回 bootstrap 索引
    means = np.mean(clean[indices], axis=1)  # 计算每个 bootstrap 样本的均值
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))  # 返回百分位法九十五置信区间

def aggregate_records(records: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:  # 按方法聚合 PH 差距、成功率和物理成本
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)  # 初始化方法到逐算例记录的分组
    for row in records:  # 遍历全部逐算例记录
        grouped[str(row["method"])].append(row)  # 将当前记录加入对应方法分组
    rng = np.random.default_rng(seed)  # 创建确定性 NumPy 随机数发生器用于 bootstrap
    output: list[dict[str, Any]] = []  # 初始化聚合输出记录
    for method in sorted(grouped, key=lambda name: (list(METHOD_LABEL).index(name) if name in METHOD_LABEL else 999, name)):  # 按预定义方法顺序遍历各分组
        rows = grouped[method]  # 读取当前方法全部逐算例记录
        assessed_p = np.asarray([float(row["assessed_p"]) for row in rows], dtype=np.float64)  # 提取评估 P 数组
        executed_p = np.asarray([float(row["executed_p"]) for row in rows], dtype=np.float64)  # 提取实际 P 数组
        assessed_h = np.asarray([float(row["assessed_h"]) for row in rows], dtype=np.float64)  # 提取评估 H 数组
        executed_h = np.asarray([float(row["executed_h"]) for row in rows], dtype=np.float64)  # 提取实际 H 数组
        assessed_p_ci = bootstrap_mean_ci(assessed_p, rng, bootstrap_samples)  # 估计评估 P 均值置信区间
        executed_p_ci = bootstrap_mean_ci(executed_p, rng, bootstrap_samples)  # 估计实际 P 均值置信区间
        assessed_h_ci = bootstrap_mean_ci(assessed_h, rng, bootstrap_samples)  # 估计评估 H 均值置信区间
        executed_h_ci = bootstrap_mean_ci(executed_h, rng, bootstrap_samples)  # 估计实际 H 均值置信区间
        output.append({"method": method, "method_label": METHOD_LABEL.get(method, method), "method_family": METHOD_FAMILY.get(method, "other"), "n_cases": len(rows), "assessed_p_mean": float(np.mean(assessed_p)), "executed_p_mean": float(np.mean(executed_p)), "delta_p_mean": float(np.mean(executed_p - assessed_p)), "assessed_h_mean": float(np.mean(assessed_h)), "executed_h_mean": float(np.mean(executed_h)), "delta_h_mean": float(np.mean(executed_h - assessed_h)), "gap_l2_mean": float(np.mean([float(row["gap_l2"]) for row in rows])), "assessed_p_ci_low": assessed_p_ci[0], "assessed_p_ci_high": assessed_p_ci[1], "executed_p_ci_low": executed_p_ci[0], "executed_p_ci_high": executed_p_ci[1], "assessed_h_ci_low": assessed_h_ci[0], "assessed_h_ci_high": assessed_h_ci[1], "executed_h_ci_low": executed_h_ci[0], "executed_h_ci_high": executed_h_ci[1], "success_rate": float(np.mean([1.0 if bool(row["success"]) else 0.0 for row in rows])), "fallback_rate": float(np.mean([1.0 if bool(row["fallback_used"]) else 0.0 for row in rows])), "mean_n_fe": float(np.mean([float(row["n_fe"]) for row in rows])), "mean_work_phys": float(np.mean([float(row["work_phys"]) for row in rows]))})  # 保存当前方法完整聚合统计
    return output  # 返回所有方法聚合记录

def parse_args() -> argparse.Namespace:  # 定义并解析命令行参数
    parser = argparse.ArgumentParser(description="Build assessed-versus-executed PH gap tables from one or more experiment runs.")  # 创建命令行解析器
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="Experiment result directory; repeat for multiple seeds.")  # 允许重复指定多个随机种子实验目录
    parser.add_argument("--case-output", type=Path, required=True, help="Output per-case PH gap CSV.")  # 声明逐算例输出路径
    parser.add_argument("--aggregate-output", type=Path, required=True, help="Output aggregate PH gap CSV.")  # 声明方法级聚合输出路径
    parser.add_argument("--h-cap", type=float, default=6.0, help="Sequential-feedback cap used only for the 0-1 plot axis.")  # 声明图形纵轴归一化上限
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Bootstrap samples for aggregate confidence intervals.")  # 声明 bootstrap 重采样次数
    parser.add_argument("--bootstrap-seed", type=int, default=20260811, help="Deterministic bootstrap seed.")  # 声明确保可复现的 bootstrap 随机种子
    return parser.parse_args()  # 返回解析后的命令行参数

def main() -> None:  # 执行多实验目录 PH 差距构建与聚合流程
    args = parse_args()  # 读取命令行参数
    all_records: list[dict[str, Any]] = []  # 初始化跨随机种子的逐算例记录
    for run_dir in args.run_dir:  # 遍历用户指定的每个实验目录
        all_records.extend(build_run_records(run_dir, args.h_cap))  # 构造并合并当前目录的逐算例 PH 差距记录
    aggregate = aggregate_records(all_records, args.bootstrap_samples, args.bootstrap_seed)  # 聚合跨算例和随机种子的 PH 统计
    write_csv(args.case_output, all_records, CASE_FIELDS)  # 写出逐算例 PH 差距表
    write_csv(args.aggregate_output, aggregate, AGG_FIELDS)  # 写出方法级 PH 差距聚合表
    print(json.dumps({"runs": len(args.run_dir), "case_rows": len(all_records), "methods": len(aggregate), "case_output": str(args.case_output), "aggregate_output": str(args.aggregate_output)}, ensure_ascii=False, indent=2))  # 输出机器可读执行摘要

if __name__ == "__main__":  # 检查脚本是否作为主程序运行
    main()  # 启动 PH 差距构建流程
