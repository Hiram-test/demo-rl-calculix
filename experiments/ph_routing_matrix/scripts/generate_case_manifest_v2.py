#!/usr/bin/env python3  # 使用当前 Python 解释器生成二维主实验算例清单
from __future__ import annotations  # 启用前向类型注解以保持类型声明简洁
import argparse  # 解析输出路径与随机种子参数
import csv  # 写出可由 GitHub Actions 和求解器读取的 CSV 清单
import hashlib  # 为每个任务生成稳定且不可碰撞的合同哈希
import json  # 将机制元数据稳定序列化后参与哈希
from pathlib import Path  # 使用跨平台路径对象管理输出文件
from typing import Any  # 为通用机制元数据字典提供类型注解
MECHANISMS: list[dict[str, Any]] = [  # 定义十个机制家族及其理论和实验属性
    {"family": "smooth", "theory_card": "smooth_regular_solution", "theory_level": "strong", "decoy": False, "new_mechanism": False},  # 定义平滑场基线机制
    {"family": "hole", "theory_card": "kirsch_finite_scale_concentration", "theory_level": "strong", "decoy": False, "new_mechanism": False},  # 定义圆孔或椭圆孔有限尺度应力集中机制
    {"family": "rounded_notch", "theory_card": "smooth_notch_gradient_scaling", "theory_level": "partial", "decoy": False, "new_mechanism": False},  # 定义圆角或光滑缺口高梯度机制
    {"family": "reentrant_corner", "theory_card": "williams_corner_singularity", "theory_level": "strong", "decoy": False, "new_mechanism": False},  # 定义重入角持续奇异机制
    {"family": "material_interface", "theory_card": "interface_jump_and_flux_balance", "theory_level": "strong", "decoy": False, "new_mechanism": False},  # 定义材料界面与场量跳跃机制
    {"family": "localized_boundary_layer", "theory_card": "local_load_decay_and_boundary_layer", "theory_level": "partial", "decoy": False, "new_mechanism": False},  # 定义局部载荷或边界层机制
    {"family": "multi_hotspot", "theory_card": "finite_budget_hotspot_competition", "theory_level": "partial", "decoy": False, "new_mechanism": False},  # 定义多热点预算竞争机制
    {"family": "decoy_qoi", "theory_card": "qoi_relevance_and_saint_venant_filtering", "theory_level": "partial", "decoy": True, "new_mechanism": False},  # 定义视觉显著但任务无关的诱饵机制
    {"family": "anisotropic", "theory_card": "directional_regularization_and_anisotropic_error", "theory_level": "partial", "decoy": False, "new_mechanism": True},  # 定义训练阶段完全留出的方向性新机制
    {"family": "mixed_shift", "theory_card": "compositional_mechanism_graph", "theory_level": "compositional", "decoy": True, "new_mechanism": True},  # 定义训练阶段完全留出的复合机制偏移
]  # 结束十个机制家族定义
LOAD_CASES = ("load_A", "load_B")  # 定义两个独立载荷状态以激发不同热点
QOI_CASES = ("global_energy", "local_response")  # 定义全局与局部两类任务关注量
BUDGETS = (("tight", 0.78), ("medium", 1.00), ("wide", 1.28))  # 定义相对于模板基准网格的三档资源预算倍率
GEOMETRY_INSTANCES = tuple(range(6))  # 定义每个机制家族六个几何参数实例
FIELDNAMES = ("case_id", "task_hash", "family", "geometry_instance", "load_case", "qoi", "budget_level", "budget_factor", "split", "shift_type", "theory_card", "theory_level", "coarse_evidence_required", "decoy_geometry", "seed")  # 定义固定输出列顺序

def split_for_case(family: str, instance: int) -> tuple[str, str]:  # 根据机制家族与几何实例给出无泄漏数据拆分
    if family == "anisotropic":  # 检查是否为完全留出的方向性新机制
        return "test", "new_mechanism"  # 将全部方向性算例放入新机制测试集
    if family == "mixed_shift":  # 检查是否为完全留出的复合机制
        return "test", "compositional_ood"  # 将全部复合机制算例放入组合 OOD 测试集
    if instance <= 2:  # 检查是否属于前三个参数实例
        return "train", "in_distribution"  # 将前三个参数实例用于训练数据生成
    if instance == 3:  # 检查是否为第四个参数实例
        return "validation", "parameter_validation"  # 将第四个参数实例用于独立方法卡和 PH 校准
    return "test", "parameter_extrapolation"  # 将最后两个参数实例用于已见机制参数外推测试

def stable_task_hash(payload: dict[str, Any]) -> str:  # 根据任务公开字段生成稳定 SHA256 哈希
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")  # 使用稳定字段排序和紧凑 JSON 编码任务
    return hashlib.sha256(encoded).hexdigest()  # 返回完整十六进制任务哈希

def build_rows(base_seed: int) -> list[dict[str, Any]]:  # 构造全部七百二十个任务记录
    rows: list[dict[str, Any]] = []  # 初始化任务记录列表
    ordinal = 0  # 初始化用于派生算例随机种子的全局序号
    for mechanism in MECHANISMS:  # 遍历十个机制家族
        family = str(mechanism["family"])  # 读取当前机制家族名称
        for instance in GEOMETRY_INSTANCES:  # 遍历六个几何参数实例
            split, shift_type = split_for_case(family, instance)  # 计算当前实例的数据拆分与偏移类型
            for load_case in LOAD_CASES:  # 遍历两个载荷状态
                for qoi in QOI_CASES:  # 遍历两个任务关注量
                    for budget_level, budget_factor in BUDGETS:  # 遍历三档资源预算
                        ordinal += 1  # 为当前任务增加全局序号
                        case_id = f"{family}-g{instance:02d}-{load_case}-{qoi}-{budget_level}"  # 构造可读且唯一的任务编号
                        seed = int(base_seed + ordinal * 1009)  # 使用大质数步长生成稳定任务随机种子
                        payload = {"case_id": case_id, "family": family, "geometry_instance": instance, "load_case": load_case, "qoi": qoi, "budget_level": budget_level, "budget_factor": budget_factor, "split": split, "shift_type": shift_type, "seed": seed}  # 汇总参与哈希的任务合同字段
                        row = {"case_id": case_id, "task_hash": stable_task_hash(payload), "family": family, "geometry_instance": instance, "load_case": load_case, "qoi": qoi, "budget_level": budget_level, "budget_factor": budget_factor, "split": split, "shift_type": shift_type, "theory_card": mechanism["theory_card"], "theory_level": mechanism["theory_level"], "coarse_evidence_required": True, "decoy_geometry": mechanism["decoy"], "seed": seed}  # 构造完整输出记录
                        rows.append(row)  # 将当前任务加入主清单
    return rows  # 返回全部任务记录

def validate_rows(rows: list[dict[str, Any]]) -> None:  # 对算例数量、唯一性和拆分比例执行硬验证
    if len(rows) != 720:  # 检查主实验是否严格包含七百二十个任务
        raise RuntimeError(f"expected 720 cases, received {len(rows)}")  # 在任务数量不符时拒绝写出清单
    case_ids = [str(row["case_id"]) for row in rows]  # 提取全部任务编号
    if len(case_ids) != len(set(case_ids)):  # 检查任务编号是否存在重复
        raise RuntimeError("case_id collision detected")  # 在发现重复任务编号时停止生成
    task_hashes = [str(row["task_hash"]) for row in rows]  # 提取全部任务哈希
    if len(task_hashes) != len(set(task_hashes)):  # 检查任务合同哈希是否存在碰撞
        raise RuntimeError("task_hash collision detected")  # 在发现哈希碰撞时停止生成
    split_counts = {name: sum(1 for row in rows if row["split"] == name) for name in ("train", "validation", "test")}  # 统计训练、验证和测试任务数量
    expected_counts = {"train": 288, "validation": 96, "test": 336}  # 固定无泄漏拆分的预期数量
    if split_counts != expected_counts:  # 检查实际拆分数量是否符合预注册协议
        raise RuntimeError(f"unexpected split counts: {split_counts}")  # 在拆分不符时拒绝继续

def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:  # 将任务清单写入 UTF-8 CSV 文件
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录并允许父目录缺失
    with path.open("w", encoding="utf-8", newline="") as handle:  # 打开目标 CSV 并禁止额外空行
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)  # 按固定列顺序建立字典写入器
        writer.writeheader()  # 写出 CSV 表头
        for row in rows:  # 遍历全部任务记录
            writer.writerow(row)  # 写出当前任务记录

def parse_args() -> argparse.Namespace:  # 定义并解析命令行参数
    parser = argparse.ArgumentParser(description="Generate the 720-case PH-routing experiment manifest.")  # 创建命令行解析器
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV path.")  # 声明必需的输出 CSV 路径
    parser.add_argument("--base-seed", type=int, default=20260811, help="Base deterministic seed.")  # 声明可覆盖的基础随机种子
    return parser.parse_args()  # 返回解析后的命令行参数

def main() -> None:  # 执行算例生成、验证与写出流程
    args = parse_args()  # 读取命令行参数
    rows = build_rows(args.base_seed)  # 构造全部参数化任务
    validate_rows(rows)  # 执行数量、唯一性与拆分硬验证
    write_rows(args.output, rows)  # 写出经过验证的任务清单
    print(json.dumps({"output": str(args.output), "cases": len(rows), "train": 288, "validation": 96, "test": 336}, ensure_ascii=False, indent=2))  # 输出机器可读生成摘要

if __name__ == "__main__":  # 检查脚本是否作为主程序执行
    main()  # 启动算例清单生成流程
