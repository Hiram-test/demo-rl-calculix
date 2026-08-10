#!/usr/bin/env python3  # 使用当前 Python 解释器绘制评估 PH 到实际执行 PH 的论文图
from __future__ import annotations  # 启用前向类型注解以简化函数声明
import argparse  # 解析输入聚合 CSV、输出路径和图形选项
import csv  # 读取方法级 PH 差距聚合表
import math  # 计算点大小和标签偏移所需的稳定变换
from pathlib import Path  # 使用跨平台路径对象管理输入和输出文件
import matplotlib.pyplot as plt  # 使用 Matplotlib 生成可复现的脚本图而非生成式图片
from matplotlib.lines import Line2D  # 构造方法家族与评估/执行标记图例
import numpy as np  # 计算描述性 P–H 趋势和数值数组
FAMILY_COLORS = {"afem": "#8c8c8c", "pso": "#2ca02c", "rl": "#d62728", "supervised": "#17becf", "surrogate": "#1f77b4", "human": "#ff7f0e", "llm": "#7b4ab5", "other": "#4d4d4d"}  # 定义与示意图一致但可复现的方法家族颜色
FAMILY_LABELS = {"afem": "AFEM / Dörfler", "pso": "PSO / search", "rl": "RL / DQN", "supervised": "supervised", "surrogate": "surrogate", "human": "human", "llm": "LLM routing", "other": "other"}  # 定义论文图例中的方法家族名称
LABEL_OFFSETS = {"DM_D": (10, 13), "DM_PSO": (10, 11), "DM_DQN": (10, -19), "LM_D": (10, 14), "LM_PSO": (10, -20), "LM_DQN": (10, 12), "SL_ONE_SHOT": (10, -20), "LLM_PH_ROUTER": (10, 13)}  # 定义已知方法的稳定标签偏移以减少重叠

def read_csv(path: Path) -> list[dict[str, str]]:  # 读取 UTF-8 聚合 CSV 并返回字典记录
    with path.open("r", encoding="utf-8", newline="") as handle:  # 打开输入文件并保持原始换行处理
        return list(csv.DictReader(handle))  # 将全部方法行加载为字典列表

def safe_float(row: dict[str, str], name: str, default: float = 0.0) -> float:  # 从 CSV 记录中安全读取浮点数
    value = row.get(name, "")  # 读取指定列的原始字符串
    if value is None or str(value).strip() == "":  # 检查列值是否为空
        return float(default)  # 在空值时返回调用方指定的默认值
    return float(value)  # 将非空字符串转换为浮点数

def point_size(mean_n_fe: float) -> float:  # 将平均真实有限元调用数映射为可读点面积
    return float(75.0 + 38.0 * math.log1p(max(mean_n_fe, 0.0)))  # 使用对数变换避免 PSO 点过度放大

def fitted_tradeoff_curve(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:  # 使用受约束反比例模型拟合描述性 P–H 下降趋势
    if len(points) < 3:  # 检查实际执行点是否足以支持稳定趋势拟合
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)  # 在点数不足时返回空曲线
    p_values = np.asarray([item[0] for item in points], dtype=np.float64)  # 提取实际执行 P 坐标
    h_values = np.asarray([item[1] for item in points], dtype=np.float64)  # 提取实际执行 H 坐标
    best_parameters: tuple[float, float, float] | None = None  # 初始化最优反比例模型参数
    best_loss = float("inf")  # 初始化最小平方误差
    for offset in np.linspace(0.04, 0.80, 240):  # 网格搜索正偏移以避免 P 接近零时发散
        design = np.column_stack((1.0 / (p_values + offset), np.ones_like(p_values)))  # 构造 h=a/(p+b)+c 的线性最小二乘设计矩阵
        coefficients, _, _, _ = np.linalg.lstsq(design, h_values, rcond=None)  # 在固定偏移下求解 a 与 c 的最小二乘估计
        amplitude = float(coefficients[0])  # 读取反比例项幅值并控制曲线单调方向
        intercept = float(coefficients[1])  # 读取曲线远端截距
        if amplitude <= 0.0:  # 检查拟合曲线是否随 P 增大而下降
            continue  # 跳过不符合 P–H 权衡方向的候选模型
        prediction = amplitude / (p_values + offset) + intercept  # 计算当前候选模型在执行点上的预测 H
        loss = float(np.mean((prediction - h_values) ** 2))  # 计算当前候选模型均方误差
        if loss < best_loss:  # 检查当前候选是否改善最优拟合
            best_loss = loss  # 更新最小拟合误差
            best_parameters = (amplitude, offset, intercept)  # 保存当前最优模型参数
    if best_parameters is None:  # 检查是否没有任何满足单调下降约束的最小二乘拟合
        left_p = float(np.min(p_values))  # 读取实际执行点的最小 P 作为左侧锚点横坐标
        right_p = float(np.max(p_values))  # 读取实际执行点的最大 P 作为右侧锚点横坐标
        left_h = float(np.max(h_values))  # 使用实际执行点最高 H 构造低压缩侧参考锚点
        right_h = float(np.min(h_values))  # 使用实际执行点最低 H 构造高压缩侧参考锚点
        offset = 0.12  # 使用固定正偏移保证反比例参考趋势有限且平滑
        denominator = 1.0 / (left_p + offset) - 1.0 / (right_p + offset)  # 计算两锚点反比例基函数差值
        if abs(denominator) <= 1.0e-12 or left_h <= right_h:  # 检查锚点是否无法形成单调下降趋势
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)  # 在退化情况下返回空曲线
        amplitude = (left_h - right_h) / denominator  # 根据两锚点求解正反比例项幅值
        intercept = left_h - amplitude / (left_p + offset)  # 根据左侧锚点求解参考趋势截距
        best_parameters = (float(amplitude), float(offset), float(intercept))  # 保存由实际范围构造的单调参考参数
    x_min = max(0.02, float(np.min(p_values)) - 0.05)  # 计算趋势曲线左端并限制在图域内
    x_max = min(0.98, float(np.max(p_values)) + 0.08)  # 计算趋势曲线右端并限制在图域内
    x_curve = np.linspace(x_min, x_max, 240)  # 生成平滑趋势曲线横坐标
    amplitude, offset, intercept = best_parameters  # 解包最优反比例模型参数
    y_curve = np.clip(amplitude / (x_curve + offset) + intercept, 0.0, 1.0)  # 计算并裁剪趋势曲线纵坐标
    return x_curve, y_curve  # 返回可直接绘制的描述性 P–H 趋势曲线

def error_extent(center: float, low: float, high: float) -> tuple[float, float]:  # 将绝对置信区间端点转换为 Matplotlib 非负误差长度
    return max(center - low, 0.0), max(high - center, 0.0)  # 返回左/下和右/上两个非负误差长度

def plot_rows(rows: list[dict[str, str]], output_png: Path, output_pdf: Path, title: str, show_frontier: bool) -> None:  # 绘制空心评估点、实心执行点、差距箭头和描述性趋势
    output_png.parent.mkdir(parents=True, exist_ok=True)  # 创建 PNG 输出目录并允许父目录缺失
    output_pdf.parent.mkdir(parents=True, exist_ok=True)  # 创建 PDF 输出目录并允许父目录缺失
    figure, axis = plt.subplots(figsize=(12.8, 9.0), constrained_layout=True)  # 创建单一主图并使用自动紧凑布局
    executed_points: list[tuple[float, float]] = []  # 初始化用于计算 P–H 趋势的实际执行点列表
    present_families: list[str] = []  # 初始化当前图中实际出现的方法家族顺序
    for row in rows:  # 遍历每个方法的聚合 PH 记录
        method = str(row["method"])  # 读取方法短名称
        label = str(row.get("method_label", method))  # 读取完整方法标签并在缺失时使用短名称
        family = str(row.get("method_family", "other"))  # 读取方法家族并在缺失时归入其他类
        if family not in present_families:  # 检查当前方法家族是否首次出现
            present_families.append(family)  # 按数据顺序保存家族图例顺序
        color = FAMILY_COLORS.get(family, FAMILY_COLORS["other"])  # 读取当前方法家族颜色
        assessed_p = safe_float(row, "assessed_p_mean")  # 读取执行前评估 P 均值
        executed_p = safe_float(row, "executed_p_mean")  # 读取实际执行 P 均值
        assessed_h = safe_float(row, "assessed_h_mean")  # 读取执行前评估 H 均值
        executed_h = safe_float(row, "executed_h_mean")  # 读取实际执行 H 均值
        executed_points.append((executed_p, executed_h))  # 将实际执行点加入趋势拟合输入
        size = point_size(safe_float(row, "mean_n_fe"))  # 根据平均真实有限元调用数计算点面积
        assessed_xerr = error_extent(assessed_p, safe_float(row, "assessed_p_ci_low", assessed_p), safe_float(row, "assessed_p_ci_high", assessed_p))  # 计算评估 P 的置信区间长度
        assessed_yerr = error_extent(assessed_h, safe_float(row, "assessed_h_ci_low", assessed_h), safe_float(row, "assessed_h_ci_high", assessed_h))  # 计算评估 H 的置信区间长度
        executed_xerr = error_extent(executed_p, safe_float(row, "executed_p_ci_low", executed_p), safe_float(row, "executed_p_ci_high", executed_p))  # 计算执行 P 的置信区间长度
        executed_yerr = error_extent(executed_h, safe_float(row, "executed_h_ci_low", executed_h), safe_float(row, "executed_h_ci_high", executed_h))  # 计算执行 H 的置信区间长度
        axis.errorbar(assessed_p, assessed_h, xerr=np.asarray([[assessed_xerr[0]], [assessed_xerr[1]]]), yerr=np.asarray([[assessed_yerr[0]], [assessed_yerr[1]]]), fmt="none", ecolor=color, elinewidth=1.0, alpha=0.35, capsize=2.5, zorder=1)  # 绘制评估 PH 的半透明置信区间
        axis.errorbar(executed_p, executed_h, xerr=np.asarray([[executed_xerr[0]], [executed_xerr[1]]]), yerr=np.asarray([[executed_yerr[0]], [executed_yerr[1]]]), fmt="none", ecolor=color, elinewidth=1.2, alpha=0.58, capsize=2.5, zorder=2)  # 绘制实际执行 PH 的置信区间
        axis.annotate("", xy=(executed_p, executed_h), xytext=(assessed_p, assessed_h), arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.65, "alpha": 0.78, "shrinkA": 7.0, "shrinkB": 7.0}, zorder=3)  # 绘制从评估点指向实际执行点的差距箭头
        axis.scatter(assessed_p, assessed_h, s=size, facecolors="white", edgecolors=color, linewidths=2.0, marker="o", zorder=4)  # 绘制空心执行前评估点
        axis.scatter(executed_p, executed_h, s=size, facecolors=color, edgecolors="black", linewidths=0.9, marker="o", zorder=5)  # 绘制实心实际执行点
        offset = LABEL_OFFSETS.get(method, (9, 8 if executed_h <= assessed_h else -18))  # 读取已知方法标签偏移或根据箭头方向生成默认偏移
        axis.annotate(label, xy=(executed_p, executed_h), xytext=offset, textcoords="offset points", fontsize=10.2, ha="left", va="center", color="black", zorder=6)  # 在实际执行点附近标注方法名称
    if show_frontier:  # 检查用户是否要求绘制描述性 P–H 权衡趋势
        trend_p, trend_h = fitted_tradeoff_curve(executed_points)  # 使用实际执行点拟合单调下降的反比例趋势
        if trend_p.size >= 2:  # 检查是否存在可绘制的有效趋势曲线
            axis.plot(trend_p, trend_h, linestyle="--", linewidth=1.7, color="black", alpha=0.72, label="Fitted P–H trade-off trend", zorder=0)  # 绘制由实际执行点拟合的虚线权衡趋势
    axis.set_xlim(0.0, 1.0)  # 固定先验压缩横轴范围为零到一
    axis.set_ylim(0.0, 1.0)  # 固定物理反馈纵轴范围为零到一
    axis.set_xlabel("P = prior / offline compression reliance", fontsize=15, labelpad=12)  # 设置横轴标题
    axis.set_ylabel("H = normalized sequential physics-feedback depth", fontsize=15, labelpad=12)  # 设置纵轴标题并明确归一化定义
    axis.set_title(title, fontsize=20, fontweight="bold", pad=16)  # 设置论文图主标题
    axis.grid(True, which="major", linewidth=0.65, alpha=0.18)  # 添加轻量主网格线辅助读数
    axis.tick_params(axis="both", which="major", labelsize=11.5, length=6.0, width=1.1)  # 设置主刻度字体和线宽
    for spine in axis.spines.values():  # 遍历四条坐标轴边框
        spine.set_linewidth(1.35)  # 加粗坐标轴边框以接近期刊图风格
        spine.set_color("black")  # 将坐标轴边框固定为黑色
    family_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLORS.get(family, FAMILY_COLORS["other"]), markeredgecolor="black", markersize=9.0, label=FAMILY_LABELS.get(family, family)) for family in present_families]  # 构造方法家族颜色图例句柄
    state_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.7, markersize=9.0, label="assessed PH"), Line2D([0], [0], marker="o", color="none", markerfacecolor="black", markeredgecolor="black", markersize=9.0, label="executed PH"), Line2D([0, 1], [0, 0], color="black", linewidth=1.5, marker=">", markevery=[1], label="assessment → execution gap")]  # 构造空心点、实心点和箭头含义图例句柄
    legend = axis.legend(handles=family_handles + state_handles, loc="lower left", fontsize=9.6, frameon=True, framealpha=0.96, edgecolor="black", borderpad=0.75, labelspacing=0.6, handlelength=2.0)  # 绘制合并后的方法家族与状态图例
    legend.get_frame().set_linewidth(0.9)  # 设置图例边框线宽
    axis.text(0.985, 0.985, "Arrow length measures PH assessment error\nPoint size scales with mean online FE calls", transform=axis.transAxes, ha="right", va="top", fontsize=9.5, bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black", "alpha": 0.88})  # 在右上角解释箭头和点大小编码
    figure.savefig(output_png, dpi=300, bbox_inches="tight")  # 保存高分辨率 PNG 论文图
    figure.savefig(output_pdf, bbox_inches="tight")  # 保存矢量 PDF 论文图
    plt.close(figure)  # 关闭图形并释放内存资源

def parse_args() -> argparse.Namespace:  # 定义并解析命令行参数
    parser = argparse.ArgumentParser(description="Plot assessed-to-executed PH gaps from an aggregate CSV.")  # 创建命令行解析器
    parser.add_argument("--input", type=Path, required=True, help="Aggregate PH gap CSV produced by build_ph_gap_table.py.")  # 声明方法级聚合 CSV 输入路径
    parser.add_argument("--output-png", type=Path, required=True, help="Destination PNG path.")  # 声明高分辨率 PNG 输出路径
    parser.add_argument("--output-pdf", type=Path, required=True, help="Destination PDF path.")  # 声明矢量 PDF 输出路径
    parser.add_argument("--title", type=str, default="Assessed vs executed P–H positioning of adaptive-meshing methods", help="Figure title.")  # 声明可覆盖的论文图标题
    parser.add_argument("--no-frontier", action="store_true", help="Disable the fitted P-H trade-off trend.")  # 声明关闭描述性趋势的可选开关
    return parser.parse_args()  # 返回解析后的命令行参数

def main() -> None:  # 执行聚合数据读取和论文图生成流程
    args = parse_args()  # 读取命令行参数
    rows = read_csv(args.input)  # 读取方法级 PH 差距聚合表
    if not rows:  # 检查输入 CSV 是否没有任何方法记录
        raise RuntimeError("input aggregate CSV is empty")  # 在输入为空时拒绝生成无意义图形
    plot_rows(rows, args.output_png, args.output_pdf, args.title, not args.no_frontier)  # 绘制并保存脚本生成的 PH 差距图
    print(f"wrote {args.output_png} and {args.output_pdf}")  # 输出生成文件路径供 CI 日志审计

if __name__ == "__main__":  # 检查脚本是否作为主程序运行
    main()  # 启动 PH 差距绘图流程
