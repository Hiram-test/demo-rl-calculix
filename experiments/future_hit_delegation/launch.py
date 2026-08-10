#!/usr/bin/env python3  # 使用稳定仓库导入路径启动未来命中资源重投实验。
from __future__ import annotations  # 启用现代类型注解并保持 Python 3.11 兼容。
import sys  # 将仓库根目录加入模块搜索路径以复用已有真实 CalculiX 实现。
from pathlib import Path  # 跨平台定位当前实验脚本所在仓库根目录。
ROOT = Path(__file__).resolve().parents[2]  # 从 experiments/future_hit_delegation 回到仓库根目录。
PROTOCOL_VERSION = "future-hit-v2-resource-delegation"  # 冻结“当前 Dörfler 支持—未来命中预测—跨轮次资源重投—真实审计”实验版本并触发 Actions。
sys.path.insert(0, str(ROOT))  # 确保 Python 能通过命名空间包导入 experiments 下的已有模块。
from experiments.cross_passage_torsion_benchmark import run_benchmark as base_module  # 正常导入已验证横向通道真实 CalculiX 基准并避免动态 dataclass 模块问题。
from experiments.future_hit_delegation import run as experiment_module  # 导入未来命中实验主体而不立即执行主函数。
experiment_module.load_base_module = lambda: base_module  # 将实验主体的模块载入器替换为已正常注册到 sys.modules 的正式基准模块。
if __name__ == "__main__":  # 仅在 Actions 直接运行启动器时开始真实实验。
    experiment_module.main()  # 执行完整 Dörfler 轨迹、三次真实 LLM 重投、真实 CalculiX 审计和有限纠错实验。
