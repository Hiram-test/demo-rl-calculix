#!/usr/bin/env python3  # 使用稳定仓库导入路径启动未来命中资源重投实验。
from __future__ import annotations  # 启用现代类型注解并保持 Python 3.11 兼容。
import sys  # 将仓库根目录加入模块搜索路径以复用已有真实 CalculiX 实现。
from pathlib import Path  # 跨平台定位当前实验脚本所在仓库根目录。
ROOT = Path(__file__).resolve().parents[2]  # 从 experiments/future_hit_delegation 回到仓库根目录。
PROTOCOL_VERSION = "future-hit-v1"  # 冻结本轮真实实验协议版本并用于触发已注册 Actions 工作流。
sys.path.insert(0, str(ROOT))  # 确保 Python 能通过命名空间包导入 experiments 下的已有模块。
from experiments.cross_passage_torsion_benchmark import run_benchmark as base_module  # 正常导入已验证横向通道真实 CalculiX 基准并避免动态 dataclass 模块问题。
from experiments.future_hit_delegation import run as experiment_module  # 导入未来命中实验主体而不立即执行主函数。
experiment_module.load_base_module = lambda: base_module  # 将实验主体的模块载入器替换为已正常注册到 sys.modules 的正式基准模块。
if __name__ == "__main__":  # 仅在 Actions 直接运行启动器时开始真实实验。
    experiment_module.main()  # 执行完整 Dörfler 轨迹、LLM 重投、真实审计和纠错实验。
