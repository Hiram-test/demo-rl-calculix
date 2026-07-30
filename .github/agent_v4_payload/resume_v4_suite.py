# 启用延迟类型注解，避免运行时解析仅用于静态说明的复合类型。
from __future__ import annotations

# 导入命令行参数解析器，用于接收已有产物目录、提供器和模型名称。
import argparse
# 导入 JSON 编码器，用于把合并后的四场景收据打印到工作流日志。
import json
# 导入目录操作模块，用于仅清除本轮需要重算的两个旧场景目录。
import shutil
# 导入子进程模块，用于逐个调用现有单场景真实求解入口。
import subprocess
# 导入解释器与模块路径接口，用于调用同一 Python 环境并加载载荷源码。
import sys
# 导入路径类型，用于安全拼接仓库内案例文件和产物目录。
from pathlib import Path


# 解析展开后的 V4 载荷根目录，约束所有脚本和案例路径都位于该目录。
ROOT = Path(__file__).resolve().parents[1]
# 当载荷根目录尚未进入模块搜索路径时补入首位，确保导入当前版本源码。
if str(ROOT) not in sys.path:
    # 把载荷根目录置于搜索路径首位，避免误用 runner 上的同名包。
    sys.path.insert(0, str(ROOT))


# 导入案例合同类型，用于从四份 JSON 恢复完整场景元数据。
from bridge_agent.contracts import CaseDefinition
# 复用完整套件已有的摘要和收据生成逻辑，使续跑产物与首轮格式一致。
from scripts.run_bridge_agent_suite import _write_summary, validate


# 固定论文所需的四份案例文件顺序，保证摘要和论文中的场景顺序稳定。
ALL_CASE_FILES: tuple[str, ...] = (
    # 支座载荷引入案例是本轮需要从头补算的第一个缺失场景。
    "bearing_load.json",
    # 圆孔腹板案例是本轮需要从头补算的第二个缺失场景。
    "circular_opening.json",
    # 多孔横隔板案例沿用上一轮已完成的真实求解产物。
    "multi_hole_pso.json",
    # 裂纹案例沿用上一轮已完成的真实弹塑性 CalculiX 产物。
    "cracked_panel.json",
)
# 只允许删除并重建这两个上一轮因响应截断而中止的场景目录。
RESUME_CASE_IDS = frozenset(
    {
        # 支座场景在区域定义响应处被 2500 token 上限截断。
        "bearing_load_introduction",
        # 圆孔场景在多边形坐标响应处被 2500 token 上限截断。
        "web_circular_opening",
    }
)


# 定义续跑入口，输入为现有四场景产物目录，输出始终为可继续构建论文的进程状态。
def main() -> int:
    """仅补算两个缺失场景，并用四场景产物重建论文输入收据。"""
    # 创建参数解析器，并说明本脚本不会重算已完成的多孔和裂纹场景。
    parser = argparse.ArgumentParser(description="Resume only the two truncated V4 cases in an existing suite artifact.")
    # 接收上一轮完整 artifact 解压后的目录；该目录必须同时包含对话状态和 runs 子目录。
    parser.add_argument("--output-dir", required=True, help="Existing V4 suite artifact directory to update in place.")
    # 接收模型提供器名称；本次正式续跑固定传入 deepseek。
    parser.add_argument("--provider", choices=["deepseek"], default="deepseek", help="Live model provider.")
    # 接收 DeepSeek 环境中配置的模型名称，并原样传给单案例入口。
    parser.add_argument("--model", default="deepseek-v4-pro", help="DeepSeek model configured by the ds environment.")
    # 解析工作流传入的三个参数。
    args = parser.parse_args()
    # 把现有套件目录转换为绝对路径，避免子进程工作目录变化影响产物位置。
    output_dir = Path(args.output_dir).resolve()
    # 保留已下载的旧 artifact，并只在目录不存在时创建父结构。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 按论文固定顺序定位四份案例合同文件。
    case_paths = [ROOT / "cases" / file_name for file_name in ALL_CASE_FILES]
    # 加载全部四个案例，使最终摘要和验证收据仍覆盖完整论文数据集。
    cases = [CaseDefinition.load(case_path) for case_path in case_paths]
    # 为四个案例建立既有 runs 目录映射；完成案例的目录不会被删除或改写。
    run_dirs = {case.case_id: output_dir / "runs" / case.case_id for case in cases}
    # 建立案例 ID 到 JSON 文件名的映射，供单案例入口选择正确输入。
    case_files_by_id = {CaseDefinition.load(case_path).case_id: case_path.name for case_path in case_paths}
    # 按论文顺序检查四个案例，但只执行明确列入续跑集合的两个案例。
    for case in cases:
        # 已完成的多孔与裂纹案例不属于续跑集合，因此直接保留其真实求解产物。
        if case.case_id not in RESUME_CASE_IDS:
            # 跳过本案例，避免重新调用 DeepSeek、PSO 或 CalculiX。
            continue
        # 解析当前缺失案例的精确输出目录。
        run_dir = run_dirs[case.case_id]
        # 仅删除该缺失案例上一轮不完整的目录，为本轮干净重算腾出位置。
        shutil.rmtree(run_dir, ignore_errors=True)
        # 组装现有单案例执行命令，继续使用同一真实求解、PSO 和 DeepSeek 决策路径。
        command = [
            # 使用当前 runner 的 Python 解释器，确保依赖和载荷环境一致。
            sys.executable,
            # 调用原 V4 单案例入口，不引入另一套实现。
            str(ROOT / "scripts" / "run_one_case.py"),
            # 指定案例文件参数名称。
            "--case",
            # 传入当前案例对应的 JSON 文件名。
            case_files_by_id[case.case_id],
            # 指定该案例的独立输出目录参数名称。
            "--output-dir",
            # 传入当前案例的绝对输出目录。
            str(run_dir),
            # 指定模型提供器参数名称。
            "--provider",
            # 传入正式 DeepSeek 提供器。
            args.provider,
            # 指定模型名称参数。
            "--model",
            # 传入 ds 环境选择的 DeepSeek 模型。
            args.model,
        ]
        # 运行当前案例；即使一个案例返回非零，也继续尝试另一个案例并保留所有产物。
        completed = subprocess.run(command, cwd=ROOT, check=False)
        # 当单案例返回非零时把案例 ID 和返回码写入日志，但不设置新的工作流门禁。
        if completed.returncode != 0:
            # 输出可定位的错误信息，便于直接识别哪个实际产物仍未完成。
            print(f"case subprocess returned {completed.returncode}: {case.case_id}", file=sys.stderr)
    # 用两个新结果和两个保留结果重建完整四场景 Markdown 摘要。
    _write_summary(output_dir, cases, run_dirs)
    # 用完整四场景产物重建机器可读验证收据；裂纹仍要求上一轮真实弹塑性证据。
    receipt = validate(output_dir, cases, run_dirs, args.provider, require_plastic=True)
    # 把收据以中文可读 JSON 输出到日志，同时文件版已由 validate 写入 artifact。
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    # 始终返回成功，让论文构建器直接消费实际产物，而不是增加额外门禁。
    return 0


# 仅在工作流直接运行本文件时启动续跑入口。
if __name__ == "__main__":
    # 把入口返回值交给操作系统；本脚本按设计不会用验证状态阻断论文生成。
    raise SystemExit(main())
