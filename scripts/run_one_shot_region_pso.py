#!/usr/bin/env python3  # 指定使用当前Python 3解释器运行。
from __future__ import annotations  # 启用延迟类型注解。
import argparse  # 提供命令行参数解析。
import json  # 提供输入与报告JSON读写。
import shlex  # 安全拆分可选的终局认证命令。
import subprocess  # 执行可选的一次高保真求解器认证。
import sys  # 配置仓库源码导入路径。
from pathlib import Path  # 管理输入输出与认证文件路径。
from typing import Any, Mapping  # 提供认证闭包类型注解。
REPO_ROOT = Path(__file__).resolve().parents[1]  # 定位实现根目录。
SRC_ROOT = REPO_ROOT / "src"  # 定位Python源码目录。
if str(SRC_ROOT) not in sys.path:  # 检查源码目录是否已在模块搜索路径中。
    sys.path.insert(0, str(SRC_ROOT))  # 将源码目录插入最高优先级。
from engineering_agent.one_shot_region_pso import algorithm_from_mapping  # 导入严格一次性编排器构造函数。
def build_command_certifier(command_template: str, output_dir: Path):  # 构造只会被编排器调用一次的外部认证闭包。
    sizes_path = output_dir / "selected_region_sizes.json"  # 规定终局尺寸粒子输入文件。
    result_path = output_dir / "high_fidelity_certification.json"  # 规定高保真认证结果文件。
    stdout_path = output_dir / "high_fidelity_certification.stdout.txt"  # 规定求解器标准输出留档文件。
    def certify(sizes: Mapping[str, float]) -> Mapping[str, Any]:  # 定义终局高保真认证调用。
        sizes_path.write_text(json.dumps(dict(sizes), ensure_ascii=False, indent=2), encoding="utf-8")  # 写入唯一终局尺寸粒子。
        tokens = shlex.split(command_template)  # 按命令行规则拆分用户提供的命令模板。
        command = [token.replace("{sizes_json}", str(sizes_path)).replace("{result_json}", str(result_path)) for token in tokens]  # 注入输入与结果文件路径。
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)  # 仅执行一次外部高保真命令。
        stdout_path.write_text(completed.stdout, encoding="utf-8")  # 保存完整求解器输出以便审计。
        if completed.returncode != 0:  # 检查高保真命令是否成功退出。
            raise RuntimeError(f"high-fidelity certifier failed with code {completed.returncode}")  # 阻止伪造成功认证。
        if not result_path.exists():  # 检查认证器是否生成约定结果文件。
            raise RuntimeError("high-fidelity certifier did not create result JSON")  # 报告接口合同缺失。
        return json.loads(result_path.read_text(encoding="utf-8"))  # 返回真实求解器认证结果。
    return certify  # 返回只在终局粒子上调用的认证闭包。
def main() -> int:  # 定义命令行入口。
    parser = argparse.ArgumentParser(description="Run one-shot delegated regional agents followed by one or two micro-PSO calibrations.")  # 创建参数解析器。
    parser.add_argument("--input", default=str(REPO_ROOT / "examples" / "one_shot_region_pso_case.json"))  # 设置默认算例输入路径。
    parser.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "one_shot_region_pso_report.json"))  # 设置默认报告输出路径。
    parser.add_argument("--pso-generations", type=int, choices=(1, 2), default=None)  # 允许命令行覆盖为一代或两代校准。
    parser.add_argument("--certifier-command", default=None)  # 接收可选的一次高保真认证命令模板。
    args = parser.parse_args()  # 解析命令行参数。
    input_path = Path(args.input).resolve()  # 解析输入JSON绝对路径。
    output_path = Path(args.output).resolve()  # 解析报告JSON绝对路径。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建报告及认证文件目录。
    payload = json.loads(input_path.read_text(encoding="utf-8"))  # 读取完整算法输入载荷。
    if args.pso_generations is not None:  # 检查是否覆盖PSO代数。
        payload.setdefault("config", {})["pso_generations"] = args.pso_generations  # 强制使用一代或两代微型PSO。
    algorithm = algorithm_from_mapping(payload)  # 构造严格一次性多智能体与微型PSO编排器。
    certifier = build_command_certifier(args.certifier_command, output_path.parent) if args.certifier_command else None  # 构造可选终局认证器。
    result = algorithm.run(certifier=certifier)  # 执行一次委派、一次估计通信调整和一到两代PSO。
    report = result.to_dict()  # 将完整审计结果转换为JSON字典。
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存完整运行报告。
    summary = {"base_error": result.base_particle.evaluation.predicted_error, "base_resource": result.base_particle.evaluation.predicted_resource, "selected_error": result.selected_particle.evaluation.predicted_error, "selected_resource": result.selected_particle.evaluation.predicted_resource, "selected_coordinates": [result.selected_particle.scale_coordinate, result.selected_particle.transfer_coordinate], "trace": report["trace"], "output": str(output_path)}  # 构造终端摘要。
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # 输出关键结果与调用计数。
    return 0  # 返回成功退出码。
if __name__ == "__main__":  # 检查脚本是否作为主程序执行。
    raise SystemExit(main())  # 执行命令行入口并传递退出码。
