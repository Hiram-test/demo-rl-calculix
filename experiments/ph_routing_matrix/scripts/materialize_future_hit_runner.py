#!/usr/bin/env python3  # 使用当前 Python 解释器恢复并可选执行 Stage-0B 完整实验源文件
from __future__ import annotations  # 启用延迟类型注解以保持脚本兼容性
import argparse  # 解析归档路径、输出脚本路径和执行参数
import base64  # 将仓库中的 Base64 文本恢复为 gzip 字节
import gzip  # 解压完整 Python 实验源文件
import subprocess  # 在用户要求时调用恢复后的实验程序
import sys  # 将当前 Python 解释器传给恢复后的实验程序
from pathlib import Path  # 使用跨平台路径管理归档和输出文件

def main() -> None:  # 定义归档恢复和可选执行入口
    parser = argparse.ArgumentParser(description="Materialize the Stage-0B future-hit delegation runner")  # 创建命令行参数解析器
    parser.add_argument("--archive", type=Path, default=Path("experiments/ph_routing_matrix/archive/run_future_hit_delegation_v2.py.gz.b64"))  # 指定仓库内压缩归档文本
    parser.add_argument("--output-script", type=Path, default=Path("experiments/ph_routing_matrix/generated/run_future_hit_delegation.py"))  # 指定恢复后的 Python 文件
    parser.add_argument("--run", action="store_true")  # 允许恢复后立即执行完整实验
    parser.add_argument("--result-dir", type=Path, default=Path("experiments/ph_routing_matrix/generated/future_hit_results"))  # 指定完整实验输出目录
    parser.add_argument("--case-seed", type=int, default=20260811)  # 指定参数化物理任务种子
    parser.add_argument("--algorithm-seed", type=int, default=20260811)  # 指定 PSO 和路由随机种子
    parser.add_argument("--quick", action="store_true")  # 允许执行较小规模快速验证
    args = parser.parse_args()  # 解析全部命令行参数
    encoded = args.archive.read_text(encoding="utf-8").strip()  # 读取并清理 Base64 归档文本
    source = gzip.decompress(base64.b64decode(encoded)).decode("utf-8")  # 解码、解压并恢复 UTF-8 Python 源文件
    args.output_script.parent.mkdir(parents=True, exist_ok=True)  # 创建恢复脚本所在目录
    args.output_script.write_text(source, encoding="utf-8")  # 写出可审计的完整实验源文件
    compile(args.output_script.read_text(encoding="utf-8"), str(args.output_script), "exec")  # 在执行前验证恢复文件语法
    print(f"materialized={args.output_script}")  # 输出恢复后的脚本路径
    if args.run:  # 检查是否要求立即执行实验
        command = [sys.executable, str(args.output_script), "--output", str(args.result_dir), "--case-seed", str(args.case_seed), "--algorithm-seed", str(args.algorithm_seed)]  # 构造完整实验命令
        if args.quick:  # 检查是否启用快速验证
            command.append("--quick")  # 将快速验证开关传给完整实验程序
        subprocess.run(command, check=True)  # 执行恢复后的真实有限元实验并在失败时返回非零状态

if __name__ == "__main__":  # 检查当前文件是否作为主程序调用
    main()  # 启动归档恢复和可选实验执行
