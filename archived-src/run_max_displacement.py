import argparse
import json
import os
import subprocess
import sys


def build_command(abaqus_exec, odb_path, instance_name, step_name, frame_index):
    script_path = os.path.join(os.path.dirname(__file__), "get_max_displacement.py")
    cmd = [abaqus_exec, "python", script_path, odb_path]
    if instance_name is not None:
        cmd.append(instance_name)
    else:
        cmd.append("None")
    if step_name is not None:
        cmd.append(step_name)
    else:
        cmd.append("None")
    if frame_index is not None:
        cmd.append(str(frame_index))
    return cmd


def summarize_outputs(odb_path, open_plot=False):
    output_dir = os.path.dirname(odb_path)
    odb_basename = os.path.splitext(os.path.basename(odb_path))[0]
    json_path = os.path.join(output_dir, "{}_max_displacement.json".format(odb_basename))

    if not os.path.exists(json_path):
        print("提示: 未找到输出 JSON 文件，跳过结果汇总。预期位置: {}".format(json_path))
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print("提示: 读取结果 JSON 失败: {}".format(exc))
        return

    max_disp = data.get("max_displacement")
    if max_disp:
        print("\n最大位移汇总:")
        magnitude = max_disp.get("magnitude")
        node_label = max_disp.get("node_label")
        instance_name = max_disp.get("instance_name")
        coordinates = max_disp.get("coordinates")
        print("  位移幅值: {:.6f}".format(magnitude) if magnitude is not None else "  位移幅值: 未知")
        print("  节点: {} @ {}".format(node_label, instance_name))
        if coordinates:
            print("  位置: ({:.3f}, {:.3f}, {:.3f})".format(*coordinates))

    total_energy = data.get("total_strain_energy")
    if total_energy is not None:
        print("\n全局应变能 (ALLSE): {:.6f}".format(total_energy))

    mesh_stats = data.get("mesh_size_distribution")
    if mesh_stats:
        print("\n网格尺寸统计:")
        print("  单元数量: {}".format(mesh_stats.get("element_count")))
        min_size = mesh_stats.get("min_size")
        max_size = mesh_stats.get("max_size")
        if min_size is not None and max_size is not None:
            print("  尺寸范围: {:.3f} ~ {:.3f}".format(min_size, max_size))
        mean_size = mesh_stats.get("mean_size")
        median_size = mesh_stats.get("median_size")
        if mean_size is not None and median_size is not None:
            print("  平均/中位数: {:.3f} / {:.3f}".format(mean_size, median_size))
        percentiles = mesh_stats.get("percentiles") or {}
        print(
            "  P10 / P90: {:.3f} / {:.3f}".format(percentiles.get("p10", 0.0), percentiles.get("p90", 0.0))
        )
        plot_path = mesh_stats.get("plot_path")
        if plot_path:
            print("  尺寸直方图: {}".format(plot_path))
            if open_plot:
                open_histogram(plot_path)


def open_histogram(plot_path):
    if not plot_path or not os.path.exists(plot_path):
        print("提示: 找不到可视化图像: {}".format(plot_path))
        return
    try:
        if sys.platform.startswith("win") and hasattr(os, "startfile"):
            os.startfile(plot_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", plot_path], check=False)
        else:
            subprocess.run(["xdg-open", plot_path], check=False)
    except Exception as exc:
        print("提示: 自动打开图像失败: {}".format(exc))


def main():
    parser = argparse.ArgumentParser(description="调用 Abaqus 脚本提取最大位移。")
    parser.add_argument("--odb-path", default="test_output/test_job.odb", help="ODB 文件路径")
    parser.add_argument("--instance", help="实例名称，默认整装配", default='The whole beam')
    parser.add_argument("--step", help="分析步名称，默认最后一个", default=None)
    parser.add_argument("--frame", type=int, help="帧索引，默认最后一帧", default=None)
    parser.add_argument("--abaqus", help="Abaqus 可执行命令，默认 'abaqus'", default="F:/SIMULIA/Commands/abaqus.bat")
    parser.add_argument("--open-plot", action="store_true", help="生成后自动打开网格尺寸分布图")

    args = parser.parse_args()

    odb_path = os.path.abspath(args.odb_path)
    if not os.path.exists(odb_path):
        print("找不到 ODB 文件: {}".format(odb_path))
        sys.exit(1)

    cmd = build_command(
        abaqus_exec=args.abaqus,
        odb_path=odb_path,
        instance_name=args.instance,
        step_name=args.step,
        frame_index=args.frame,
    )

    print("执行命令: {}".format(" ".join('"{}"'.format(c) if " " in c else c for c in cmd)))

    try:
        completed = subprocess.run(cmd, check=True)
        summarize_outputs(odb_path, open_plot=args.open_plot)
        sys.exit(completed.returncode)
    except subprocess.CalledProcessError as exc:
        print("运行失败，退出码: {}".format(exc.returncode))
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()

