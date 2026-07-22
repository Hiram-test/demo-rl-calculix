from odbAccess import openOdb
from abaqusConstants import NODAL
import math
import sys
import json
import os
import statistics


def convert_to_python(obj):
    if isinstance(obj, dict):
        return {convert_to_python(k): convert_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_python(item) for item in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "__float__"):
        return float(obj)
    if hasattr(obj, "__int__"):
        return int(obj)
    return obj


def max_pair_distance(coords):
    max_len = 0.0
    for i in range(len(coords) - 1):
        x1, y1, z1 = coords[i]
        for j in range(i + 1, len(coords)):
            x2, y2, z2 = coords[j]
            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > max_len:
                max_len = dist
    return max_len


def percentile(sorted_values, ratio):
    if not sorted_values:
        return 0.0
    if ratio <= 0.0:
        return sorted_values[0]
    if ratio >= 1.0:
        return sorted_values[-1]
    position = ratio * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def build_histogram(values, bins=20):
    if not values:
        return [], []
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return [len(values)], [min_val - 0.5, max_val + 0.5]
    bin_width = (max_val - min_val) / float(bins)
    edges = [min_val + i * bin_width for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for val in values:
        idx = int((val - min_val) / bin_width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return counts, edges


def configure_matplotlib_fonts(matplotlib_module):
    try:
        from matplotlib import font_manager as fm
    except Exception:
        return None

    candidates = ["SimHei", "Microsoft YaHei", "Microsoft JhengHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]
    for font_name in candidates:
        try:
            fm.findfont(font_name, fallback_to_default=False)
            matplotlib_module.rcParams["font.family"] = font_name
            matplotlib_module.rcParams["axes.unicode_minus"] = False
            return font_name
        except Exception:
            continue
    matplotlib_module.rcParams["axes.unicode_minus"] = False
    print("Warning: 未找到常见中文字体，将可能出现缺失字形。")
    return None


def save_histogram_plot(values, bin_edges, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Warning: 无法导入 matplotlib，跳过网格尺寸直方图绘制: {}".format(exc))
        return None

    configure_matplotlib_fonts(matplotlib)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bin_edges, color="#1f77b4", edgecolor="#0f3b66")
    plt.xlabel("特征长度 (模型单位)")
    plt.ylabel("单元数量")
    plt.title("网格尺寸分布")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
        print("网格尺寸分布图已保存: {}".format(output_path))
        return output_path
    finally:
        plt.close()


def compute_mesh_size_distribution(instance, histogram_bins=30, plot_output_path=None):
    node_lookup = {node.label: tuple(node.coordinates) for node in instance.nodes}
    element_sizes = []

    for element in instance.elements:
        coords = []
        missing = False
        for label in element.connectivity:
            coord = node_lookup.get(label)
            if coord is None:
                missing = True
                break
            coords.append(coord)
        if missing or len(coords) < 2:
            continue
        size = max_pair_distance(coords)
        if size > 0.0:
            element_sizes.append(size)

    if not element_sizes:
        print("Warning: 未能计算到任何单元尺寸。")
        return None

    element_sizes.sort()

    histogram_counts, histogram_edges = build_histogram(element_sizes, bins=histogram_bins)
    plot_path = None
    if histogram_counts and histogram_edges and plot_output_path:
        plot_path = save_histogram_plot(element_sizes, histogram_edges, plot_output_path)

    stats = {
        "element_count": len(element_sizes),
        "min_size": element_sizes[0],
        "max_size": element_sizes[-1],
        "mean_size": statistics.fmean(element_sizes),
        "median_size": statistics.median(element_sizes),
        "std_size": statistics.pstdev(element_sizes) if len(element_sizes) > 1 else 0.0,
        "percentiles": {
            "p10": percentile(element_sizes, 0.10),
            "p90": percentile(element_sizes, 0.90),
        },
        "histogram": {
            "counts": histogram_counts,
            "bin_edges": histogram_edges,
        },
        "plot_path": plot_path,
        "units": "与模型一致",
    }

    return stats


def get_total_strain_energy(step):
    for region in step.historyRegions.values():
        history_outputs = getattr(region, "historyOutputs", None)
        if not history_outputs:
            continue
        if "ALLSE" in history_outputs:
            history = history_outputs["ALLSE"]
            if history.data:
                return history.data[-1][1]
    print("Warning: 未能在历史输出中找到 ALLSE，总应变能将返回 0.")
    return 0.0


def get_max_displacement(odb_path, step_name=None, frame_index=-1, instance_name=None):
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        if not odb.steps:
            raise ValueError("ODB中没有分析步。")

        if step_name:
            if step_name not in odb.steps:
                raise KeyError("未找到指定分析步 '{}'。".format(step_name))
            step = odb.steps[step_name]
        else:
            step = odb.steps[list(odb.steps.keys())[-1]]
            step_name = step.name

        frames = step.frames
        if not frames:
            raise ValueError("分析步 '{}' 中没有帧。".format(step_name))

        frame = frames[frame_index]

        if "U" not in frame.fieldOutputs:
            raise ValueError("帧中不存在位移场 'U'。")

        u_field = frame.fieldOutputs["U"]

        instances = odb.rootAssembly.instances
        if instance_name:
            if instance_name not in instances:
                raise KeyError("未找到实例 '{}'。".format(instance_name))
            target_instances = [(instance_name, instances[instance_name])]
        else:
            target_instances = list(instances.items())

        max_result = {
            "magnitude": -1.0,
            "node_label": None,
            "instance_name": None,
            "coordinates": None,
            "displacement_vector": None,
        }

        output_dir = os.path.dirname(odb_path)
        odb_basename = os.path.splitext(os.path.basename(odb_path))[0]
        plot_path = os.path.join(output_dir, "{}_mesh_size_distribution.png".format(odb_basename))

        selected_instance = None
        for inst_key, inst in target_instances:
            inst_field = u_field.getSubset(region=inst, position=NODAL)
            for val in inst_field.values:
                disp_vec = val.data
                magnitude = math.sqrt(sum(component ** 2 for component in disp_vec))
                if magnitude > max_result["magnitude"]:
                    node = inst.getNodeFromLabel(val.nodeLabel)
                    max_result.update(
                        {
                            "magnitude": magnitude,
                            "node_label": val.nodeLabel,
                            "instance_name": inst_key,
                            "coordinates": list(node.coordinates),
                            "displacement_vector": list(disp_vec),
                        }
                    )
                    selected_instance = inst

        if max_result["node_label"] is None:
            raise RuntimeError("未能在指定区域找到位移结果。")

        max_result["step_name"] = step_name
        max_result["frame_index"] = frame_index

        mesh_stats = None
        if selected_instance is None and target_instances:
            selected_instance = target_instances[0][1]
        if selected_instance is not None:
            mesh_stats = compute_mesh_size_distribution(
                selected_instance, histogram_bins=30, plot_output_path=plot_path
            )

        total_strain_energy = get_total_strain_energy(step)

        return {
            "max_displacement": max_result,
            "mesh_size_distribution": mesh_stats,
            "total_strain_energy": total_strain_energy,
        }
    finally:
        if "odb" in locals():
            close_method = getattr(odb, "close", None)
            is_closed = getattr(odb, "isClosed", None)
            if callable(is_closed):
                closed_flag = is_closed()
            else:
                closed_flag = getattr(odb, "closed", False)
            if callable(close_method) and not closed_flag:
                close_method()


def main():
    if len(sys.argv) < 2:
        print("用法: abaqus python get_max_displacement.py <odb_path> [instance_name] [step_name] [frame_index]")
        sys.exit(1)

    odb_path = sys.argv[1]
    instance_name = sys.argv[2] if len(sys.argv) >= 3 and sys.argv[2] != "None" else None
    step_name = sys.argv[3] if len(sys.argv) >= 4 and sys.argv[3] != "None" else None
    frame_index = int(sys.argv[4]) if len(sys.argv) >= 5 else -1

    result = get_max_displacement(
        odb_path=odb_path,
        step_name=step_name,
        frame_index=frame_index,
        instance_name=instance_name,
    )

    result = convert_to_python(result)

    print("提取结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    output_dir = os.path.dirname(odb_path)
    odb_basename = os.path.splitext(os.path.basename(odb_path))[0]
    output_path = os.path.join(output_dir, "{}_max_displacement.json".format(odb_basename))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("结果已保存至: {}".format(output_path))


if __name__ == "__main__":
    main()

