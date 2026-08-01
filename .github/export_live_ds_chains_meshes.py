from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

SOURCE_RUN = "30707886295"
INPUT = Path("input")
OUTPUT = Path("output")
OUTPUT.mkdir(parents=True, exist_ok=True)

CASES = {
    "bearing_load_introduction": "01_支座载荷",
    "web_circular_opening": "02_单圆孔",
    "diaphragm_multi_opening_budget": "03_三孔横隔板",
    "bridge_web_crack": "04_裂纹腹板",
}


def find_case_file(root: Path, case_id: str, name: str) -> Path:
    candidates = [p for p in root.rglob(name) if case_id in p.as_posix()]
    if not candidates:
        candidates = list(root.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"{case_id}: {name}")
    return min(candidates, key=lambda p: len(p.as_posix()))


def resolve_relative(case_root: Path, relative: str) -> Path:
    direct = case_root / relative
    if direct.is_file():
        return direct
    matches = list(case_root.rglob(Path(relative).name))
    if not matches:
        raise FileNotFoundError(relative)
    return matches[0]


def mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    nodes = np.asarray(data["nodes"], dtype=float)
    elements = np.asarray(data["elements"], dtype=int)
    if elements.size and elements.min() >= 1 and elements.max() == len(nodes):
        elements = elements - 1
    return nodes, elements


def element_areas(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    values = []
    for element in elements:
        xy = nodes[element, :2]
        x, y = xy[:, 0], xy[:, 1]
        values.append(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return np.asarray(values)


def draw_mesh(npz_path: Path, png_path: Path, title: str, zoom: bool = False) -> None:
    nodes, elements = mesh_arrays(npz_path)
    segments = []
    for element in elements:
        xy = nodes[element, :2]
        for index in range(len(xy)):
            segments.append([xy[index], xy[(index + 1) % len(xy)]])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.add_collection(LineCollection(segments, linewidths=0.22))
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    if zoom and len(elements):
        areas = element_areas(nodes, elements)
        threshold = np.quantile(areas, 0.08)
        chosen = elements[areas <= threshold]
        cloud = nodes[np.unique(chosen), :2]
        if len(cloud):
            xmin, ymin = cloud.min(axis=0)
            xmax, ymax = cloud.max(axis=0)
            dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
            ax.set_xlim(xmin - 0.25 * dx, xmax + 0.25 * dx)
            ax.set_ylim(ymin - 0.25 * dy, ymax + 0.25 * dy)
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


combined_raw = [
    "四案例 live DeepSeek 原始回复",
    "",
    f"来源：GitHub Actions Run {SOURCE_RUN}。",
    "只提取 model_io 中的 raw_text；未翻译、未改写、未删减。",
    "",
]
combined_chain = [
    "四案例 live DeepSeek 决策链与 Skill 反馈",
    "",
    f"来源：GitHub Actions Run {SOURCE_RUN}。",
    "",
]

for case_id, folder_name in CASES.items():
    source = INPUT / case_id
    target = OUTPUT / folder_name
    target.mkdir(parents=True, exist_ok=True)

    trace_path = find_case_file(source, case_id, "agent_trace.json")
    case_root = trace_path.parent
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    shutil.copy2(trace_path, target / "agent_trace.json")

    final_report_candidates = list(case_root.glob("final_report.json"))
    if final_report_candidates:
        shutil.copy2(final_report_candidates[0], target / "final_report.json")

    raw_lines = [
        f"{folder_name} live DeepSeek 原始回复",
        "",
        f"case_id: {case_id}",
        f"status: {trace.get('status')}",
        "",
    ]
    response_paths = sorted((case_root / "model_io").glob("iteration_*_response.json"))
    for response_path in response_paths:
        response = json.loads(response_path.read_text(encoding="utf-8"))
        raw_text = response.get("raw_text", "")
        iteration = response_path.stem.split("_")[1]
        block = ["=" * 88, f"第 {int(iteration)} 轮", "=" * 88, raw_text, ""]
        raw_lines.extend(block)
        combined_raw.extend(["#" * 96, f"{folder_name}｜第 {int(iteration)} 轮", "#" * 96, raw_text, ""])
    (target / "01_DeepSeek逐轮原始回复.txt").write_text("\n".join(raw_lines).rstrip() + "\n", encoding="utf-8")

    chain_lines = [
        f"{folder_name} live 决策链",
        "",
        f"case_id: {case_id}",
        f"status: {trace.get('status')}",
        f"model_calls: {trace.get('counters', {}).get('model_calls')}",
        f"solver_calls: {trace.get('counters', {}).get('solver_calls')}",
        f"optimizer_calls: {trace.get('counters', {}).get('optimizer_calls')}",
        "",
    ]
    for event in trace.get("events", []):
        if event.get("type") == "model_decision":
            decision = event.get("decision", {})
            block = [
                "=" * 88,
                f"第 {event.get('iteration')} 轮 DeepSeek 决策",
                f"工程摘要：{event.get('engineer_facing_summary', '')}",
                f"当前判断：{event.get('current_judgment', '')}",
                f"主要不确定性：{event.get('main_uncertainty', '')}",
                f"决策类型：{decision.get('type', '')}",
            ]
            if decision.get("type") == "call_skill":
                block.extend([
                    f"Skill：{decision.get('skill', '')}",
                    "参数：",
                    json.dumps(decision.get("arguments", {}), ensure_ascii=False, indent=2),
                    f"理由：{decision.get('why', '')}",
                    f"预期结果：{decision.get('expected_result', '')}",
                ])
            else:
                block.extend(["最终回答：", str(decision.get("answer", ""))])
            block.append("")
            chain_lines.extend(block)
            combined_chain.extend([f"[{folder_name}]", *block])
        elif event.get("type") == "skill_call":
            result = event.get("result", {})
            block = [
                f"Skill 返回：{event.get('request', {}).get('skill', '')}",
                f"状态：{result.get('status', '')}",
                f"摘要：{result.get('summary', '')}",
                "关键数据：",
                json.dumps(result.get("data", {}), ensure_ascii=False, indent=2),
                "",
            ]
            chain_lines.extend(block)
            combined_chain.extend([f"[{folder_name}]", *block])
    (target / "02_逐轮决策与Skill反馈.txt").write_text("\n".join(chain_lines).rstrip() + "\n", encoding="utf-8")

    pso_path = find_case_file(case_root, case_id, "region_pso_1.json")
    pso = json.loads(pso_path.read_text(encoding="utf-8"))
    shutil.copy2(pso_path, target / "region_pso_1.json")
    for artifact_name in ("region_partition_1.json", "region_validation_1.json", "region_comparison_1.json"):
        matches = list(case_root.rglob(artifact_name))
        if matches:
            shutil.copy2(matches[0], target / artifact_name)

    mesh_specs = [
        ("optimized", pso["best"]["raw_file"], "PSO optimized actual mesh"),
        ("uniform", pso["_precomputed_validation"]["uniform"]["raw_file"], "Same-budget uniform actual mesh"),
    ]
    heuristic = pso.get("_precomputed_validation", {}).get("heuristic", {}).get("raw_file")
    if heuristic:
        mesh_specs.append(("heuristic", heuristic, "Fixed heuristic actual mesh"))

    for label, relative, title in mesh_specs:
        mesh_path = resolve_relative(case_root, relative)
        copied = target / f"mesh_{label}.npz"
        shutil.copy2(mesh_path, copied)
        nodes, elements = mesh_arrays(copied)
        count = len(elements)
        draw_mesh(copied, target / f"mesh_{label}_full.png", f"{title} ({count} elements)")
        if label == "optimized":
            draw_mesh(copied, target / "mesh_optimized_hotspot_zoom.png", f"Optimized mesh hotspot zoom ({count} elements)", zoom=True)

    partition_images = list(case_root.rglob("partition_evidence.png"))
    if partition_images:
        shutil.copy2(partition_images[0], target / "partition_evidence.png")

(OUTPUT / "00_四案例_DeepSeek原始回复_合并.txt").write_text("\n".join(combined_raw).rstrip() + "\n", encoding="utf-8")
(OUTPUT / "00_四案例_决策链与Skill反馈_合并.txt").write_text("\n".join(combined_chain).rstrip() + "\n", encoding="utf-8")
(OUTPUT / "README.txt").write_text(
    "本包来自 GitHub Actions Run 30707886295。\n"
    "每个案例包含：DeepSeek逐轮原始回复、完整agent_trace、可读决策链、PSO/区域JSON、实际优化/均匀/启发式网格NPZ与PNG、热点局部放大。\n"
    "四个主运行均成功；原工作流failure来自收尾审计对conversation_id字段的错误强制要求。\n",
    encoding="utf-8",
)
shutil.make_archive("四案例_live_DeepSeek_决策链与实际网格", "zip", OUTPUT)
