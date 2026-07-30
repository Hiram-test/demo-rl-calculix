#!/usr/bin/env python3  # 使用参数吸附修复层启动开放发现实验。
from __future__ import annotations  # 允许现代类型注解。

import sys  # 把底层主函数退出码传回操作系统。
from typing import Any  # 表示模型提供的动态 JSON 参数。

import scripts.run_deepseek_crack_open_discovery as experiment  # 复用已经通过静态合同测试的开放发现主体。


def _aligned_geometry_energy(arguments: dict[str, Any]) -> dict[str, Any]:  # 把模型提出的连续裂纹延长量吸附到结构网格步长。
    nx = int(arguments.get("nx", max(experiment.INITIAL_LEVELS)))  # 读取或使用当前最细网格等级。
    requested_extension = float(arguments.get("extension_mm", experiment.WIDTH / nx))  # 保存模型原始请求量用于审计。
    if requested_extension <= 0.0 or requested_extension > 5.0:  # 验证请求仍处于工具合同允许范围。
        raise ValueError("extension_mm must lie in (0, 5]")  # 拒绝超出工程工具边界的请求。
    grid_step = experiment.WIDTH / nx  # 计算当前结构网格的节点间距。
    step_count = max(1, int(round(requested_extension / grid_step)))  # 把连续请求转换为至少一个网格步长。
    used_extension = step_count * grid_step  # 计算吸附后的裂纹延长量。
    if used_extension > 5.0:  # 检查四舍五入后是否越过工具上限。
        step_count = max(1, int(5.0 // grid_step))  # 选择不超过上限的最大完整网格步数。
        used_extension = step_count * grid_step  # 更新最终使用的对齐延长量。
    base = experiment._solve(nx, experiment.HALF_CRACK)  # 求解或读取原裂纹模型。
    extended = experiment._solve(nx, experiment.HALF_CRACK + used_extension)  # 求解与网格节点严格对齐的延长裂纹模型。
    added_surface = 2.0 * used_extension * experiment.THICKNESS  # 计算两个裂尖新增裂纹表面积。
    energy_change = float(extended["strain_energy_n_mm"] - base["strain_energy_n_mm"])  # 计算总应变能变化。
    return {"tool": "compare_nearby_geometry_energy", "nx": nx, "requested_extension_mm": requested_extension, "used_extension_mm": used_extension, "parameter_repair": "snapped_to_integer_mesh_steps", "grid_step_mm": grid_step, "base_energy_n_mm": float(base["strain_energy_n_mm"]), "extended_energy_n_mm": float(extended["strain_energy_n_mm"]), "energy_change_n_mm": energy_change, "added_crack_surface_mm2": added_surface, "energy_change_per_added_surface_n_per_mm": energy_change / added_surface, "controlled_variables": ["外形尺寸", "材料参数", "远场应力", "边界约束", "网格划分数"]}  # 返回修复记录和真实能量证据。


experiment._geometry_energy = _aligned_geometry_energy  # 只替换工具参数修复层，不改变模型提示、动作目录或完成条件。


if __name__ == "__main__":  # 仅在脚本直接执行时启动实验。
    raise SystemExit(experiment.main())  # 使用原主体的命令行参数、模型调用和轨迹冻结逻辑。
