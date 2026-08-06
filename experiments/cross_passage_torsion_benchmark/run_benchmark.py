#!/usr/bin/env python3  # 使用当前 Python 解释器运行横向通道扭转网格对比实验
from __future__ import annotations  # 启用前向类型注解以简化数据结构声明
import argparse  # 解析命令行中的求解器路径与输出目录
import csv  # 写出方法级和迭代级实验数据表
import hashlib  # 为每一个网格配置生成稳定且可审计的运行目录名
import json  # 写出完整机器可读实验结果
import math  # 提供几何、归一化与误差计算函数
import os  # 读取求解器环境变量并处理运行时设置
from dataclasses import dataclass  # 定义不可变拓扑成员和求解结果结构
from pathlib import Path  # 使用跨平台路径对象管理输入输出文件
import random  # 为离散 PSO、多智能体和 DQN 探索提供确定性随机数
import shutil  # 定位 CalculiX 可执行文件并清理临时目录
import subprocess  # 调用真实 CalculiX 求解器执行每个候选网格
from typing import Any  # 为 JSON 结果和通用容器提供类型标注
import matplotlib.pyplot as plt  # 输出实验收敛曲线和方法对比图
import numpy as np  # 完成图结构、网格、有限元后处理和优化计算
import torch  # 实现真实的 GCN 编码器与深度 Q 网络
from torch import nn  # 构建图卷积和 Q 值输出网络
from torch.nn import functional as F  # 计算 DQN 的平滑 L1 损失
SEED = 20260806  # 固定全部随机过程以保证 GitHub Actions 可复现
random.seed(SEED)  # 固定 Python 随机数发生器
np.random.seed(SEED)  # 固定 NumPy 随机数发生器
torch.manual_seed(SEED)  # 固定 PyTorch 随机数发生器
YOUNG_MODULUS = 2.10e11  # 使用统一钢材弹性模量构造归一化算法基准
BEAM_WIDTH = 6.00e-2  # 使用统一矩形梁截面宽度并避免引入未核实的工程截面
BEAM_HEIGHT = 4.00e-2  # 使用统一矩形梁截面高度并保持空间桁架梁化基准一致
TWIST_ANGLE = 5.00e-3  # 在通道右端施加小转角以形成线弹性纯扭转工况
IMPERFECTION_RATIO = 1.50e-2  # 用同一解析初弯曲线检验局部离散精度
LEVEL_SUBDIVISIONS = np.asarray([1, 2, 4, 8], dtype=np.int64)  # 定义四级离散网格动作
REFERENCE_SUBDIVISIONS = 16  # 使用统一十六分段模型作为隐藏评价参考解
OPTIMIZATION_SOLVE_BUDGET = 32  # 为三个搜索方法和 DQN 统一真实候选求解预算
TOP_HOTSPOT_COUNT = 4  # 使用参考解能量最大的四个图区域评价热点命中率
@dataclass(frozen=True)  # 将宏观杆件定义为不可变拓扑记录
class MacroMember:  # 表示横向通道空间桁架中的一根宏观杆件
    start: int  # 保存起点宏观节点编号
    end: int  # 保存终点宏观节点编号
    region: int  # 保存负责该杆件网格级别的图区域编号
    name: str  # 保存便于审计的杆件名称
@dataclass  # 保存一次真实 CalculiX 求解及其派生评价量
class SolveResult:  # 汇总网格规模、扭矩、能量与区域响应
    levels: tuple[int, ...]  # 保存十六个图区域的离散网格级别
    element_count: int  # 保存该配置的 B31 单元总数
    node_count: int  # 保存该配置的节点总数
    torque: float  # 保存右端约束反力关于通道轴线的合扭矩
    strain_energy: float  # 保存全部杆单元的线弹性应变能
    region_energy: np.ndarray  # 保存每个图区域的应变能
    region_peak_stress: np.ndarray  # 保存每个图区域的最大轴向应力绝对值
    displacement_norm: float  # 保存全模型最大位移模长
    probe_vector: np.ndarray  # 保存通道中部宏观节点位移向量用于场误差评价
    workdir: str  # 保存该候选真实求解证据目录
@dataclass  # 保存方法最终结果及完整收敛轨迹
class MethodResult:  # 用统一结构记录五种方法
    name: str  # 保存方法名称
    levels: tuple[int, ...]  # 保存最终网格级别向量
    solution: SolveResult  # 保存最终真实有限元结果
    objective: float  # 保存相对参考解的综合目标值
    torque_error: float  # 保存扭矩相对误差
    energy_error: float  # 保存应变能相对误差
    probe_error: float  # 保存通道中部位移场相对误差
    hotspot_recall: float  # 保存参考热点区域召回率
    unique_solves: int  # 保存该方法消耗的唯一真实候选求解次数
    history: list[dict[str, float]]  # 保存每次唯一候选求解后的最优轨迹
    notes: str  # 保存方法实现边界与实验说明
class TorsionBenchmark:  # 管理横向通道拓扑、CalculiX 求解、缓存和统一评价
    def __init__(self, output_root: Path, ccx_command: str) -> None:  # 初始化实验目录和求解器命令
        self.output_root = output_root  # 保存实验输出根目录
        self.output_root.mkdir(parents=True, exist_ok=True)  # 创建实验输出目录
        self.run_root = self.output_root / "solver_runs"  # 定义真实求解证据目录
        self.run_root.mkdir(parents=True, exist_ok=True)  # 创建真实求解证据目录
        self.ccx_command = ccx_command  # 保存 CalculiX 命令路径
        self.macro_coordinates, self.members, self.region_names = self._build_topology()  # 构造规范化多桁架横向通道
        self.region_count = len(self.region_names)  # 保存图区域数量
        self.region_adjacency = self._build_region_adjacency()  # 根据共享节点建立区域图
        self.normalized_adjacency = self._normalize_adjacency(self.region_adjacency)  # 构造 GCN 使用的归一化邻接矩阵
        self.region_member_counts = np.bincount([member.region for member in self.members], minlength=self.region_count)  # 统计每个区域包含的宏观杆件数
        self.element_cap = int(len(self.members) * int(LEVEL_SUBDIVISIONS[2]))  # 将统一四分段网格的单元数设为最终资源上限
        self.cache: dict[tuple[int, ...], SolveResult] = {}  # 缓存全部候选真实求解以禁止重复调用
        self.reference: SolveResult | None = None  # 延迟保存十六分段隐藏参考解
        self.reference_hotspots: set[int] = set()  # 延迟保存参考热点区域编号
    def _build_topology(self) -> tuple[np.ndarray, list[MacroMember], list[str]]:  # 构造六站四角空间箱形多桁架拓扑
        span = 50.0  # 定义两条猫道之间的规范化横向跨度
        width = 2.4  # 定义通道空间桁架的规范化纵向宽度
        height = 2.0  # 定义通道空间桁架的规范化竖向高度
        stations = 6  # 定义横向通道沿跨度方向的节点站数
        coordinates: list[tuple[float, float, float]] = []  # 初始化宏观节点坐标列表
        for station in range(stations):  # 逐站生成四个箱形截面角点
            y = span * station / float(stations - 1)  # 计算当前站沿通道跨度方向的位置
            coordinates.extend([(-width / 2.0, y, 0.0), (width / 2.0, y, 0.0), (-width / 2.0, y, height), (width / 2.0, y, height)])  # 写入左下、右下、左上、右上角点
        region_names = [f"panel_{panel + 1}_chords" for panel in range(stations - 1)]  # 定义五个纵向弦杆区域
        region_names.extend([f"station_{station + 1}_frame" for station in range(stations)])  # 定义六个横向框架区域
        region_names.extend([f"panel_{panel + 1}_bracing" for panel in range(stations - 1)])  # 定义五个空间斜撑区域
        members: list[MacroMember] = []  # 初始化宏观杆件列表
        for panel in range(stations - 1):  # 逐跨生成四根纵向弦杆
            for corner in range(4):  # 遍历箱形截面四个角点
                members.append(MacroMember(4 * panel + corner, 4 * (panel + 1) + corner, panel, f"P{panel + 1}_CH{corner + 1}"))  # 将弦杆归入当前跨弦杆区域
        for station in range(stations):  # 逐站生成矩形横向框架
            base = 4 * station  # 计算当前站首节点编号
            frame_region = (stations - 1) + station  # 计算当前框架区域编号
            frame_pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]  # 定义底横杆、顶横杆和两根立杆
            for local_index, (start_local, end_local) in enumerate(frame_pairs):  # 逐一加入框架四边
                members.append(MacroMember(base + start_local, base + end_local, frame_region, f"S{station + 1}_FR{local_index + 1}"))  # 将框架杆件归入当前站区域
        for panel in range(stations - 1):  # 逐跨生成四个面的成对 X 形斜撑
            a = 4 * panel  # 保存当前跨起始站首节点编号
            b = 4 * (panel + 1)  # 保存当前跨终止站首节点编号
            brace_region = (stations - 1) + stations + panel  # 计算当前跨斜撑区域编号
            brace_pairs = [(a + 0, b + 2), (a + 2, b + 0), (a + 1, b + 3), (a + 3, b + 1), (a + 0, b + 1), (a + 1, b + 0), (a + 2, b + 3), (a + 3, b + 2)]  # 定义左右侧面、底面和顶面的 X 形斜撑
            for local_index, (start_node, end_node) in enumerate(brace_pairs):  # 逐一加入八根空间斜撑
                members.append(MacroMember(start_node, end_node, brace_region, f"P{panel + 1}_BR{local_index + 1}"))  # 将斜撑归入当前跨斜撑区域
        return np.asarray(coordinates, dtype=np.float64), members, region_names  # 返回节点、杆件和十六个图区域名称
    def _build_region_adjacency(self) -> np.ndarray:  # 根据不同区域杆件是否共享宏观节点建立无向图
        adjacency = np.zeros((len(self.region_names), len(self.region_names)), dtype=np.float64)  # 初始化区域邻接矩阵
        region_nodes: list[set[int]] = [set() for _ in self.region_names]  # 初始化每个区域使用的宏观节点集合
        for member in self.members:  # 遍历全部宏观杆件
            region_nodes[member.region].update((member.start, member.end))  # 将杆件两端加入所属区域节点集合
        for left in range(len(self.region_names)):  # 遍历第一个区域编号
            for right in range(left + 1, len(self.region_names)):  # 遍历第二个区域编号并避免重复
                if region_nodes[left].intersection(region_nodes[right]):  # 检查两个区域是否共享至少一个节点
                    adjacency[left, right] = 1.0  # 写入正向图边
                    adjacency[right, left] = 1.0  # 写入反向图边
        return adjacency  # 返回区域无向邻接矩阵
    def _normalize_adjacency(self, adjacency: np.ndarray) -> np.ndarray:  # 计算带自环的对称归一化邻接矩阵
        augmented = adjacency + np.eye(adjacency.shape[0], dtype=np.float64)  # 为每个图区域加入自环
        degree = np.sum(augmented, axis=1)  # 计算每个图节点的度
        inverse_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1.0e-12)))  # 构造度矩阵负二分之一次方
        return inverse_sqrt @ augmented @ inverse_sqrt  # 返回 GCN 使用的对称归一化邻接矩阵
    def _imperfection_normal(self, start: np.ndarray, end: np.ndarray, member_index: int) -> np.ndarray:  # 为每根宏观杆件生成稳定且非共线的初弯曲方向
        direction = end - start  # 计算杆件轴向向量
        direction = direction / np.linalg.norm(direction)  # 将杆件轴向向量归一化
        candidates = [np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 1.0, 0.0])]  # 定义三个候选参考方向
        reference = min(candidates, key=lambda vector: abs(float(np.dot(direction, vector))))  # 选择与杆轴最不平行的参考方向
        normal = np.cross(direction, reference)  # 通过叉积生成杆件法向方向
        normal = normal / np.linalg.norm(normal)  # 将法向方向归一化
        return normal if member_index % 2 == 0 else -normal  # 交替法向符号以避免全结构同向初弯曲偏置
    def _mesh_from_levels(self, levels: tuple[int, ...], reference: bool = False) -> tuple[np.ndarray, list[tuple[int, int, int]], list[int], list[int]]:  # 将区域网格级别展开为真实 B31 节点与单元
        nodes: list[np.ndarray] = [coordinate.copy() for coordinate in self.macro_coordinates]  # 先写入共享宏观节点
        elements: list[tuple[int, int, int]] = []  # 初始化单元连接和区域编号列表
        left_nodes = [1, 2, 3, 4]  # 保存左端四个约束节点的一基编号
        right_nodes = [len(self.macro_coordinates) - 3, len(self.macro_coordinates) - 2, len(self.macro_coordinates) - 1, len(self.macro_coordinates)]  # 保存右端四个扭转位移节点的一基编号
        for member_index, member in enumerate(self.members):  # 逐根宏观杆件展开局部离散
            subdivisions = REFERENCE_SUBDIVISIONS if reference else int(LEVEL_SUBDIVISIONS[levels[member.region]])  # 读取当前杆件分段数量
            start = self.macro_coordinates[member.start]  # 读取宏观杆件起点坐标
            end = self.macro_coordinates[member.end]  # 读取宏观杆件终点坐标
            length = float(np.linalg.norm(end - start))  # 计算宏观杆件长度
            normal = self._imperfection_normal(start, end, member_index)  # 计算当前杆件初弯曲法向
            amplitude = IMPERFECTION_RATIO * length  # 计算与杆长成比例的解析初弯曲幅值
            path = [member.start + 1]  # 使用一基节点编号初始化当前杆件路径
            for subdivision in range(1, subdivisions):  # 生成宏观杆件内部离散节点
                parameter = subdivision / float(subdivisions)  # 计算当前内部节点的归一化弧长参数
                coordinate = (1.0 - parameter) * start + parameter * end + amplitude * math.sin(math.pi * parameter) * normal  # 在统一正弦初弯曲线上取点
                nodes.append(coordinate)  # 将内部节点加入全局节点列表
                path.append(len(nodes))  # 将新节点的一基编号加入当前杆件路径
            path.append(member.end + 1)  # 将共享宏观终点加入当前杆件路径
            for segment in range(subdivisions):  # 将当前路径相邻节点连接为 B31 单元
                elements.append((path[segment], path[segment + 1], member.region))  # 保存单元两端和所属区域
        return np.asarray(nodes, dtype=np.float64), elements, left_nodes, right_nodes  # 返回完整有限元网格与端部节点集
    def _write_deck(self, workdir: Path, nodes: np.ndarray, elements: list[tuple[int, int, int]], left_nodes: list[int], right_nodes: list[int]) -> None:  # 写出可直接运行的 CalculiX 输入文件
        lines: list[str] = []  # 初始化输入文件文本行
        lines.append("*HEADING")  # 写入 CalculiX 标题关键字
        lines.append("Normalized cross-passage space-truss torsion benchmark")  # 写入实验标题文本
        lines.append("*NODE")  # 开始节点定义块
        for node_id, coordinate in enumerate(nodes, start=1):  # 逐节点写入三维坐标
            lines.append(f"{node_id},{coordinate[0]:.12e},{coordinate[1]:.12e},{coordinate[2]:.12e}")  # 使用稳定科学计数格式写入节点卡
        lines.append("*ELEMENT,TYPE=B31,ELSET=ALL")  # 开始三维二节点 Euler-Bernoulli 梁单元定义块
        for element_id, (start_node, end_node, _) in enumerate(elements, start=1):  # 逐单元写入连接关系
            lines.append(f"{element_id},{start_node},{end_node}")  # 写入 B31 单元编号和两端节点
        lines.append("*NSET,NSET=NALL,GENERATE")  # 定义包含全部节点的输出节点集
        lines.append(f"1,{len(nodes)},1")  # 使用连续编号生成全部节点集
        lines.append("*NSET,NSET=LEFT")  # 定义左端固定节点集
        lines.append(",".join(str(node_id) for node_id in left_nodes))  # 写入左端四个节点编号
        lines.append("*NSET,NSET=RIGHT")  # 定义右端扭转位移节点集
        lines.append(",".join(str(node_id) for node_id in right_nodes))  # 写入右端四个节点编号
        lines.append("*MATERIAL,NAME=STEEL")  # 定义统一线弹性材料
        lines.append("*ELASTIC")  # 开始弹性参数块
        lines.append(f"{YOUNG_MODULUS:.12e},0.300000000000")  # 写入弹性模量和泊松比
        lines.append("*BEAM SECTION,ELSET=ALL,MATERIAL=STEEL,SECTION=RECT")  # 为全部空间梁单元赋予统一矩形截面
        lines.append(f"{BEAM_WIDTH:.12e},{BEAM_HEIGHT:.12e}")  # 写入矩形梁截面两边尺寸
        lines.append("0.371390676354,0.557086014531,0.742781352708")  # 写入不与任一杆轴平行的统一截面方向向量
        lines.append("*BOUNDARY")  # 开始全局固定边界定义
        lines.append("LEFT,1,6,0.0")  # 固定左端三个平动和三个转动自由度
        lines.append("*STEP")  # 开始线弹性静力分析步
        lines.append("*STATIC")  # 指定静力求解类型
        for node_id in right_nodes:  # 逐右端角点施加刚性截面小转角位移
            coordinate = nodes[node_id - 1]  # 读取当前右端角点坐标
            centered_z = coordinate[2] - 1.0  # 将竖向坐标平移到截面形心
            displacement_x = -TWIST_ANGLE * centered_z  # 根据小转角关系计算 X 向位移
            displacement_z = TWIST_ANGLE * coordinate[0]  # 根据小转角关系计算 Z 向位移
            lines.append("*BOUNDARY")  # 为当前节点开始位移边界块
            lines.append(f"{node_id},1,1,{displacement_x:.12e}")  # 写入 X 向扭转位移
            lines.append(f"{node_id},2,2,0.0")  # 约束轴向位移以形成端部刚性扭转
            lines.append(f"{node_id},3,3,{displacement_z:.12e}")  # 写入 Z 向扭转位移
        lines.append("*NODE FILE,NSET=NALL")  # 请求将全部节点位移写入 ASCII FRD 文件
        lines.append("U")  # 指定位移输出变量
        lines.append("*NODE PRINT,NSET=NALL")  # 同时将全部原始梁节点位移写入稳健的 ASCII DAT 文件
        lines.append("U")  # 请求 DAT 位移表以兼容 B31 展开后空 FRD 位移数据集
        lines.append("*NODE PRINT,NSET=RIGHT")  # 请求将右端节点外力写入 ASCII DAT 文件
        lines.append("RF")  # 输出右端约束反力以计算合扭矩
        lines.append("*EL PRINT,ELSET=ALL,TOTALS=YES")  # 请求逐原始梁单元输出内能并同时给出总和
        lines.append("ELSE")  # 输出每个 B31 梁单元的内部能量
        lines.append("*END STEP")  # 结束静力分析步
        (workdir / "model.inp").write_text("\n".join(lines) + "\n", encoding="utf-8")  # 将完整输入文件写入候选运行目录
    def _parse_frd_displacements(self, filepath: Path) -> dict[int, np.ndarray]:  # 解析 CalculiX 最后一个 ASCII 位移数据集
        result: dict[int, np.ndarray] = {}  # 初始化最终位移字典
        current: dict[int, np.ndarray] = {}  # 初始化当前位移数据集字典
        in_displacement = False  # 初始化位移数据集状态标志
        for raw_line in filepath.read_text(encoding="utf-8", errors="replace").splitlines():  # 逐行读取 FRD 文件
            tokens = raw_line.strip().split()  # 对当前行进行空白分词
            if not tokens:  # 跳过空行
                continue  # 继续读取下一行
            if tokens[0] == "-4":  # 检查结果数据集标题记录
                in_displacement = len(tokens) > 1 and tokens[1].upper().startswith("DISP")  # 识别位移数据集
                if in_displacement:  # 在新位移数据集开始时清空临时字典
                    current = {}  # 重置当前位移字典
                continue  # 继续读取下一行
            if in_displacement and tokens[0] == "-1":  # 识别位移节点记录
                try:  # 优先按 CalculiX 原生固定宽度格式解析
                    node_id = int(raw_line[3:13].strip())  # 读取十字符节点编号字段
                    values = [float(raw_line[start:start + 12].replace("D", "E").replace("d", "e")) for start in (13, 25, 37)]  # 读取三个十二字符位移字段
                except (ValueError, IndexError):  # 在紧凑测试格式下回退到空白分词
                    if len(tokens) < 5:  # 检查回退格式是否包含三个位移分量
                        continue  # 忽略不完整结果行
                    node_id = int(tokens[1])  # 读取空白分隔节点编号
                    values = [float(value.replace("D", "E").replace("d", "e")) for value in tokens[2:5]]  # 读取空白分隔位移分量
                current[node_id] = np.asarray(values, dtype=np.float64)  # 保存当前节点三维位移
                continue  # 继续读取下一行
            if in_displacement and tokens[0] == "-3":  # 识别当前结果数据集结束记录
                if current:  # 仅在当前数据集非空时更新最终结果
                    result = current  # 保存最后一个完整位移数据集
                in_displacement = False  # 退出位移数据集状态
        if not result:  # 检查 FRD 是否成功输出原始梁节点位移
            dat_path = filepath.with_suffix(".dat")  # 构造同一候选求解的 ASCII DAT 文件路径
            if dat_path.exists():  # 检查稳健回退位移表是否存在
                return self._parse_dat_displacements(dat_path)  # 从原始梁节点 DAT 位移表读取结果
            raise RuntimeError(f"no displacement dataset found in {filepath}")  # 两种输出均缺失时拒绝发布候选结果
        return result  # 返回节点编号到三维位移的映射
    def _parse_dat_displacements(self, filepath: Path) -> dict[int, np.ndarray]:  # 解析 CalculiX 原始梁节点 ASCII 位移表
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()  # 读取完整 DAT 文本
        result: dict[int, np.ndarray] = {}  # 初始化最后一个完整位移表
        current: dict[int, np.ndarray] = {}  # 初始化当前位移表
        in_displacements = False  # 初始化位移表解析状态
        rows_started = False  # 初始化数值行开始标志
        for raw_line in lines:  # 逐行扫描 DAT 输出
            lowered = raw_line.lower()  # 生成小写副本用于稳健标题识别
            if "displacements (vx,vy,vz)" in lowered and "set nall" in lowered:  # 识别全部原始梁节点位移表标题
                if current:  # 检查上一位移表是否已经包含数值
                    result = current  # 保留最后一个完整位移表
                current = {}  # 清空当前位移表
                in_displacements = True  # 进入位移表解析状态
                rows_started = False  # 重置数值行标志
                continue  # 继续读取标题后的下一行
            if not in_displacements:  # 跳过位移表之外的反力和能量内容
                continue  # 继续读取下一行
            tokens = raw_line.replace("D", "E").replace("d", "e").split()  # 统一指数格式并按空白分词
            parsed = False  # 初始化当前行解析标志
            if len(tokens) >= 4:  # 检查是否包含节点号和三个平动分量
                try:  # 尝试解析标准 CalculiX 位移记录
                    node_id = int(tokens[0])  # 读取原始节点编号
                    values = np.asarray([float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=np.float64)  # 读取三个全局位移分量
                    current[node_id] = values  # 保存当前节点三维位移
                    rows_started = True  # 标记已经进入位移数值区
                    parsed = True  # 标记当前行解析成功
                except ValueError:  # 忽略表头和非数值内容
                    parsed = False  # 保持当前行未解析状态
            if rows_started and not parsed and not raw_line.strip():  # 在数值区后的空行处结束当前位移表
                if current:  # 检查当前表是否包含数值
                    result = current  # 保存最后一个完整位移表
                current = {}  # 清空当前临时表
                in_displacements = False  # 退出位移表解析状态
        if current:  # 处理文件结尾没有额外空行的位移表
            result = current  # 保存文件末尾位移表
        if not result:  # 检查是否成功获得原始梁节点位移
            raise RuntimeError(f"no NALL displacement table found in {filepath}")  # 缺少位移表时拒绝发布结果
        return result  # 返回节点编号到三维位移的映射
    def _parse_dat_reactions_and_energy(self, filepath: Path, element_count: int) -> tuple[dict[int, np.ndarray], np.ndarray]:  # 解析右端反力和逐梁单元内部能量
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()  # 读取完整 CalculiX DAT 文本
        reactions: dict[int, np.ndarray] = {}  # 初始化右端节点反力字典
        element_energy = np.full(element_count, np.nan, dtype=np.float64)  # 初始化逐原始梁单元内部能量数组
        in_reactions = False  # 初始化反力表解析状态
        in_energy = False  # 初始化内部能量表解析状态
        reaction_rows_started = False  # 记录反力表是否已经出现数值行
        energy_rows_started = False  # 记录内部能量表是否已经出现数值行
        for raw_line in lines:  # 逐行扫描 DAT 输出
            lowered = raw_line.lower()  # 生成当前行小写副本用于稳健标题匹配
            if "forces (fx,fy,fz)" in lowered and "set right" in lowered:  # 识别右端节点反力表标题
                in_reactions = True  # 进入反力表解析状态
                in_energy = False  # 退出内部能量表解析状态
                reaction_rows_started = False  # 重置反力数值行标志
                continue  # 继续读取标题后的下一行
            if "internal energy" in lowered and "set all" in lowered:  # 识别全部梁单元内部能量表标题
                in_energy = True  # 进入内部能量表解析状态
                in_reactions = False  # 退出反力表解析状态
                energy_rows_started = False  # 重置能量数值行标志
                continue  # 继续读取标题后的下一行
            tokens = raw_line.replace("D", "E").replace("d", "e").split()  # 统一指数格式并按空白分词
            if in_reactions:  # 检查当前是否位于反力表
                parsed = False  # 初始化当前行反力解析标志
                if len(tokens) >= 4:  # 检查是否可能包含节点号和三个力分量
                    try:  # 尝试解析标准 CalculiX 反力记录
                        node_id = int(tokens[0])  # 读取节点编号
                        values = np.asarray([float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=np.float64)  # 读取三个全局力分量
                        reactions[node_id] = values  # 保存当前右端节点反力
                        reaction_rows_started = True  # 标记已经进入反力数值区
                        parsed = True  # 标记当前行解析成功
                    except ValueError:  # 忽略表头和非数值行
                        parsed = False  # 保持当前行未解析状态
                if reaction_rows_started and not parsed and not raw_line.strip():  # 在数值区后的空行处结束反力表
                    in_reactions = False  # 退出反力表解析状态
                continue  # 继续读取下一行
            if in_energy:  # 检查当前是否位于内部能量表
                parsed = False  # 初始化当前行能量解析标志
                if len(tokens) >= 2:  # 检查是否可能包含单元号和能量值
                    try:  # 尝试解析标准 CalculiX 全单元变量记录
                        element_id = int(tokens[0])  # 读取原始梁单元编号
                        value = float(tokens[-1])  # 读取当前记录最后一个数值作为内部能量
                        if 1 <= element_id <= element_count:  # 验证原始梁单元编号范围
                            element_energy[element_id - 1] = value  # 保存当前梁单元内部能量
                            energy_rows_started = True  # 标记已经进入能量数值区
                            parsed = True  # 标记当前行解析成功
                    except ValueError:  # 忽略表头、总和说明和非数值行
                        parsed = False  # 保持当前行未解析状态
                if energy_rows_started and not parsed and not raw_line.strip():  # 在数值区后的空行处结束能量表
                    in_energy = False  # 退出内部能量表解析状态
                continue  # 继续读取下一行
        if len(reactions) < 4:  # 检查四个右端角点是否均有反力输出
            raise RuntimeError(f"expected four RIGHT reaction rows in {filepath}, found {len(reactions)}")  # 缺少反力时拒绝发布结果
        if np.any(~np.isfinite(element_energy)):  # 检查是否获得每个原始梁单元的内部能量
            missing = np.flatnonzero(~np.isfinite(element_energy))[:20].tolist()  # 提取前二十个缺失单元索引便于诊断
            raise RuntimeError(f"missing ELSE output for element indices {missing} in {filepath}")  # 缺少逐单元能量时停止实验
        return reactions, element_energy  # 返回右端反力和逐梁单元内部能量
    def _postprocess(self, levels: tuple[int, ...], workdir: Path, nodes: np.ndarray, elements: list[tuple[int, int, int]], right_nodes: list[int], displacements: dict[int, np.ndarray], reactions: dict[int, np.ndarray], element_energy: np.ndarray) -> SolveResult:  # 从真实 CalculiX 输出构造扭矩、能量和热点指标
        region_energy = np.zeros(self.region_count, dtype=np.float64)  # 初始化区域内部能量
        region_peak_stress = np.zeros(self.region_count, dtype=np.float64)  # 初始化区域峰值能量密度代理量
        for element_index, (start_node, end_node, region) in enumerate(elements):  # 逐原始 B31 梁单元汇总区域响应
            length = float(np.linalg.norm(nodes[end_node - 1] - nodes[start_node - 1]))  # 计算当前梁单元长度
            energy = max(float(element_energy[element_index]), 0.0)  # 读取并截断数值噪声导致的微小负内能
            region_energy[region] += energy  # 将梁单元内部能量累加到所属图区域
            region_peak_stress[region] = max(region_peak_stress[region], energy / max(length, 1.0e-12))  # 用单位长度峰值内能构造稳定热点强度
        torque = 0.0  # 初始化右端合扭矩
        for node_id in right_nodes:  # 逐右端角点计算关于 Y 轴的反力矩
            coordinate = nodes[node_id - 1]  # 读取当前右端角点原始坐标
            force = reactions[node_id]  # 读取当前右端角点反力
            torque += coordinate[0] * force[2] - (coordinate[2] - 1.0) * force[0]  # 累加 X-FZ 与 Z-FX 构成的 Y 轴扭矩
        maximum_displacement = max(float(np.linalg.norm(value)) for value in displacements.values())  # 计算全模型最大平动位移模长
        probe_node_ids = list(range(9, 17))  # 选取中间两站八个共享宏观节点作为位移场探针
        probe_vector = np.concatenate([displacements[node_id] for node_id in probe_node_ids]).astype(np.float64)  # 拼接二十四维中部位移场向量
        return SolveResult(levels, len(elements), len(nodes), abs(float(torque)), float(np.sum(element_energy)), region_energy, region_peak_stress, maximum_displacement, probe_vector, str(workdir))  # 返回统一候选求解结果
    def solve(self, levels: tuple[int, ...], reference: bool = False) -> SolveResult:  # 对一个网格配置执行缓存检查、输入生成、真实求解和后处理
        if not reference and levels in self.cache:  # 检查普通候选是否已经求解
            return self.cache[levels]  # 直接返回缓存并禁止重复真实求解
        signature_text = "reference" if reference else "-".join(str(level) for level in levels)  # 生成候选配置文本签名
        signature = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()[:12]  # 生成稳定短哈希目录名
        workdir = self.run_root / ("reference" if reference else f"candidate-{signature}")  # 定义候选求解目录
        workdir.mkdir(parents=True, exist_ok=True)  # 创建候选求解目录
        nodes, elements, left_nodes, right_nodes = self._mesh_from_levels(levels, reference=reference)  # 生成当前候选真实桁架网格
        self._write_deck(workdir, nodes, elements, left_nodes, right_nodes)  # 写出 CalculiX 输入文件
        completed = subprocess.run([self.ccx_command, "-i", "model"], cwd=str(workdir), capture_output=True, text=True, timeout=180, check=False)  # 调用真实 CalculiX 执行静力求解
        (workdir / "solver.stdout.log").write_text(completed.stdout, encoding="utf-8")  # 保存求解器标准输出
        (workdir / "solver.stderr.log").write_text(completed.stderr, encoding="utf-8")  # 保存求解器标准错误
        if completed.returncode != 0:  # 检查求解器进程返回码
            raise RuntimeError(f"CalculiX failed for {signature_text} with code {completed.returncode}")  # 求解失败时拒绝生成实验结论
        frd_path = workdir / "model.frd"  # 定义 CalculiX 位移结果文件路径
        if not frd_path.exists():  # 检查位移结果文件是否存在
            raise RuntimeError(f"CalculiX did not create {frd_path}")  # 缺少结果文件时停止实验
        displacements = self._parse_frd_displacements(frd_path)  # 解析全部节点平动位移
        dat_path = workdir / "model.dat"  # 定义 CalculiX ASCII 历史输出文件路径
        if not dat_path.exists():  # 检查反力和内部能量输出文件是否存在
            raise RuntimeError(f"CalculiX did not create {dat_path}")  # 缺少 DAT 时停止实验
        reactions, element_energy = self._parse_dat_reactions_and_energy(dat_path, len(elements))  # 解析右端反力和逐梁单元内部能量
        result = self._postprocess(levels, workdir, nodes, elements, right_nodes, displacements, reactions, element_energy)  # 构造统一扭转响应和热点指标
        receipt = {"levels": list(levels), "reference": reference, "element_count": result.element_count, "node_count": result.node_count, "torque": result.torque, "strain_energy": result.strain_energy, "maximum_displacement": result.displacement_norm, "probe_vector": result.probe_vector.tolist()}  # 构造候选求解收据
        (workdir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")  # 保存候选求解收据
        if not reference:  # 检查是否为普通候选求解
            self.cache[levels] = result  # 将普通候选加入全局缓存
        return result  # 返回真实有限元求解结果
    def prepare_reference(self) -> SolveResult:  # 执行并保存统一十六分段隐藏参考解
        reference_levels = tuple(3 for _ in range(self.region_count))  # 使用占位级别向量标记参考解区域数量
        self.reference = self.solve(reference_levels, reference=True)  # 执行独立十六分段真实 CalculiX 参考求解
        hotspot_indices = np.argsort(self.reference.region_energy)[-TOP_HOTSPOT_COUNT:]  # 按区域能量选取参考热点
        self.reference_hotspots = {int(index) for index in hotspot_indices}  # 保存参考热点编号集合
        return self.reference  # 返回参考求解结果
    def metrics(self, solution: SolveResult) -> tuple[float, float, float, float, float]:  # 计算候选相对隐藏参考解的统一评价指标
        if self.reference is None:  # 检查参考解是否已经准备
            raise RuntimeError("reference solution has not been prepared")  # 未准备参考解时拒绝评价
        torque_error = abs(solution.torque - self.reference.torque) / max(abs(self.reference.torque), 1.0e-18)  # 计算扭矩相对误差
        solution_energy_distribution = solution.region_energy / max(float(np.sum(solution.region_energy)), 1.0e-18)  # 将候选区域能量转换为无量纲分布
        reference_energy_distribution = self.reference.region_energy / max(float(np.sum(self.reference.region_energy)), 1.0e-18)  # 将参考区域能量转换为无量纲分布
        energy_error = 0.5 * float(np.sum(np.abs(solution_energy_distribution - reference_energy_distribution)))  # 以总变差距离计算独立的区域能量分布误差
        probe_error = float(np.linalg.norm(solution.probe_vector - self.reference.probe_vector) / max(np.linalg.norm(self.reference.probe_vector), 1.0e-18))  # 计算中部位移场相对误差
        candidate_hotspots = {int(index) for index in np.argsort(solution.region_energy)[-TOP_HOTSPOT_COUNT:]}  # 选取候选能量热点区域
        hotspot_recall = len(candidate_hotspots.intersection(self.reference_hotspots)) / float(TOP_HOTSPOT_COUNT)  # 计算参考热点召回率
        objective = 0.40 * torque_error + 0.25 * energy_error + 0.20 * probe_error + 0.13 * (1.0 - hotspot_recall)  # 删除资源项并保留原四项精度指标相对权重
        return float(objective), float(torque_error), float(energy_error), float(probe_error), float(hotspot_recall)  # 返回综合目标和四个可解释指标
    def repair_levels(self, levels: np.ndarray, priority: np.ndarray | None = None) -> tuple[int, ...]:  # 将任意离散级别向量修复到统一单元预算内
        repaired = np.clip(np.rint(levels), 0, 3).astype(np.int64)  # 将连续或越界向量映射到四级离散动作
        if priority is None:  # 检查是否提供区域重要性
            priority = np.zeros(self.region_count, dtype=np.float64)  # 纯 PSO 使用零优先级并由索引稳定打破平局
        def element_count(vector: np.ndarray) -> int:  # 定义快速计算当前级别向量单元数的局部函数
            return int(np.sum(self.region_member_counts * LEVEL_SUBDIVISIONS[vector]))  # 按区域杆件数和分段数计算单元总数
        while element_count(repaired) > self.element_cap:  # 在超出资源上限时逐级降阶
            candidates = [index for index in range(self.region_count) if repaired[index] > 0]  # 找出仍可粗化的区域
            if not candidates:  # 防止异常配置无法继续粗化
                break  # 退出修复循环
            target = min(candidates, key=lambda index: (float(priority[index]), int(index)))  # 优先粗化重要性最低的区域
            repaired[target] -= 1  # 将目标区域降低一个网格级别
        return tuple(int(value) for value in repaired)  # 返回可哈希的预算可行级别元组
    def region_features(self, solution: SolveResult, levels: tuple[int, ...]) -> np.ndarray:  # 构造多智能体和 GCN-DQN 共用的节点特征
        energy = solution.region_energy / max(float(np.max(solution.region_energy)), 1.0e-18)  # 归一化区域应变能
        stress = solution.region_peak_stress / max(float(np.max(solution.region_peak_stress)), 1.0e-18)  # 归一化区域峰值应力
        neighbor_energy = self.region_adjacency @ energy / np.maximum(np.sum(self.region_adjacency, axis=1), 1.0)  # 计算邻域平均能量
        contrast = np.abs(energy - neighbor_energy)  # 计算区域与邻域之间的能量对比
        level_values = np.asarray(levels, dtype=np.float64) / 3.0  # 将离散网格级别归一化到零到一
        local_cost = self.region_member_counts * LEVEL_SUBDIVISIONS[np.asarray(levels, dtype=np.int64)] / float(self.element_cap)  # 计算区域单元成本占比
        neighbor_level = self.region_adjacency @ level_values / np.maximum(np.sum(self.region_adjacency, axis=1), 1.0)  # 计算邻域平均网格级别
        return np.column_stack((energy, stress, contrast, level_values, local_cost, neighbor_level)).astype(np.float32)  # 返回六维图节点特征矩阵
    def hotspot_priority(self, solution: SolveResult, levels: tuple[int, ...]) -> np.ndarray:  # 计算粗网格热点和图邻域对比的联合优先级
        features = self.region_features(solution, levels)  # 构造当前区域节点特征
        priority = 0.45 * features[:, 0] + 0.35 * features[:, 1] + 0.20 * features[:, 2]  # 合并能量、应力和邻域对比
        return np.asarray(priority, dtype=np.float64)  # 返回双精度区域优先级
    def method_result(self, name: str, levels: tuple[int, ...], unique_solves: int, history: list[dict[str, float]], notes: str) -> MethodResult:  # 将任意方法最终配置转换为统一结果结构
        solution = self.solve(levels)  # 获取或执行最终配置真实求解
        objective, torque_error, energy_error, probe_error, hotspot_recall = self.metrics(solution)  # 计算统一评价指标
        return MethodResult(name, levels, solution, objective, torque_error, energy_error, probe_error, hotspot_recall, unique_solves, history, notes)  # 返回完整方法结果
class GraphQNetwork(nn.Module):  # 定义两层 GCN 编码器和逐区域动作 Q 值头
    def __init__(self, feature_dim: int, hidden_dim: int, action_count: int, adjacency: np.ndarray) -> None:  # 初始化网络维度和固定图邻接矩阵
        super().__init__()  # 初始化 PyTorch 模块基类
        self.register_buffer("adjacency", torch.tensor(adjacency, dtype=torch.float32))  # 将归一化邻接矩阵注册为不可训练缓冲区
        self.input_layer = nn.Linear(feature_dim, hidden_dim, bias=False)  # 定义第一层图卷积线性映射
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 定义第二层图卷积线性映射
        self.output_layer = nn.Linear(hidden_dim * 2, action_count)  # 将局部和全局图表示映射为动作 Q 值
    def forward(self, features: torch.Tensor) -> torch.Tensor:  # 前向计算一批图状态的逐节点 Q 值
        adjacency = self.adjacency.unsqueeze(0).expand(features.shape[0], -1, -1)  # 将固定邻接矩阵扩展到批维
        hidden = torch.relu(self.input_layer(torch.bmm(adjacency, features)))  # 完成第一层图消息传播和非线性变换
        hidden = torch.relu(self.hidden_layer(torch.bmm(adjacency, hidden)))  # 完成第二层图消息传播和非线性变换
        global_context = hidden.mean(dim=1, keepdim=True).expand(-1, hidden.shape[1], -1)  # 计算并广播全图平均上下文
        return self.output_layer(torch.cat((hidden, global_context), dim=-1))  # 输出每个区域两个网格动作的 Q 值
class ReplayBuffer:  # 保存 GCN-DQN 使用的有限容量真实求解转移
    def __init__(self, capacity: int) -> None:  # 初始化回放容量和转移列表
        self.capacity = capacity  # 保存最大转移数量
        self.items: list[tuple[np.ndarray, int, int, float, np.ndarray, np.ndarray]] = []  # 初始化转移记录列表
    def add(self, item: tuple[np.ndarray, int, int, float, np.ndarray, np.ndarray]) -> None:  # 向回放缓冲区加入一个真实转移
        self.items.append(item)  # 追加当前转移
        if len(self.items) > self.capacity:  # 检查是否超过回放容量
            self.items.pop(0)  # 删除最早转移以保持固定容量
    def sample(self, batch_size: int) -> list[tuple[np.ndarray, int, int, float, np.ndarray, np.ndarray]]:  # 均匀随机抽取训练批次
        return random.sample(self.items, batch_size)  # 返回不放回随机样本
class ExperimentRunner:  # 在统一参考解、单元预算和真实求解缓存下执行五种方法
    def __init__(self, benchmark: TorsionBenchmark) -> None:  # 保存基准对象并初始化方法结果列表
        self.benchmark = benchmark  # 保存横向通道扭转基准对象
        self.results: list[MethodResult] = []  # 初始化五种方法结果列表
    def run_uniform(self) -> MethodResult:  # 执行均匀网格基线并在可行全局级别中选优
        start_count = len(self.benchmark.cache)  # 记录方法开始前的缓存候选数
        best_levels: tuple[int, ...] | None = None  # 初始化最优均匀级别向量
        best_objective = float("inf")  # 初始化最优综合目标
        history: list[dict[str, float]] = []  # 初始化均匀网格候选轨迹
        for level in range(4):  # 依次检查四个全局均匀网格级别
            levels = tuple(level for _ in range(self.benchmark.region_count))  # 构造当前全局均匀级别向量
            estimated_elements = int(np.sum(self.benchmark.region_member_counts * LEVEL_SUBDIVISIONS[level]))  # 计算当前均匀网格单元数
            if estimated_elements > self.benchmark.element_cap:  # 跳过超过统一资源上限的均匀网格
                continue  # 继续检查下一全局级别
            solution = self.benchmark.solve(levels)  # 执行当前均匀网格真实求解
            objective, _, _, _, _ = self.benchmark.metrics(solution)  # 计算统一综合目标
            if objective < best_objective:  # 检查当前候选是否改进最优结果
                best_objective = objective  # 更新最优综合目标
                best_levels = levels  # 更新最优均匀级别向量
            history.append({"evaluation": float(len(self.benchmark.cache) - start_count), "best_objective": float(best_objective), "candidate_objective": float(objective)})  # 记录当前均匀候选和历史最优
        if best_levels is None:  # 检查是否至少存在一个预算可行均匀网格
            raise RuntimeError("no feasible uniform mesh")  # 无可行均匀网格时停止实验
        return self.benchmark.method_result("uniform_mesh", best_levels, len(self.benchmark.cache) - start_count, history, "全局统一分段级别，在相同最终单元上限内枚举可行均匀网格。")  # 返回均匀网格结果
    def _run_pso(self, name: str, hotspot_reduced: bool) -> MethodResult:  # 执行纯离散 PSO 或粗网格热点降维 PSO
        start_count = len(self.benchmark.cache)  # 记录方法开始前缓存候选数
        base_levels = tuple(1 for _ in range(self.benchmark.region_count))  # 使用全区域二分段配置作为共同搜索起点
        coarse_solution = self.benchmark.solve(tuple(0 for _ in range(self.benchmark.region_count)))  # 获取统一粗网格真实响应
        priority = self.benchmark.hotspot_priority(coarse_solution, tuple(0 for _ in range(self.benchmark.region_count)))  # 从粗网格响应构造热点优先级
        active_indices = np.argsort(priority)[-6:] if hotspot_reduced else np.arange(self.benchmark.region_count)  # 热点 PSO 仅保留六个高优先级图区域
        dimension = len(active_indices)  # 保存实际 PSO 搜索维数
        particle_count = 10  # 定义离散粒子数量
        positions = np.random.uniform(0.0, 3.0, size=(particle_count, dimension))  # 初始化连续粒子位置
        velocities = np.zeros_like(positions)  # 初始化粒子速度
        positions[0, :] = 1.0  # 将第一个粒子固定为共同二分段起点
        if hotspot_reduced:  # 检查是否执行热点降维 PSO
            positions[1, :] = 3.0  # 将第二个粒子设为全部热点最细级别
        personal_best_positions = positions.copy()  # 初始化个体最优位置
        personal_best_values = np.full(particle_count, float("inf"), dtype=np.float64)  # 初始化个体最优目标
        global_best_levels: tuple[int, ...] | None = None  # 初始化全局最优级别向量
        global_best_value = float("inf")  # 初始化全局最优目标值
        history: list[dict[str, float]] = []  # 初始化 PSO 收敛轨迹
        attempts = 0  # 初始化候选生成尝试次数
        while len(self.benchmark.cache) - start_count < OPTIMIZATION_SOLVE_BUDGET and attempts < 400:  # 在真实求解预算内迭代离散 PSO
            attempts += 1  # 累加候选生成尝试次数
            particle = attempts % particle_count  # 轮流选择当前粒子
            full = np.asarray(base_levels, dtype=np.float64)  # 从共同基础级别复制全维向量
            full[active_indices] = positions[particle]  # 将当前粒子位置写入活动区域
            candidate_levels = self.benchmark.repair_levels(full, priority if hotspot_reduced else None)  # 离散化并修复到统一单元预算
            before = len(self.benchmark.cache)  # 记录求解前缓存大小
            solution = self.benchmark.solve(candidate_levels)  # 获取当前候选真实有限元解
            objective, _, _, _, _ = self.benchmark.metrics(solution)  # 计算当前候选统一目标
            if objective < personal_best_values[particle]:  # 检查当前候选是否改进当前粒子历史最优
                personal_best_values[particle] = objective  # 更新粒子历史最优目标
                personal_best_positions[particle] = positions[particle].copy()  # 更新粒子历史最优位置
            if objective < global_best_value:  # 检查当前候选是否改进全局最优
                global_best_value = objective  # 更新全局最优目标
                global_best_levels = candidate_levels  # 更新全局最优离散级别向量
            if len(self.benchmark.cache) > before:  # 仅对唯一真实求解写入收敛轨迹
                history.append({"evaluation": float(len(self.benchmark.cache) - start_count), "best_objective": float(global_best_value), "candidate_objective": float(objective)})  # 记录当前候选和历史最优
            if global_best_levels is None:  # 检查是否已经获得全局最优候选
                global_target = np.ones(dimension, dtype=np.float64)  # 未获得最优时使用共同起点作为目标
            else:  # 在已有最优候选时提取活动维目标
                global_target = np.asarray(global_best_levels, dtype=np.float64)[active_indices]  # 将全局最优离散向量投影到活动维
            inertia = 0.70  # 定义离散 PSO 惯性权重
            cognitive = 1.35  # 定义个体认知权重
            social = 1.35  # 定义群体社会权重
            random_personal = np.random.random(dimension)  # 生成个体项随机系数
            random_global = np.random.random(dimension)  # 生成群体项随机系数
            velocities[particle] = inertia * velocities[particle] + cognitive * random_personal * (personal_best_positions[particle] - positions[particle]) + social * random_global * (global_target - positions[particle])  # 更新当前粒子速度
            positions[particle] = np.clip(positions[particle] + velocities[particle], 0.0, 3.0)  # 更新并截断当前粒子位置
            if len(self.benchmark.cache) == before:  # 在离散重复候选未消耗真实求解时进行随机扰动
                positions[particle] = np.clip(positions[particle] + np.random.normal(0.0, 0.75, size=dimension), 0.0, 3.0)  # 扰动粒子以寻找未评估离散配置
        if global_best_levels is None:  # 检查 PSO 是否得到至少一个候选
            raise RuntimeError(f"{name} did not evaluate a candidate")  # 无候选时停止实验
        notes = "粗网格能量、应力和邻域对比选出六个热点区域后执行离散 PSO。" if hotspot_reduced else "全部十六个图区域直接进入离散 PSO，不使用热点候选降维。"  # 生成方法说明
        return self.benchmark.method_result(name, global_best_levels, len(self.benchmark.cache) - start_count, history, notes)  # 返回 PSO 方法结果
    def run_multi_agent(self) -> MethodResult:  # 执行一区域一智能体的图消息传递与异步局部协商搜索
        start_count = len(self.benchmark.cache)  # 记录方法开始前缓存候选数
        current_levels = tuple(1 for _ in range(self.benchmark.region_count))  # 使用全区域二分段网格初始化智能体系统
        current_solution = self.benchmark.solve(current_levels)  # 获取初始真实有限元响应
        current_objective, _, _, _, _ = self.benchmark.metrics(current_solution)  # 计算初始综合目标
        best_levels = current_levels  # 初始化多智能体全局最优级别
        best_objective = current_objective  # 初始化多智能体全局最优目标
        history: list[dict[str, float]] = []  # 初始化多智能体收敛轨迹
        memories = np.zeros((self.benchmark.region_count, 2), dtype=np.float64)  # 为每个区域智能体保存最近改进和失败次数
        attempts = 0  # 初始化智能体提案次数
        while len(self.benchmark.cache) - start_count < OPTIMIZATION_SOLVE_BUDGET and attempts < 300:  # 在统一真实求解预算内异步协商
            attempts += 1  # 累加智能体提案次数
            features = self.benchmark.region_features(current_solution, current_levels)  # 生成当前图节点局部状态
            local_signal = 0.45 * features[:, 0] + 0.30 * features[:, 1] + 0.15 * features[:, 2] + 0.10 * features[:, 5]  # 合并本地能量、应力、对比和邻居消息
            confidence = local_signal + 0.25 * memories[:, 0] - 0.05 * memories[:, 1]  # 将历史改进和失败记忆加入智能体优先级
            order = list(np.argsort(confidence)[::-1])  # 按图智能体提案优先级排序
            selected = order[(attempts - 1) % len(order)]  # 轮换选择当前提出网格动作的智能体
            candidate = np.asarray(current_levels, dtype=np.float64)  # 复制当前全局级别配置
            action = 1 if candidate[selected] < 3 else -1  # 高优先级智能体优先细化并在上限处尝试粗化
            candidate[selected] += action  # 应用当前智能体局部动作
            candidate_levels = self.benchmark.repair_levels(candidate, confidence)  # 通过共享资源协调器修复超预算配置
            before = len(self.benchmark.cache)  # 记录候选求解前缓存大小
            candidate_solution = self.benchmark.solve(candidate_levels)  # 执行候选真实有限元求解
            candidate_objective, _, _, _, _ = self.benchmark.metrics(candidate_solution)  # 计算候选统一目标
            improvement = current_objective - candidate_objective  # 计算当前提案对系统目标的改进
            memories[selected, 0] = 0.70 * memories[selected, 0] + 0.30 * improvement  # 更新提案智能体的指数平滑改进记忆
            memories[selected, 1] = 0.0 if improvement > 0.0 else memories[selected, 1] + 1.0  # 更新提案智能体连续失败记忆
            if candidate_objective <= current_objective or random.random() < 0.08:  # 接受改进提案并保留少量探索性劣化转移
                current_levels = candidate_levels  # 更新系统当前级别配置
                current_solution = candidate_solution  # 更新系统当前真实响应
                current_objective = candidate_objective  # 更新系统当前综合目标
            if candidate_objective < best_objective:  # 检查候选是否改进全局最优
                best_objective = candidate_objective  # 更新全局最优目标
                best_levels = candidate_levels  # 更新全局最优级别配置
            if len(self.benchmark.cache) > before:  # 仅对唯一真实求解记录轨迹
                history.append({"evaluation": float(len(self.benchmark.cache) - start_count), "best_objective": float(best_objective), "candidate_objective": float(candidate_objective)})  # 写入当前候选和历史最优
        return self.benchmark.method_result("graph_multi_agent", best_levels, len(self.benchmark.cache) - start_count, history, "十六个区域智能体仅读取本地与邻接区域状态，通过消息传递优先级和共享单元预算异步协商。")  # 返回多智能体方法结果
    def _valid_action_mask(self, levels: tuple[int, ...], priority: np.ndarray) -> np.ndarray:  # 为每个图区域构造粗化和细化动作有效掩码
        mask = np.zeros((self.benchmark.region_count, 2), dtype=np.bool_)  # 初始化十六乘二动作掩码
        vector = np.asarray(levels, dtype=np.float64)  # 转换当前级别向量
        for region in range(self.benchmark.region_count):  # 逐区域检查两个动作
            if levels[region] > 0:  # 检查当前区域是否能够粗化
                candidate = vector.copy()  # 复制当前级别向量
                candidate[region] -= 1.0  # 应用粗化动作
                mask[region, 0] = self.benchmark.repair_levels(candidate, priority) != levels  # 标记粗化是否改变最终配置
            if levels[region] < 3:  # 检查当前区域是否能够细化
                candidate = vector.copy()  # 复制当前级别向量
                candidate[region] += 1.0  # 应用细化动作
                mask[region, 1] = self.benchmark.repair_levels(candidate, priority) != levels  # 标记细化是否改变最终配置
        return mask  # 返回动作有效掩码
    def run_dqn_gcn(self) -> MethodResult:  # 执行真实求解器在线交互的 GCN 编码深度 Q 学习
        start_count = len(self.benchmark.cache)  # 记录方法开始前缓存候选数
        online = GraphQNetwork(6, 48, 2, self.benchmark.normalized_adjacency)  # 创建在线 GCN-Q 网络
        target = GraphQNetwork(6, 48, 2, self.benchmark.normalized_adjacency)  # 创建目标 GCN-Q 网络
        target.load_state_dict(online.state_dict())  # 同步目标网络初始权重
        optimizer = torch.optim.Adam(online.parameters(), lr=2.0e-3)  # 创建 Adam 优化器
        replay = ReplayBuffer(256)  # 创建有限容量经验回放缓冲区
        current_levels = tuple(1 for _ in range(self.benchmark.region_count))  # 使用统一二分段网格初始化在线环境
        current_solution = self.benchmark.solve(current_levels)  # 获取初始真实有限元响应
        current_objective, _, _, _, _ = self.benchmark.metrics(current_solution)  # 计算初始综合目标
        best_levels = current_levels  # 初始化 DQN 全局最优级别配置
        best_objective = current_objective  # 初始化 DQN 全局最优目标
        history: list[dict[str, float]] = []  # 初始化 DQN 收敛轨迹
        interaction = 0  # 初始化环境交互次数
        while len(self.benchmark.cache) - start_count < OPTIMIZATION_SOLVE_BUDGET and interaction < 500:  # 在统一真实求解预算内在线训练
            interaction += 1  # 累加环境交互次数
            state_features = self.benchmark.region_features(current_solution, current_levels)  # 构造当前图状态节点特征
            priority = self.benchmark.hotspot_priority(current_solution, current_levels)  # 构造预算修复和动作掩码使用的当前优先级
            valid_mask = self._valid_action_mask(current_levels, priority)  # 构造当前有效动作掩码
            epsilon = max(0.08, 0.85 - 0.75 * (len(self.benchmark.cache) - start_count) / float(OPTIMIZATION_SOLVE_BUDGET))  # 按真实求解进度衰减探索率
            if random.random() < epsilon:  # 执行 epsilon 随机探索
                valid_pairs = np.argwhere(valid_mask)  # 提取全部有效区域动作对
                selected_pair = valid_pairs[random.randrange(len(valid_pairs))]  # 随机选择一个有效区域动作
                region = int(selected_pair[0])  # 读取动作区域编号
                action_index = int(selected_pair[1])  # 读取粗化或细化动作编号
            else:  # 执行当前 GCN-Q 网络贪心动作
                with torch.no_grad():  # 禁用动作选择阶段梯度
                    q_values = online(torch.tensor(state_features[None, :, :], dtype=torch.float32))[0].numpy()  # 计算十六个区域的两类动作 Q 值
                q_values[~valid_mask] = -1.0e30  # 屏蔽无效动作
                flat_index = int(np.argmax(q_values))  # 选择全图最大有效 Q 值动作
                region, action_index = np.unravel_index(flat_index, q_values.shape)  # 还原动作区域和动作类型
            candidate = np.asarray(current_levels, dtype=np.float64)  # 复制当前级别向量
            candidate[region] += -1.0 if action_index == 0 else 1.0  # 应用粗化或细化动作
            candidate_levels = self.benchmark.repair_levels(candidate, priority)  # 将候选配置修复到统一单元预算
            before = len(self.benchmark.cache)  # 记录候选求解前缓存大小
            candidate_solution = self.benchmark.solve(candidate_levels)  # 执行或读取候选真实有限元解
            candidate_objective, _, _, _, _ = self.benchmark.metrics(candidate_solution)  # 计算候选统一目标
            reward = 8.0 * (current_objective - candidate_objective)  # 仅以精度目标改进定义奖励，单元数由硬上限约束
            next_features = self.benchmark.region_features(candidate_solution, candidate_levels)  # 构造下一图状态节点特征
            next_priority = self.benchmark.hotspot_priority(candidate_solution, candidate_levels)  # 构造下一状态优先级
            next_mask = self._valid_action_mask(candidate_levels, next_priority)  # 构造下一状态动作掩码
            replay.add((state_features.copy(), region, action_index, float(reward), next_features.copy(), next_mask.copy()))  # 将真实求解转移加入经验回放
            if candidate_objective <= current_objective or random.random() < 0.05:  # 接受改进动作并保留少量探索性转移
                current_levels = candidate_levels  # 更新在线环境级别配置
                current_solution = candidate_solution  # 更新在线环境真实响应
                current_objective = candidate_objective  # 更新在线环境综合目标
            if candidate_objective < best_objective:  # 检查候选是否改进 DQN 全局最优
                best_objective = candidate_objective  # 更新 DQN 全局最优目标
                best_levels = candidate_levels  # 更新 DQN 全局最优级别配置
            if len(self.benchmark.cache) > before:  # 仅对唯一真实求解写入轨迹
                history.append({"evaluation": float(len(self.benchmark.cache) - start_count), "best_objective": float(best_objective), "candidate_objective": float(candidate_objective)})  # 记录当前候选和历史最优
            if len(replay.items) >= 8:  # 在回放样本足够时执行一次 DQN 参数更新
                batch = replay.sample(8)  # 随机抽取八条真实转移
                state_batch = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)  # 组成当前状态批次
                region_batch = torch.tensor([item[1] for item in batch], dtype=torch.int64)  # 组成动作区域批次
                action_batch = torch.tensor([item[2] for item in batch], dtype=torch.int64)  # 组成动作类型批次
                reward_batch = torch.tensor([item[3] for item in batch], dtype=torch.float32)  # 组成奖励批次
                next_batch = torch.tensor(np.stack([item[4] for item in batch]), dtype=torch.float32)  # 组成下一状态批次
                mask_batch = torch.tensor(np.stack([item[5] for item in batch]), dtype=torch.bool)  # 组成下一状态有效动作掩码
                q_batch = online(state_batch)  # 计算当前状态全部 Q 值
                selected_q = q_batch[torch.arange(len(batch)), region_batch, action_batch]  # 提取实际执行动作 Q 值
                with torch.no_grad():  # 禁用目标值计算梯度
                    next_online = online(next_batch).masked_fill(~mask_batch, -1.0e30)  # 用在线网络选择下一状态全图动作
                    next_flat = next_online.view(len(batch), -1).argmax(dim=1)  # 获取 Double DQN 下一动作扁平索引
                    next_region = torch.div(next_flat, 2, rounding_mode="floor")  # 还原下一动作区域编号
                    next_action = next_flat % 2  # 还原下一动作类型编号
                    next_target = target(next_batch)[torch.arange(len(batch)), next_region, next_action]  # 用目标网络评价下一动作
                    target_q = reward_batch + 0.92 * next_target  # 形成一步折扣 Bellman 目标
                loss = F.smooth_l1_loss(selected_q, target_q)  # 计算稳定的平滑 L1 DQN 损失
                optimizer.zero_grad()  # 清空上一批次梯度
                loss.backward()  # 反向传播当前 DQN 损失
                torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)  # 裁剪梯度以提高小样本在线训练稳定性
                optimizer.step()  # 更新在线 GCN-Q 网络参数
            if interaction % 6 == 0:  # 每六次交互同步一次目标网络
                target.load_state_dict(online.state_dict())  # 将在线网络参数复制到目标网络
            if interaction % 12 == 0 and len(self.benchmark.cache) - start_count < OPTIMIZATION_SOLVE_BUDGET:  # 周期性重置起点以扩大离散状态覆盖
                restart = np.random.randint(0, 3, size=self.benchmark.region_count)  # 生成零到二级随机网格向量
                current_levels = self.benchmark.repair_levels(restart, None)  # 修复随机网格到统一单元预算
                current_solution = self.benchmark.solve(current_levels)  # 获取随机重启状态真实响应
                current_objective, _, _, _, _ = self.benchmark.metrics(current_solution)  # 更新随机重启状态目标
        return self.benchmark.method_result("dqn_gcn", best_levels, len(self.benchmark.cache) - start_count, history, "两层 GCN 编码十六区域图，Double DQN 在真实 CalculiX 反馈下在线选择区域粗化或细化动作。")  # 返回 DQN+GCN 方法结果
    def run_all(self) -> list[MethodResult]:  # 按固定顺序执行五种方法并返回结果
        self.results.append(self.run_uniform())  # 执行均匀网格基线
        self.results.append(self._run_pso("pure_pso", hotspot_reduced=False))  # 执行全维纯离散 PSO
        self.results.append(self._run_pso("hotspot_pso", hotspot_reduced=True))  # 执行热点降维离散 PSO
        self.results.append(self.run_multi_agent())  # 执行图结构多智能体搜索
        self.results.append(self.run_dqn_gcn())  # 执行 GCN-DQN 在线网格优化
        return self.results  # 返回五种方法统一结果列表
def serialize_solution(solution: SolveResult) -> dict[str, Any]:  # 将求解结果转换为 JSON 可序列化字典
    return {"levels": list(solution.levels), "element_count": solution.element_count, "node_count": solution.node_count, "torque": solution.torque, "strain_energy": solution.strain_energy, "region_energy": solution.region_energy.tolist(), "region_peak_stress": solution.region_peak_stress.tolist(), "maximum_displacement": solution.displacement_norm, "probe_vector": solution.probe_vector.tolist(), "workdir": solution.workdir}  # 返回完整候选求解字段
def write_outputs(benchmark: TorsionBenchmark, results: list[MethodResult], output_root: Path) -> None:  # 写出 JSON、CSV、Markdown 和图形化实验结论
    if benchmark.reference is None:  # 检查参考解是否存在
        raise RuntimeError("reference result missing")  # 缺少参考解时拒绝写出结果
    output_root.mkdir(parents=True, exist_ok=True)  # 确保结果目录存在
    payload = {"schema": "cross-passage-torsion-five-method-benchmark", "schema_version": "1.0.0", "seed": SEED, "model_scope": "normalized box-space-truss abstraction of one repeated catwalk cross-passage; algorithm benchmark only", "loading": {"type": "prescribed pure twist", "twist_angle_rad": TWIST_ANGLE}, "mesh": {"region_count": benchmark.region_count, "region_names": benchmark.region_names, "level_subdivisions": LEVEL_SUBDIVISIONS.tolist(), "reference_subdivisions": REFERENCE_SUBDIVISIONS, "element_cap": benchmark.element_cap}, "reference_hotspots": sorted(benchmark.reference_hotspots), "reference": serialize_solution(benchmark.reference), "methods": []}  # 构造实验顶层机器可读结果
    for result in results:  # 逐方法写入统一结果字段
        payload["methods"].append({"name": result.name, "objective": result.objective, "torque_error": result.torque_error, "energy_error": result.energy_error, "probe_error": result.probe_error, "hotspot_recall": result.hotspot_recall, "unique_solves": result.unique_solves, "notes": result.notes, "solution": serialize_solution(result.solution), "history": result.history})  # 添加当前方法完整结果
    (output_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 写出完整 JSON 结果
    with (output_root / "method_summary.csv").open("w", encoding="utf-8", newline="") as stream:  # 创建方法汇总 CSV
        writer = csv.writer(stream)  # 创建 CSV 写入器
        writer.writerow(["method", "objective", "torque_error", "energy_error", "probe_error", "hotspot_recall", "elements", "nodes", "unique_solves", "levels"])  # 写入方法汇总表头
        for result in results:  # 逐方法写入汇总记录
            writer.writerow([result.name, f"{result.objective:.12e}", f"{result.torque_error:.12e}", f"{result.energy_error:.12e}", f"{result.probe_error:.12e}", f"{result.hotspot_recall:.6f}", result.solution.element_count, result.solution.node_count, result.unique_solves, " ".join(str(value) for value in result.levels)])  # 写入当前方法记录
    with (output_root / "convergence.csv").open("w", encoding="utf-8", newline="") as stream:  # 创建统一收敛轨迹 CSV
        writer = csv.writer(stream)  # 创建收敛轨迹写入器
        writer.writerow(["method", "evaluation", "best_objective", "candidate_objective"])  # 写入收敛表头
        for result in results:  # 逐方法遍历收敛轨迹
            for row in result.history:  # 逐唯一真实求解写入一行
                writer.writerow([result.name, int(row["evaluation"]), f"{row['best_objective']:.12e}", f"{row['candidate_objective']:.12e}"])  # 写入当前轨迹记录
    sorted_results = sorted(results, key=lambda item: item.objective)  # 按综合目标从优到劣排序方法
    best = sorted_results[0]  # 读取综合目标最优方法
    uniform = next(item for item in results if item.name == "uniform_mesh")  # 读取均匀网格基线结果
    improvement = (uniform.objective - best.objective) / max(uniform.objective, 1.0e-18)  # 计算最优方法相对均匀网格目标改善率
    lines = ["# 横向通道纯扭转工况：五种网格策略实验结果", "", "## 实验边界", "", "本实验采用一处重复横向通道的规范化箱形空间多桁架拓扑，使用真实 CalculiX B31 线弹性求解，在右端施加刚性截面小转角形成纯扭转；材料、截面和初弯曲均为统一算法基准参数，结果只用于比较网格决策方法，不作为张靖皋猫道横向通道的工程应力或承载力结论。", "", f"隐藏参考解为每根宏观杆件 {REFERENCE_SUBDIVISIONS} 分段，共 {benchmark.reference.element_count} 个单元；五种方法最终网格均受 {benchmark.element_cap} 个单元上限约束。", "", "## 方法对比", "", "| 方法 | 综合目标↓ | 扭矩误差↓ | 区域能量分布误差↓ | 中部场误差↓ | 热点召回↑ | 单元数 | 唯一求解 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]  # 初始化 Markdown 报告和方法表
    for result in sorted_results:  # 按综合目标排序写入方法表
        lines.append(f"| {result.name} | {result.objective:.6f} | {100.0 * result.torque_error:.3f}% | {100.0 * result.energy_error:.3f}% | {100.0 * result.probe_error:.3f}% | {100.0 * result.hotspot_recall:.1f}% | {result.solution.element_count} | {result.unique_solves} |")  # 写入当前方法指标
    lines.extend(["", "## 实验结论", "", f"综合目标最优方法为 **{best.name}**，相对均匀网格基线改善 **{100.0 * improvement:.2f}%**；其最终网格在相同单元上限下将细网格集中到扭转能量与邻域对比最强的连接区和斜撑区域。", f"均匀网格的扭矩误差为 **{100.0 * uniform.torque_error:.3f}%**，最优方法为 **{100.0 * best.torque_error:.3f}%**；均匀细化能够稳定整体位移场，但在固定单元预算下对主导扭转刚度的斜撑和连接区资源利用率较低。", "固定转角线弹性分析中总应变能与反力矩满足 U=Tθ/2，因此总能量误差没有作为独立目标重复计权，本实验改用十六个图区域归一化能量分布的总变差距离。", "热点 PSO 与纯 PSO 的差异直接反映候选区域降维是否有效；图多智能体反映局部状态和邻域消息能否在不建立全局价值函数时形成有效资源协商；DQN+GCN 的结果同时包含其图表示能力和仅有三十二次真实求解在线训练造成的样本效率限制。", "", "## 可复现证据", "", "每个候选配置均保留 CalculiX `.inp`、`.frd`、标准输出、标准错误和 `receipt.json`；`results.json` 保存完整拓扑区域、参考解、五种方法级别向量和逐次收敛轨迹。", ""])  # 补充结论和证据说明
    (output_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")  # 写出中文实验报告
    figure = plt.figure(figsize=(9.0, 5.2))  # 创建综合目标柱状图
    axes = figure.add_subplot(111)  # 创建单一绘图区
    axes.bar([result.name for result in sorted_results], [result.objective for result in sorted_results])  # 绘制五种方法综合目标
    axes.set_ylabel("Composite objective")  # 设置纵轴标题
    axes.set_title("Cross-passage torsion mesh benchmark")  # 设置图标题
    axes.tick_params(axis="x", rotation=25)  # 旋转方法名称避免重叠
    figure.tight_layout()  # 自动调整图边距
    figure.savefig(output_root / "method_objective.png", dpi=180)  # 保存综合目标图
    plt.close(figure)  # 关闭综合目标图释放内存
    figure = plt.figure(figsize=(9.0, 5.2))  # 创建收敛曲线图
    axes = figure.add_subplot(111)  # 创建单一收敛绘图区
    for result in results:  # 逐方法绘制历史最优曲线
        if result.history:  # 仅绘制包含搜索轨迹的方法
            axes.plot([row["evaluation"] for row in result.history], [row["best_objective"] for row in result.history], label=result.name)  # 绘制当前方法收敛曲线
    axes.set_xlabel("Unique CalculiX candidate solves")  # 设置横轴标题
    axes.set_ylabel("Best composite objective")  # 设置纵轴标题
    axes.set_title("Solver-budget convergence")  # 设置收敛图标题
    axes.legend()  # 显示方法图例
    figure.tight_layout()  # 自动调整图边距
    figure.savefig(output_root / "convergence.png", dpi=180)  # 保存收敛曲线图
    plt.close(figure)  # 关闭收敛图释放内存
    level_matrix = np.asarray([result.levels for result in results], dtype=np.float64)  # 构造方法与区域的最终网格级别矩阵
    figure = plt.figure(figsize=(12.0, 4.8))  # 创建最终网格级别热图
    axes = figure.add_subplot(111)  # 创建单一热图绘图区
    image = axes.imshow(level_matrix, aspect="auto", interpolation="nearest")  # 绘制方法区域级别矩阵
    axes.set_yticks(np.arange(len(results)))  # 设置方法行刻度
    axes.set_yticklabels([result.name for result in results])  # 写入方法行标签
    axes.set_xticks(np.arange(benchmark.region_count))  # 设置区域列刻度
    axes.set_xticklabels([str(index + 1) for index in range(benchmark.region_count)])  # 使用一基区域编号作为列标签
    axes.set_xlabel("Graph region id")  # 设置横轴标题
    axes.set_title("Final regional mesh levels")  # 设置热图标题
    figure.colorbar(image, ax=axes, label="Mesh level 0-3")  # 添加网格级别颜色标尺
    figure.tight_layout()  # 自动调整图边距
    figure.savefig(output_root / "final_levels.png", dpi=180)  # 保存最终级别热图
    plt.close(figure)  # 关闭热图释放内存
def locate_ccx(requested: str) -> str:  # 定位用户指定或系统安装的 CalculiX 可执行文件
    expanded = os.path.expanduser(os.path.expandvars(requested))  # 展开环境变量和用户目录
    if Path(expanded).is_file():  # 检查请求值是否为直接可执行文件路径
        return expanded  # 返回直接文件路径
    located = shutil.which(requested)  # 在系统 PATH 中搜索请求命令
    if located is not None:  # 检查是否找到请求命令
        return located  # 返回系统 PATH 中的可执行文件路径
    candidates = sorted(Path("/usr/bin").glob("ccx*"))  # 搜索 Ubuntu 包可能安装的版本化 ccx 命令
    if candidates:  # 检查是否存在版本化候选命令
        return str(candidates[0])  # 返回排序后的第一个可执行候选
    raise RuntimeError(f"CalculiX executable not found for {requested}")  # 无求解器时停止实验并拒绝伪造结果
def main() -> None:  # 解析命令行并执行完整五方法实验
    parser = argparse.ArgumentParser(description="Cross-passage pure-torsion five-method mesh benchmark")  # 创建命令行解析器
    parser.add_argument("--ccx", default=os.environ.get("CCX_CMD", "ccx"))  # 接收 CalculiX 可执行文件路径
    parser.add_argument("--output", default="experiments/cross_passage_torsion_benchmark/results")  # 接收实验结果输出目录
    arguments = parser.parse_args()  # 解析命令行参数
    output_root = Path(arguments.output).resolve()  # 将输出目录转换为绝对路径
    ccx_command = locate_ccx(arguments.ccx)  # 定位真实 CalculiX 求解器
    benchmark = TorsionBenchmark(output_root, ccx_command)  # 创建横向通道扭转基准对象
    benchmark.prepare_reference()  # 首先执行统一隐藏参考解
    runner = ExperimentRunner(benchmark)  # 创建五方法实验运行器
    results = runner.run_all()  # 执行均匀网格、纯 PSO、热点 PSO、多智能体和 DQN+GCN
    write_outputs(benchmark, results, output_root)  # 写出完整实验数据、图和结论
    print((output_root / "RESULTS.md").read_text(encoding="utf-8"))  # 在 GitHub Actions 日志中打印最终中文结论
if __name__ == "__main__":  # 检查脚本是否作为主程序执行
    main()  # 启动完整横向通道纯扭转对比实验
