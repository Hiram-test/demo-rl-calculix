# -*- coding: utf-8 -*-
"""
Abaqus工具函数模块
提取重复使用的函数，避免代码重复
"""
from itertools import combinations
import numpy as np


def find_edge_adjacent_cells(part_obj):
    """
    查找给定部件中每个Cell的所有"共边"相邻Cell。
    
    :param part_obj: Abaqus part对象
    :return: dict[cell_index -> set of adjacent cell indices]
    """
    all_cells = part_obj.cells
    edge_to_cell_map = {}
    for cell in all_cells:
        edge_indices = cell.getEdges()
        for edge_index in edge_indices:
            edge_to_cell_map.setdefault(edge_index, set()).add(cell.index)

    cell_adjacency = {c.index: set() for c in all_cells}
    for edge, owner_cell_indices in edge_to_cell_map.items():
        if len(owner_cell_indices) > 1:
            for cell1_idx, cell2_idx in combinations(owner_cell_indices, 2):
                cell_adjacency[cell1_idx].add(cell2_idx)
                cell_adjacency[cell2_idx].add(cell1_idx)
    return cell_adjacency


def get_cell_geometric_features(cell, part):
    """
    为单个Cell提取精简且有效的几何特征。
    此版本经过多次修正，确保使用的所有API方法都真实存在且调用方式正确。

    :param cell: Abaqus Part.cell 对象。
    :param part: 拥有该cell的 Abaqus Part 对象。
    :return: 包含关键几何特征的字典。
    """
    
    # --- 准备工作: 获取顶点坐标 ---
    coords = []
    try:
        vertex_indices = cell.getVertices()
        if vertex_indices:
            vertex_objects = [part.vertices[i] for i in vertex_indices]
            coords = np.array([v.pointOn[0] for v in vertex_objects])
    except Exception:
        pass

    # --- 1. 基础尺寸与形状描述符 ---
    
    try:
        volume = cell.getSize()
    except Exception:
        volume = 0.0

    bounding_box_aspect_ratio = -1.0
    if len(coords) > 0:
        try:
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            dims = max_coords - min_coords
            positive_dims = sorted([d for d in dims if d > 1e-9])
            if len(positive_dims) >= 2:
                bounding_box_aspect_ratio = positive_dims[-1] / positive_dims[0]
            elif len(positive_dims) == 1:
                bounding_box_aspect_ratio = 1.0
        except Exception:
            pass

    # --- 2. 高级形状与位置描述符 ---

    max_edge_curvature = -1.0
    try:
        max_edge_curvature = 0.0
        edge_indices = cell.getEdges()
        if edge_indices:
            sample_parameters = [0.0, 0.25, 0.5, 0.75, 1.0]
            all_curvatures_on_cell = []
            for idx in edge_indices:
                edge = part.edges[idx]
                curvatures_on_this_edge = []
                for p in sample_parameters:
                    try:
                        curvature_data = edge.getCurvature(parameter=p)
                        if curvature_data and 'curvature' in curvature_data:
                            curvatures_on_this_edge.append(abs(curvature_data['curvature']))
                    except Exception:
                        continue
                if curvatures_on_this_edge:
                    all_curvatures_on_cell.append(max(curvatures_on_this_edge))
            if all_curvatures_on_cell:
                max_edge_curvature = max(all_curvatures_on_cell)
    except Exception:
        pass

    # 是否位于外部边界: 使用 face.getCells() 进行修正
    is_on_exterior = -1.0
    try:
        is_on_exterior = 0.0 # 默认是内部
        face_indices = cell.getFaces()
        if face_indices:
            for idx in face_indices:
                face = part.faces[idx]
                # --- FINAL CORRECTION ---
                # The 'face' object has a getCells() method, not getAdjacentCells().
                # If a face belongs to only one cell, it is an exterior face.
                if len(face.getCells()) == 1:
                    is_on_exterior = 1.0
                    break # We found one exterior face, so the cell is on the boundary.
                # --- END CORRECTION ---
    except Exception:
        pass

    centroid = np.array([0.0, 0.0, 0.0])
    if len(coords) > 0:
        try:
            centroid = np.mean(coords, axis=0)
        except Exception:
            pass
            
    return {
        'volume': volume,
        'bounding_box_aspect_ratio': bounding_box_aspect_ratio,
        'max_edge_curvature': max_edge_curvature,
        'is_on_exterior': is_on_exterior,
        'centroid_x': centroid[0],
        'centroid_y': centroid[1],
        'centroid_z': centroid[2],
    }