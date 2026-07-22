# -*- coding: utf-8 -*-
from odbAccess import *
from abaqusConstants import CENTROID, WHOLE_ELEMENT
import sys
import json
import math
import os

def convert_to_python_types(obj):
    """
    递归转换 Abaqus 数据类型为 Python 原生类型，以便 JSON 序列化。
    """
    if isinstance(obj, dict):
        return {convert_to_python_types(k): convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, float):
        return float(obj)
    elif hasattr(obj, 'item'):  # numpy-like types
        return obj.item()
    elif hasattr(obj, '__float__'):  # any type that can be converted to float
        return float(obj)
    elif hasattr(obj, '__int__'):  # any type that can be converted to int
        return int(obj)
    else:
        return obj

def get_element_averaged_field(frame, instance, field_name):
    """
    高效提取指定场在每个单元的平均值。
    Abaqus的CENTROID位置已经为我们做好了平均。

    参数:
    - frame: ODB Frame对象
    - instance: ODB Instance对象
    - field_name: 场的名称 (e.g., 'S' for stress, 'SENER' for strain energy density)

    返回:
    - a dict: {element_label: value}。如果场是标量，value是float；如果是张量/向量，value是list。
    """
    element_data = {}
    if field_name not in frame.fieldOutputs:
        print("Warning: Field '{}' not found in the output database.".format(field_name))
        return element_data

    field = frame.fieldOutputs[field_name]
    
    # 优先使用CENTROID值，因为它已经是单元平均值，效率最高
    if field_name == 'EVOL':
        field_subset = field.getSubset(region=instance, position=WHOLE_ELEMENT)
    else:
        field_subset = field.getSubset(region=instance, position=CENTROID)
    for val in field_subset.values:
        # .data 可能是标量float或向量/张量list，取决于场类型
        element_data[val.elementLabel] = val.data
            
    return element_data

def get_total_strain_energy(step):
    """
    从历史输出中获取整个模型的总应变能(ALLSE)。
    返回最后一个时间点的ALLSE值。
    """
    # 历史输出的Region key通常是'Assembly ASSEMBLY'
    for region in step.historyRegions.values():
        if 'ALLSE' in region.historyOutputs:
            history_data = region.historyOutputs['ALLSE'].data
            if history_data:
                # 返回最后一个时间点的值 (time, value)
                return history_data[-1][1]
    print("Warning: Total Strain Energy 'ALLSE' not found in history output.")
    return 0.0

def extract_physical_features(odb_path, step_name=None, frame_index=-1, instance_name=None):
    """
    为机器学习模型提取精简且高效的物理特征。
    
    提取特征包括:
    - Mises Stress (单元平均值)
    - Strain Energy Density (SENER, 单元平均值)
    - Stress Components (S11, S22, S33, S12, ... 单元平均值)
    - Strain Components (E11, E22, ... 单元平均值)
    - Total Strain Energy of the model (ALLSE, 全局标量)
    """
    try:
        odb = openOdb(path=odb_path, readOnly=True)

        # --- 确定分析步、帧和实例 ---
        step_key = step_name if step_name else list(odb.steps.keys())[-1]
        step = odb.steps[step_key]
        
        frame = step.frames[frame_index]
        
        inst_key = instance_name if instance_name else list(odb.rootAssembly.instances.keys())[0]
        instance = odb.rootAssembly.instances[inst_key]
        
        print("Processing ODB: '{}'".format(odb_path))
        print("Step: '{}', Frame: {}, Instance: '{}'".format(step_key, frame_index, inst_key))

        # --- 提取单元级（节点）特征 ---
        # 使用一个字典来存储所有单元的特征
        element_features = {}
        all_element_labels = [elem.label for elem in instance.elements]
        for label in all_element_labels:
            element_features[label] = {}

        # 1. 冯·米塞斯应力 (标量)
        stress_data = get_element_averaged_field(frame, instance, 'S')

        for label, s in stress_data.items():
            s11, s22, s33, s12, s13, s23 = s[0], s[1], s[2], s[3], s[4], s[5]
            mises = math.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) + 3 * (s12**2 + s13**2 + s23**2))
            element_features[label]['mises'] = mises

        # 2. 应变能密度 (标量)
        sener_data = get_element_averaged_field(frame, instance, 'SENER')
        for label, sener_val in sener_data.items():
            element_features[label]['strain_energy_density'] = sener_val

        evol_data = get_element_averaged_field(frame, instance, 'EVOL')
        for label, evol_val in evol_data.items():
            sener_val = element_features[label].get('strain_energy_density')
            element_features[label]['strain_energy'] = sener_val * evol_val

        # 3. 应力分量 (向量/张量) - 可选，但对GNN很有用
        stress_data = get_element_averaged_field(frame, instance, 'S')
        s_labels = ['s11', 's22', 's33', 's12', 's13', 's23']
        for label, s_vals in stress_data.items():
            for i, sl in enumerate(s_labels):
                element_features[label][sl] = s_vals[i]

        # 4. 应变分量 (向量/张量) - 可选
        strain_field_key = 'LE' if 'LE' in frame.fieldOutputs else 'E'
        strain_data = get_element_averaged_field(frame, instance, strain_field_key)
        e_labels = ['e11', 'e22', 'e33', 'e12', 'e13', 'e23'] # 假设和应力顺序一致
        for label, e_vals in strain_data.items():
            for i, el in enumerate(e_labels):
                 element_features[label][el] = e_vals[i]

        # --- 提取全局（图级）特征 ---
        total_strain_energy = get_total_strain_energy(step)
        
        # --- 组装最终结果 ---
        result = {
            "model_features": {
                "total_strain_energy": total_strain_energy,
                "num_elements": len(all_element_labels),
                "num_nodes": len(instance.nodes)
            },
            "element_features": element_features
        }

        odb.close()
        return result

    except Exception as e:
        print("Error extracting physical features: {}".format(e))
        import traceback
        traceback.print_exc()
        # 确保在出错时关闭odb
        if 'odb' in locals() and not odb.isClosed():
            odb.close()
        return None

if __name__ == '__main__':
    # 用法: abaqus python extract_physical_features.py <odb_path> [instance_name] [step_name]
    if len(sys.argv) < 2:
        print("Usage: python extract_physical_features.py <odb_path> [instance_name] [step_name]")
        sys.exit(1)

    odb_file_path = sys.argv[1]
    instance_name = sys.argv[2] if len(sys.argv) >= 3 else 'The whole beam'
    step_name = sys.argv[3] if len(sys.argv) >= 4 else None
    
    physical_features = extract_physical_features(
        odb_path=odb_file_path,
        instance_name=instance_name,
        step_name=step_name,
    )
    
    if physical_features:
        # 转换所有数据为 Python 原生类型
        physical_features = convert_to_python_types(physical_features)
        
        # 生成输出文件名：基于输入的 ODB 文件名
        odb_dir = os.path.dirname(odb_file_path)
        odb_basename = os.path.splitext(os.path.basename(odb_file_path))[0]
        output_filename = "{}_physical_features.json".format(odb_basename)
        output_path = os.path.join(odb_dir, output_filename)
        
        # 保存到 JSON 文件
        with open(output_path, 'w') as f:
            json.dump(physical_features, f, indent=2)
        
        print("Physical features successfully saved to: {}".format(output_path))
