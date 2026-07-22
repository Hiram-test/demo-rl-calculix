# -*- coding: utf-8 -*-
"""
有限元分析脚本 - 仅进行FEA分析，不包含网格生成
功能：基于已有网格的CAE文件，创建作业、提交分析并提取结果
"""
import sys
import os
import json
import codecs

# 添加项目根目录到sys.path以便导入abaqus_utils
# 当通过execfile执行时，__file__可能未定义，因此从CAE文件路径推断项目根目录
try:
    # 尝试使用__file__（如果可用）
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 如果__file__未定义，使用硬编码的项目根目录路径
    script_dir = 'D:/BIM2FEA/multi_graph'
    
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from abaqus import *
from abaqusConstants import *
import part
import mesh
import job
import abaqus_utils

# --- 从命令行参数获取变量 ---
if len(sys.argv) < 3:
    print("Error: Missing command line arguments.")
    print("Usage: abaqus cae -noGUI script.py -- <cae_file_path> <job_name>")
    sys.exit(1)
    
cae_file_path = sys.argv[-2]
job_name = sys.argv[-1]
model_name = 'Model-1'
instance_name = 'The whole beam'

# --- 开始执行Abaqus命令 ---
print('=== FEA Analysis Script Started ===')
print('  CAE File: {}'.format(cae_file_path))
print('  Job Name: {}'.format(job_name))
print('  Model Name: {}'.format(model_name))
print('  Instance Name: {}'.format(instance_name))

# --- 主程序 ---

# 1. 打开CAE文件（应该已经包含网格）
print('\n--- Step 1: Opening meshed CAE file ---')
openMdb(pathName=cae_file_path)
my_model = mdb.models[model_name]
myAssembly = my_model.rootAssembly
myInstance = myAssembly.instances[instance_name]

# 检查网格是否已存在
num_elements = len(myInstance.elements)
num_nodes = len(myInstance.nodes)
print('  Mesh found: {} elements, {} nodes'.format(num_elements, num_nodes))

if num_elements == 0:
    print('  Warning: No elements found in the CAE file. Mesh may not be generated.')
    print('  Proceeding anyway, but analysis may fail.')

# 2. 创建并提交作业
print('\n--- Step 2: Creating and submitting job ---')
my_job = mdb.Job(
    name=str(job_name),  # 确保job_name是字符串格式
    model=model_name,
)
# 注意：parallelizationMethodExplicit在Abaqus 2024中已被移除
# my_job.setValues(parallelizationMethodExplicit=DOMAIN, numDomains=8)
# try:
#     my_job.setValues(numCpus=8, numDomains=8)  # 使用numCpus替代
# except:
#     pass  # 如果设置失败，使用默认值
# my_job.setValues(numGPUs=1)
my_job.submit(consistencyChecking=OFF)
my_job.waitForCompletion()
print('Job "{}" has completed.'.format(job_name))

# --- 数据提取与整合 ---
print("\n--- Step 3: Starting Data Extraction ---")

# 3. 获取拓扑关系（邻接关系）
print("Finding cell adjacencies...")
cell_adjacency_map = abaqus_utils.find_edge_adjacent_cells(myInstance)
print("Adjacency map created for {} cells.".format(len(cell_adjacency_map)))

# 4. 建立单元到Cell的映射关系
print("Mapping elements to parent cells...")
cell_to_elements_map = {}
all_cells_from_part = myInstance.cells
for cell in all_cells_from_part:
    # 直接从cell对象获取其拥有的所有元素，效率极高
    elements = cell.getElements()
    cell_to_elements_map[cell.index] = [e.label for e in elements]
# 使用循环累加替代sum函数（Abaqus环境中sum可能不可用）
total_elements = 0
for v in cell_to_elements_map.values():
    total_elements += len(v)
print("Element mapping complete. Found {} elements across {} cells.".format(total_elements, len(all_cells_from_part)))

# 5. 整合所有信息到单一数据结构中
print("\nAggregating all features into a final data structure...")
all_cells_data = []
# 按索引排序以保证每次运行输出顺序一致
sorted_cells = sorted(all_cells_from_part, key=lambda c: c.index) 

for cell in sorted_cells:
    cell_index = cell.index
    
    # 获取几何特征
    geometric_features = abaqus_utils.get_cell_geometric_features(cell, myInstance)
    
    # 准备要写入JSON的数据
    geom_data_for_json = {
        'volume': geometric_features['volume'],
        'bounding_box_aspect_ratio': geometric_features['bounding_box_aspect_ratio'],
        'max_edge_curvature': geometric_features['max_edge_curvature'],
        'is_on_exterior': geometric_features['is_on_exterior'],
        'centroid_x': geometric_features['centroid_x'],
        'centroid_y': geometric_features['centroid_y'],
        'centroid_z': geometric_features['centroid_z']
    }

    # 组装当前cell的所有信息
    cell_data = {
        'cell_index': cell_index,
        'element_labels': sorted(cell_to_elements_map.get(cell_index, [])),
        'adjacent_cell_indices': sorted(list(cell_adjacency_map.get(cell_index, set()))),
        'geometric_features': geom_data_for_json
    }
    all_cells_data.append(cell_data)

print("Data aggregation complete.")

# 6. 保存整合后的数据到单一JSON文件
output_filename = '{}_comprehensive_data.json'.format(job_name)
try:
    with codecs.open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_cells_data, f, indent=2, ensure_ascii=False)
    print('\nComprehensive data for {} cells saved to: {}'.format(len(all_cells_data), output_filename))
except Exception as e:
    print('Error saving comprehensive data: {}'.format(str(e)))

print("\n=== FEA Analysis Script Finished Successfully ===")
print("Summary:")
print("  - Elements: {}".format(total_elements))
print("  - Nodes: {}".format(num_nodes))
print("  - Cells: {}".format(len(all_cells_data)))
print("  - Output JSON: {}".format(output_filename))

