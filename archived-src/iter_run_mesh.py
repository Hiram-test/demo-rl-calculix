# -*- coding: utf-8 -*-
import sys
import os
import json
import codecs

# 添加项目根目录到sys.path以便导入abaqus_utils
# 当通过execfile执行时，__file__可能未定义，因此从CAE文件路径推断项目根目录
# CAE文件在项目根目录，所以我们从它的路径获取根目录
try:
    # 尝试使用__file__（如果可用）
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 如果__file__未定义，从CAE文件路径推断（它在项目根目录）
    # 注意：此时sys.argv还未解析，但我们可以先添加当前工作目录的父目录
    # 或者我们可以硬编码项目根目录路径
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
if len(sys.argv) < 5:
    print("Error: Missing command line arguments.")
    print("Usage: abaqus cae -noGUI script.py -- <cae_file_path> <mesh_size> <mesh_density_file> <job_name>")
    sys.exit(1)
    
cae_file_path = sys.argv[-4]
global_mesh_size = float(sys.argv[-3])
mesh_density_file = sys.argv[-2]
job_name = sys.argv[-1]
model_name = 'Model-1'
instance_name = 'The whole beam'

# 从本地JSON文件读取mesh density配置
with codecs.open(mesh_density_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 检查是否使用基于edge的密度设置
use_edge_based = data.get('use_edge_based', False)

if use_edge_based and 'edge_mesh_density' in data:
    # 使用edge-based密度设置
    edge_to_seed_size_map_raw = data['edge_mesh_density']
    edge_to_seed_size_map = {int(k): float(v) for k, v in edge_to_seed_size_map_raw.items()}
    cell_to_seed_size_map = {}
    print('Using edge-based mesh density with {} edges'.format(len(edge_to_seed_size_map)))
else:
    # 使用cell-based密度设置（旧方式）
    cell_to_seed_size_map_raw = data['cell_mesh_density']
    cell_to_seed_size_map = {int(k): float(v) for k, v in cell_to_seed_size_map_raw.items()}
    edge_to_seed_size_map = {}
    print('Using cell-based mesh density with {} cells'.format(len(cell_to_seed_size_map)))

# --- 开始执行Abaqus命令 ---
print('Script started with arguments:')
print('  CAE File: {}'.format(cae_file_path))
print('  Mesh Size: {}'.format(global_mesh_size))
print('  Mesh Density File: {}'.format(mesh_density_file))
print('  Job Name: {}'.format(job_name))


# --- 主程序 ---

# 1. 打开CAE文件
openMdb(pathName=cae_file_path)
my_model = mdb.models[model_name]
myAssembly = my_model.rootAssembly
myInstance = myAssembly.instances[instance_name]


# 2.1 先进行全局播种
print('Starting global seeding...')
myAssembly.seedPartInstance(regions=(myInstance,), size=global_mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
print('Global seeding complete.')

# 2. 根据配置进行局部播种
print('Starting local seeding...')

if edge_to_seed_size_map:
    # 方式1：直接按edge设置种子（避免覆盖问题）
    print('Processing {} edges from mesh density configuration'.format(len(edge_to_seed_size_map)))
    all_edges = myInstance.edges
    
    for edge_index in sorted(edge_to_seed_size_map.keys()):
        seed_size = edge_to_seed_size_map[edge_index]
        
        # 直接获取edge对象并设置种子
        target_edge = all_edges[edge_index]
        myAssembly.seedEdgeBySize(edges=[target_edge], size=seed_size, deviationFactor=0.1, 
                                minSizeFactor=0.1, constraint=FINER)
        
        step_number = sorted(edge_to_seed_size_map.keys()).index(edge_index) + 1
        if step_number % 50 == 0 or step_number == len(edge_to_seed_size_map):
            print('  - Seeded Edge[{}] with size {} (step {}/{})'.format(
                edge_index, seed_size, step_number, len(edge_to_seed_size_map)))
    
elif cell_to_seed_size_map:
    # 方式2：按cell设置种子（旧方式，可能有覆盖问题）
    print('Processing {} cells from mesh density configuration'.format(len(cell_to_seed_size_map)))
    all_cells = myInstance.cells
    
    for cell_index in sorted(cell_to_seed_size_map.keys()):
        seed_size = cell_to_seed_size_map[cell_index]
        
        # a. 获取当前索引对应的Cell对象
        target_cell = all_cells[cell_index]
        
        # b. 获取该Cell的所有边的索引
        edge_indices_of_cell = target_cell.getEdges()
        
        # c. 根据边的索引，从instance的边仓库(edges repository)中获取边的对象
        edges_to_seed = [myInstance.edges[edge_index] for edge_index in edge_indices_of_cell]
        
        # d. 对找到的边进行局部播种（按尺寸）
        #    constraint=FINER确保局部种子覆盖全局种子
        myAssembly.seedEdgeBySize(edges=edges_to_seed, size=seed_size, deviationFactor=0.1, 
                                minSizeFactor=0.1, constraint=FINER)
        
        step_number = sorted(cell_to_seed_size_map.keys()).index(cell_index) + 1
        print('  - Seeded edges of Cell[{}] with size {} (step {}/{}).'.format(
            cell_index, seed_size, step_number, len(cell_to_seed_size_map)))

print('Local seeding complete.')


myAssembly.generateMesh(regions=(myInstance,))
print('Mesh generation for all cells complete.')
print('Number of elements: {}'.format(len(myInstance.elements)))

# 4. 创建并提交作业
my_job = mdb.Job(
name=str(job_name),  # 确保job_name是字符串格式
model=model_name,
)
# 注意：parallelizationMethodExplicit在Abaqus 2024中已被移除
# my_job.setValues(parallelizationMethodExplicit=DOMAIN, numDomains=8)
try:
    my_job.setValues(numCpus=8, numDomains=8)  # 使用numCpus替代
except:
    pass  # 如果设置失败，使用默认值
# my_job.setValues(numGPUs=1)
my_job.submit()
my_job.waitForCompletion()
print('Job "{}" has completed.'.format(job_name))

# --- 数据提取与整合 ---
print("\n--- Starting Data Extraction ---")

# 5. 获取拓扑关系（邻接关系）和edge到cells的映射
print("Finding cell adjacencies and edge-to-cells mapping...")
# 构建edge到cells的映射
edge_to_cells_map = {}
all_cells_temp = myInstance.cells
for cell in all_cells_temp:
    edge_indices = cell.getEdges()
    for edge_index in edge_indices:
        if edge_index not in edge_to_cells_map:
            edge_to_cells_map[edge_index] = []
        edge_to_cells_map[edge_index].append(cell.index)

# 使用现有函数获取cell邻接关系
cell_adjacency_map = abaqus_utils.find_edge_adjacent_cells(myInstance)
print("Adjacency map created for {} cells.".format(len(cell_adjacency_map)))
print("Edge-to-cells map created for {} edges.".format(len(edge_to_cells_map)))

# 6. 【高效】建立单元到Cell的映射关系
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

# 7. 整合所有信息到单一数据结构中
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

# 8. 保存整合后的数据到单一JSON文件
output_filename = '{}_comprehensive_data.json'.format(job_name)
try:
    with codecs.open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_cells_data, f, indent=2, ensure_ascii=False)
    print('\nComprehensive data for {} cells saved to: {}'.format(len(all_cells_data), output_filename))
except Exception as e:
    print('Error saving comprehensive data: {}'.format(str(e)))

# 9. 保存edge到cells的映射到单独的JSON文件
edge_mapping_filename = '{}_edge_to_cells.json'.format(job_name)
try:
    # 将edge_to_cells_map转换为可JSON序列化的格式（key转为字符串）
    edge_to_cells_data = {str(edge_id): cell_list for edge_id, cell_list in edge_to_cells_map.items()}
    with codecs.open(edge_mapping_filename, 'w', encoding='utf-8') as f:
        json.dump(edge_to_cells_data, f, indent=2, ensure_ascii=False)
    print('Edge-to-cells mapping for {} edges saved to: {}'.format(len(edge_to_cells_map), edge_mapping_filename))
except Exception as e:
    print('Error saving edge-to-cells mapping: {}'.format(str(e)))

print("\nScript finished successfully.")
