# -*- coding: utf-8 -*-
"""
独立的网格生成脚本
功能：根据mesh_density配置对CAE文件进行网格划分，并输出网格划分结果
"""
import sys
import os
import json
import codecs
from abaqus import *
from abaqusConstants import *
import part
import mesh
import assembly

# --- 从命令行参数获取变量 ---
if len(sys.argv) < 4:
    print("Error: Missing command line arguments.")
    print("Usage: abaqus cae -noGUI mesh_generation.py -- <cae_file_path> <mesh_density_file> <output_cae_file>")
    print("  or: abaqus cae -noGUI mesh_generation.py -- <cae_file_path> <mesh_density_file> <output_cae_file> <model_name> <part_name>")
    sys.exit(1)
    
cae_file_path = sys.argv[-3]
mesh_density_file = sys.argv[-2]
output_cae_file = sys.argv[-1]

model_name = 'Model-1'
instance_name = 'The whole beam'

# 从本地JSON文件读取mesh density配置
print('Reading mesh density configuration from: {}'.format(mesh_density_file))
with codecs.open(mesh_density_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 检查是否使用edge-based方式
use_edge_based = data.get('use_edge_based', False)
print('Mesh density mode: {}'.format('Edge-based' if use_edge_based else 'Cell-based'))

if use_edge_based and 'edge_mesh_density' in data:
    # 使用edge-based方式：直接对每条edge设置种子
    edge_to_seed_size_map_raw = data['edge_mesh_density']
    edge_to_seed_size_map = {int(k): float(v) for k, v in edge_to_seed_size_map_raw.items()}
    cell_to_seed_size_map = None
    print('Loaded edge_mesh_density for {} edges'.format(len(edge_to_seed_size_map)))
else:
    # 使用cell-based方式（向后兼容）
    cell_to_seed_size_map_raw = data['cell_mesh_density']
    cell_to_seed_size_map = {int(k): float(v) for k, v in cell_to_seed_size_map_raw.items()}
    edge_to_seed_size_map = None
    print('Loaded cell_mesh_density for {} cells'.format(len(cell_to_seed_size_map)))

# --- 开始执行Abaqus命令 ---
print('\n=== Mesh Generation Script Started ===')
print('  CAE File: {}'.format(cae_file_path))
print('  Mesh Density File: {}'.format(mesh_density_file))
print('  Output CAE File: {}'.format(output_cae_file))
print('  Model Name: {}'.format(model_name))
print('  Instance Name: {}'.format(instance_name))

# 1. 打开CAE文件
print('\n--- Step 1: Opening CAE file ---')
openMdb(pathName=cae_file_path)
my_model = mdb.models[model_name]
myAssembly = my_model.rootAssembly
myInstance = myAssembly.instances[instance_name]
print('CAE file opened successfully.')
print('  Total cells in instance: {}'.format(len(myInstance.cells)))

# 2. 设置种子点（根据模式：edge-based或cell-based）
print('\n--- Step 2: Setting seed sizes ---')

if use_edge_based and edge_to_seed_size_map is not None:
    # Edge-based模式：直接对每条edge设置种子
    print('Using edge-based seeding mode...')
    print('Processing {} edges from edge_mesh_density configuration'.format(len(edge_to_seed_size_map)))
    seeded_edges = 0
    
    for edge_index in sorted(edge_to_seed_size_map.keys()):
        seed_size = edge_to_seed_size_map[edge_index]
        
        # 获取edge对象
        try:
            edge_obj = myInstance.edges[edge_index]
            
            # 对单条edge设置种子（使用Assembly对象的方法）
            myAssembly.seedEdgeBySize(edges=[edge_obj], size=seed_size, deviationFactor=0.1, 
                                    minSizeFactor=0.1, constraint=FINER)
            
            seeded_edges += 1
            if seeded_edges % 100 == 0 or seeded_edges == len(edge_to_seed_size_map):
                print('  - Seeded {} / {} edges...'.format(seeded_edges, len(edge_to_seed_size_map)))
        except Exception as e:
            print('  - Warning: Failed to seed edge {}: {}'.format(edge_index, str(e)))
    
    print('Edge-based seeding complete. {} edges seeded.'.format(seeded_edges))
    
else:
    # Cell-based模式：遍历cell，对每个cell的所有边设置种子（向后兼容）
    print('Using cell-based seeding mode (legacy)...')
    all_cells = myInstance.cells
    print('Processing {} cells from mesh density configuration'.format(len(cell_to_seed_size_map)))
    seeded_cells = 0
    
    for cell_index in sorted(cell_to_seed_size_map.keys()):
        seed_size = cell_to_seed_size_map[cell_index]
        
        # a. 获取当前索引对应的Cell对象
        target_cell = all_cells[cell_index]
        
        # b. 获取该Cell的所有边的索引
        edge_indices_of_cell = target_cell.getEdges()
        
        # c. 根据边的索引，从部件的边仓库(edges repository)中获取边的对象
        edges_to_seed = [myInstance.edges[edge_index] for edge_index in edge_indices_of_cell]
        
        # d. 对找到的边进行局部播种（按尺寸）
        #    不使用constraint参数，让后面的种子设置可以覆盖前面的
        myAssembly.seedEdgeBySize(edges=edges_to_seed, size=seed_size, deviationFactor=0.1, 
                                minSizeFactor=0.1)
        
        seeded_cells += 1
        if seeded_cells % 10 == 0 or seeded_cells == len(cell_to_seed_size_map):
            print('  - Seeded {} / {} cells...'.format(seeded_cells, len(cell_to_seed_size_map)))
    
    print('Cell-based seeding complete. {} cells seeded.'.format(seeded_cells))

# 3. 生成网格
print('\n--- Step 3: Generating mesh ---')
myAssembly.generateMesh(regions=(myInstance,))
print('Mesh generation for all cells complete.')

# 4. 统计网格信息
print('\n--- Step 4: Collecting mesh statistics ---')
all_elements = myInstance.elements
all_nodes = myInstance.nodes
num_elements = len(all_elements)
num_nodes = len(all_nodes)

print('  Total number of elements: {}'.format(num_elements))
print('  Total number of nodes: {}'.format(num_nodes))

# 统计每个cell的元素数量
cell_element_counts = {}
all_cells_from_part = myInstance.cells
for cell in all_cells_from_part:
    elements = cell.getElements()
    cell_element_counts[cell.index] = len(elements) if elements else 0

print('  Number of cells: {}'.format(len(all_cells_from_part)))
# 使用循环累加替代sum函数（Abaqus环境中sum可能不可用）
cells_with_elements = 0
for count in cell_element_counts.values():
    if count > 0:
        cells_with_elements += 1
print('  Cells with elements: {}'.format(cells_with_elements))

# 统计元素类型分布
element_types = {}
for elem in all_elements:
    elem_type = elem.type
    element_types[elem_type] = element_types.get(elem_type, 0) + 1

print('\n  Element type distribution:')
for elem_type, count in sorted(element_types.items()):
    print('    {}: {}'.format(elem_type, count))

# 5. 保存带网格的CAE文件
print('\n--- Step 5: Saving meshed CAE file ---')
mdb.saveAs(pathName=output_cae_file)
print('Meshed CAE file saved to: {}'.format(output_cae_file))

# 6. 输出网格划分结果到JSON文件
print('\n--- Step 6: Exporting mesh results ---')
# 将element_types的键转换为字符串（JSON要求字典键必须是字符串）
element_types_str = {str(k): v for k, v in element_types.items()}
mesh_results = {
    'mesh_statistics': {
        'total_elements': num_elements,
        'total_nodes': num_nodes,
        'total_cells': len(all_cells_from_part),
        'cells_with_elements': cells_with_elements,
        'element_type_distribution': element_types_str
    },
    'cell_element_counts': cell_element_counts,
    'mesh_configuration': {
        'input_cae_file': cae_file_path,
        'mesh_density_file': mesh_density_file,
        'output_cae_file': output_cae_file,
        'model_name': model_name,
        'instance_name': instance_name,
        'seeding_mode': 'edge-based' if use_edge_based else 'cell-based',
        'num_edges_seeded': len(edge_to_seed_size_map) if use_edge_based and edge_to_seed_size_map else 0,
        'num_cells_seeded': len(cell_to_seed_size_map) if not use_edge_based and cell_to_seed_size_map else 0
    }
}

# 添加seed sizes到结果中（根据模式）
if use_edge_based and edge_to_seed_size_map:
    mesh_results['edge_seed_sizes'] = edge_to_seed_size_map
if not use_edge_based and cell_to_seed_size_map:
    mesh_results['cell_seed_sizes'] = cell_to_seed_size_map

# 保存结果到JSON文件
output_json_file = output_cae_file.replace('.cae', '_mesh_results.json')
try:
    with codecs.open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(mesh_results, f, indent=2, ensure_ascii=False)
    print('Mesh results saved to: {}'.format(output_json_file))
except Exception as e:
    print('Error saving mesh results: {}'.format(str(e)))
