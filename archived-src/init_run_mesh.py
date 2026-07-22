# -*- coding: utf-8 -*-
import sys
import os
import json
import codecs

# Add project root directory to sys.path to import abaqus_utils
# When executed via execfile, __file__ may be undefined, so infer project root from CAE file path
# CAE file is in project root, so we get root directory from its path
try:
    # Try to use __file__ (if available)
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # If __file__ is undefined, infer from CAE file path (which is in project root)
    # Note: sys.argv is not yet parsed, but we can add the parent directory of current working directory
    # Or we can hard-code the project root directory path
    script_dir = 'D:/BIM2FEA/multi_graph'
    
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from abaqus import *
from abaqusConstants import *
import part
import mesh
import job
import abaqus_utils

# --- Get variables from command line arguments ---
if len(sys.argv) < 4:
    print("Error: Missing command line arguments.")
    print("Usage: abaqus cae -noGUI script.py -- <cae_file_path> <mesh_size> <job_name>")
    sys.exit(1)
    
cae_file_path = sys.argv[-3]
mesh_size = float(sys.argv[-2])
job_name = sys.argv[-1]
model_name = 'Model-1'
instance_name = 'The whole beam'

# --- Start executing Abaqus commands ---
print('Script started with arguments:')
print('  CAE File: {}'.format(cae_file_path))
print('  Mesh Size: {}'.format(mesh_size))
print('  Job Name: {}'.format(job_name))


# --- Main Program ---

# 1. Open CAE file
openMdb(pathName=cae_file_path)
my_model = mdb.models[model_name]
myAssembly = my_model.rootAssembly
myInstance = myAssembly.instances[instance_name]

# 2. Mesh generation
# 2.1 Perform global seeding first
print('Starting global seeding...')
myAssembly.seedPartInstance(regions=(myInstance,), size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
print('Global seeding complete.')

# 2.2 Perform local refined seeding
print('Starting local seeding based on cell dictionary...')
all_cells = myInstance.cells

# First loop: seed all cells
for target_cell in all_cells:
    # b. Get the indices of all edges of this Cell
    edge_indices_of_cell = target_cell.getEdges()
    
    # c. Get edge objects from part's edges repository based on edge indices
    #    Note: Need to create a temporary sequence to store edge objects
    edges_to_seed = [myInstance.edges[edge_index] for edge_index in edge_indices_of_cell]
    
    # d. Perform local seeding on the found edges (by size)
    #    constraint=FINER ensures local seeds override global seeds
    myAssembly.seedEdgeBySize(edges=edges_to_seed, size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1, constraint=FINER)

print('Local seeding complete.')


myAssembly.generateMesh(regions=(myInstance,))
print('Mesh generation for all cells complete.')
print('Number of elements: {}'.format(len(myInstance.elements)))


# 3. Create and submit job (this section can be commented out to save time if only extracting geometry and mesh data)
my_job = mdb.Job(
name=str(job_name),  # Ensure job_name is in string format
model=model_name)
# Note: parallelizationMethodExplicit has been removed in Abaqus 2024

# try:
#     my_job.setValues(numCpus=8, numDomains=8)  # Use numCpus instead
# except:
#     pass  # If setting fails, use default values
# try:
#     my_job.setValues(numGPUs=1)
# except:
#     pass  # If GPU setting is not supported, skip

my_job.submit()
my_job.waitForCompletion()
print('Job "{}" has completed.'.format(job_name))

# --- Data Extraction and Integration ---
print("\n--- Starting Data Extraction ---")

# 4. Get topological relationships (adjacency) and edge-to-cells mapping
print("Finding cell adjacencies and edge-to-cells mapping...")
# Build edge-to-cells mapping
edge_to_cells_map = {}
all_cells_temp = myInstance.cells
for cell in all_cells_temp:
    edge_indices = cell.getEdges()
    for edge_index in edge_indices:
        if edge_index not in edge_to_cells_map:
            edge_to_cells_map[edge_index] = []
        edge_to_cells_map[edge_index].append(cell.index)

# Use existing function to get cell adjacency relationships
cell_adjacency_map = abaqus_utils.find_edge_adjacent_cells(myInstance)
print("Adjacency map created for {} cells.".format(len(cell_adjacency_map)))
print("Edge-to-cells map created for {} edges.".format(len(edge_to_cells_map)))

# 5. [Efficient] Build element-to-Cell mapping relationship
print("Mapping elements to parent cells...")
cell_to_elements_map = {}
all_cells_from_part = myInstance.cells
for cell in all_cells_from_part:
    # Directly get all elements owned by the cell object, extremely efficient
    elements = cell.getElements()
    cell_to_elements_map[cell.index] = [e.label for e in elements]
# Use loop accumulation instead of sum function (sum may not be available in Abaqus environment)
total_elements = 0
for v in cell_to_elements_map.values():
    total_elements += len(v)
print("Element mapping complete. Found {} elements across {} cells.".format(total_elements, len(all_cells_from_part)))

# 6. Integrate all information into a single data structure
print("\nAggregating all features into a final data structure...")
all_cells_data = []
# Sort by index to ensure consistent output order for each run
sorted_cells = sorted(all_cells_from_part, key=lambda c: c.index) 

for cell in sorted_cells:
    cell_index = cell.index
    
    # Get geometric features
    geometric_features = abaqus_utils.get_cell_geometric_features(cell, myInstance)
    
    # Prepare data to write to JSON
    geom_data_for_json = {
        'volume': geometric_features['volume'],
        'bounding_box_aspect_ratio': geometric_features['bounding_box_aspect_ratio'],
        'max_edge_curvature': geometric_features['max_edge_curvature'],
        'is_on_exterior': geometric_features['is_on_exterior'],
        'centroid_x': geometric_features['centroid_x'],
        'centroid_y': geometric_features['centroid_y'],
        'centroid_z': geometric_features['centroid_z']
    }

    # Assemble all information for current cell
    cell_data = {
        'cell_index': cell_index,
        'element_labels': sorted(cell_to_elements_map.get(cell_index, [])),
        'adjacent_cell_indices': sorted(list(cell_adjacency_map.get(cell_index, set()))),
        'geometric_features': geom_data_for_json
    }
    all_cells_data.append(cell_data)

print("Data aggregation complete.")

# 7. 保存整合后的数据到单一JSON文件
output_filename = '{}_comprehensive_data.json'.format(job_name)
try:
    with codecs.open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_cells_data, f, indent=2, ensure_ascii=False)
    print('\nComprehensive data for {} cells saved to: {}'.format(len(all_cells_data), output_filename))
except Exception as e:
    print('Error saving comprehensive data: {}'.format(str(e)))

# 8. Save edge-to-cells mapping to a separate JSON file
edge_mapping_filename = '{}_edge_to_cells.json'.format(job_name)
try:
    # Convert edge_to_cells_map to JSON-serializable format (convert key to string)
    edge_to_cells_data = {str(edge_id): cell_list for edge_id, cell_list in edge_to_cells_map.items()}
    with codecs.open(edge_mapping_filename, 'w', encoding='utf-8') as f:
        json.dump(edge_to_cells_data, f, indent=2, ensure_ascii=False)
    print('Edge-to-cells mapping for {} edges saved to: {}'.format(len(edge_to_cells_map), edge_mapping_filename))
except Exception as e:
    print('Error saving edge-to-cells mapping: {}'.format(str(e)))

print("\nScript finished successfully.")
