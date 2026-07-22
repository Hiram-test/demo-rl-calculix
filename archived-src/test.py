# -*- coding: utf-8 -*-
"""
测试脚本：根据cell_mesh_density.json绘制网格、运行FEA并提取结果进行评估

功能：
1. 读取cell_mesh_density.json文件
2. 在CAE上绘制对应的网格
3. 保存绘制好后的CAE文件
4. 运行对应的FEA
5. 提取相关属性（element数量、全局应变能等），用于评估网格密度的效果
"""
import os
import sys
import json
import subprocess
import time
from pathlib import Path

# Abaqus命令，确保'abaqus'在系统路径，或通过环境变量'ABAQUS_CMD'指定完整路径
ABAQUS_CMD = os.environ.get("ABAQUS_CMD", "abaqus")

def ensure_dir(directory_path: str) -> None:
    """确保目录存在"""
    os.makedirs(directory_path, exist_ok=True)

def run_mesh_generation(base_cae_file: str, mesh_density_file: str, output_cae_file: str, 
                        work_dir: str) -> bool:
    """
    运行网格生成脚本
    
    :param base_cae_file: 基础CAE文件路径
    :param mesh_density_file: 网格密度配置文件路径
    :param output_cae_file: 输出CAE文件路径
    :param work_dir: 工作目录
    :return: 是否成功
    """
    mesh_gen_script = os.path.abspath("mesh_generation.py")
    base_cae_file = os.path.abspath(base_cae_file)
    mesh_density_file = os.path.abspath(mesh_density_file)
    output_cae_file = os.path.abspath(output_cae_file)
    
    if os.name == 'nt':  # Windows
        command = f'{ABAQUS_CMD} cae noGUI={mesh_gen_script} -- "{base_cae_file}" "{mesh_density_file}" "{output_cae_file}"'
        use_shell = True
    else:
        command = [
            ABAQUS_CMD, 'cae',
            f'noGUI={mesh_gen_script}',
            '--',
            base_cae_file,
            mesh_density_file,
            output_cae_file
        ]
        use_shell = False
    
    print(f"[MESH] Running mesh generation...")
    print(f"  Command: {command if isinstance(command, str) else ' '.join(command)}")
    print(f"  Work directory: {work_dir}")
    
    result = subprocess.run(
        command,
        shell=use_shell,
        cwd=work_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"[MESH] Error: Mesh generation failed!")
        print(f"  Return code: {result.returncode}")
        print(f"  STDOUT:\n{result.stdout}")
        print(f"  STDERR:\n{result.stderr}")
        return False
    
    print(f"[MESH] Mesh generation completed successfully.")
    return True

def run_fea_analysis(meshed_cae_file: str, job_name: str, work_dir: str) -> tuple:
    """
    运行FEA分析脚本
    
    :param meshed_cae_file: 带网格的CAE文件路径
    :param job_name: 作业名称
    :param work_dir: 工作目录
    :return: (是否成功, 运行时间(秒))
    """
    run_fea_script = os.path.abspath("run_fea_analysis.py")
    meshed_cae_file = os.path.abspath(meshed_cae_file)
    
    if os.name == 'nt':  # Windows
        command = f'{ABAQUS_CMD} cae noGUI={run_fea_script} -- "{meshed_cae_file}" {job_name}'
        use_shell = True
    else:
        command = [
            ABAQUS_CMD, 'cae',
            f'noGUI={run_fea_script}',
            '--',
            meshed_cae_file,
            job_name
        ]
        use_shell = False
    
    print(f"[FEA] Running FEA analysis...")
    print(f"  Command: {command if isinstance(command, str) else ' '.join(command)}")
    print(f"  Work directory: {work_dir}")
    
    # 记录开始时间
    start_time = time.time()
    
    result = subprocess.run(
        command,
        shell=use_shell,
        cwd=work_dir,
        capture_output=True,
        text=True
    )
    
    # 记录结束时间并计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if result.returncode != 0:
        print(f"[FEA] Error: FEA analysis failed!")
        print(f"  Return code: {result.returncode}")
        print(f"  STDOUT:\n{result.stdout}")
        print(f"  STDERR:\n{result.stderr}")
        return False, elapsed_time
    
    print(f"[FEA] FEA analysis completed successfully.")
    print(f"[FEA] 运行时间 (Elapsed Time): {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    return True, elapsed_time

def extract_results(odb_file_path: str, work_dir: str = None, max_elements: int = 50000) -> dict:
    """
    从ODB文件中提取结果
    
    :param odb_file_path: ODB文件路径
    :param work_dir: 工作目录
    :param max_elements: 最大单元数阈值，用于计算resource_usage (默认: 50000)
    :return: 提取的结果字典，如果失败返回None
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    extract_script = os.path.join(project_root, "extract_results.py")
    odb_file_path = os.path.abspath(odb_file_path)
    
    # extract_results.py 的参数格式: <odb_path> [instance_name] [step_name]
    # 我们只传递odb_path，使用默认的instance和step
    if os.name == 'nt':  # Windows
        command = f'"{ABAQUS_CMD}" python "{extract_script}" "{odb_file_path}"'
        use_shell = True
    else:
        command = [
            ABAQUS_CMD,
            "python",
            extract_script,
            odb_file_path
        ]
        use_shell = False
    
    print(f"[EXTRACT] Extracting results from ODB...")
    print(f"  Command: {command if isinstance(command, str) else ' '.join(command)}")
    
    if work_dir is None:
        work_dir = project_root
    
    result = subprocess.run(
        command,
        shell=use_shell,
        cwd=work_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"[EXTRACT] Error: Result extraction failed!")
        print(f"  Return code: {result.returncode}")
        print(f"  STDOUT:\n{result.stdout}")
        print(f"  STDERR:\n{result.stderr}")
        return None
    
    # extract_results.py 会将结果保存到JSON文件，而不是输出到stdout
    # 生成的文件名格式: <odb_basename>_physical_features.json
    odb_dir = os.path.dirname(odb_file_path)
    odb_basename = os.path.splitext(os.path.basename(odb_file_path))[0]
    output_filename = f"{odb_basename}_physical_features.json"
    output_path = os.path.join(odb_dir, output_filename)
    
    if not os.path.exists(output_path):
        print(f"[EXTRACT] Error: Expected output file not found: {output_path}")
        return None
    
    try:
        # 从生成的JSON文件读取结果
        with open(output_path, 'r', encoding='utf-8') as f:
            state_features = json.load(f)
        
        # 添加resource_usage计算
        model_features = state_features.get('model_features', {})
        num_elements = model_features.get('num_elements', 0)
        state_features['resource_usage'] = min(1.0, num_elements / max_elements) if max_elements > 0 else 0.0
        
        print(f"[EXTRACT] Successfully extracted results:")
        print(f"  - Elements: {num_elements}")
        print(f"  - Nodes: {model_features.get('num_nodes', 0)}")
        print(f"  - ALLSE: {model_features.get('total_strain_energy', 0.0):.6e}")
        
        return state_features
    except Exception as e:
        print(f"[EXTRACT] Error: Failed to load JSON file: {e}")
        return None

def load_mesh_results(mesh_results_file: str) -> dict:
    """
    加载网格生成结果
    
    :param mesh_results_file: 网格结果JSON文件路径
    :return: 结果字典，如果文件不存在返回None
    """
    if not os.path.exists(mesh_results_file):
        return None
    
    try:
        with open(mesh_results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[LOAD] Error loading mesh results: {e}")
        return None

def load_baseline_data(baseline_cache_dir: str = 'checkpoints/baseline_cache') -> dict:
    """
    从baseline缓存目录加载baseline数据
    
    :param baseline_cache_dir: baseline缓存目录
    :return: baseline数据字典，如果找不到返回None
    """
    if not os.path.exists(baseline_cache_dir):
        print(f"[BASELINE] Warning: Baseline cache directory not found: {baseline_cache_dir}")
        return None
    
    # 查找缓存文件（通常是唯一的）
    cache_files = [f for f in os.listdir(baseline_cache_dir) if f.endswith('.json')]
    if not cache_files:
        print(f"[BASELINE] Warning: No baseline cache files found in {baseline_cache_dir}")
        return None
    
    # 使用第一个找到的缓存文件
    cache_file = os.path.join(baseline_cache_dir, cache_files[0])
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
        print(f"[BASELINE] Loaded baseline data from: {cache_file}")
        return baseline_data
    except Exception as e:
        print(f"[BASELINE] Error loading baseline data: {e}")
        return None

def evaluate_mesh_quality(mesh_results: dict, fea_results: dict, baseline_data: dict = None, fea_runtime: float = None) -> dict:
    """
    评估网格质量
    
    :param mesh_results: 网格生成结果
    :param fea_results: FEA分析结果
    :param baseline_data: baseline数据（可选）
    :param fea_runtime: FEA运行时间（秒，可选）
    :return: 评估结果字典
    """
    evaluation = {
        'mesh_statistics': {},
        'fea_statistics': {},
        'quality_metrics': {},
        'baseline_comparison': {}
    }
    
    # 网格统计信息
    if mesh_results:
        mesh_stats = mesh_results.get('mesh_statistics', {})
        evaluation['mesh_statistics'] = {
            'total_elements': mesh_stats.get('total_elements', 0),
            'total_nodes': mesh_stats.get('total_nodes', 0),
            'total_cells': mesh_stats.get('total_cells', 0),
            'cells_with_elements': mesh_stats.get('cells_with_elements', 0)
        }
    
    # FEA统计信息
    if fea_results:
        # extract_results.py 返回的数据结构:
        # {
        #   "model_features": {"total_strain_energy": ..., "num_elements": ..., "num_nodes": ...},
        #   "element_features": {...}
        # }
        model_features = fea_results.get('model_features', {})
        
        evaluation['fea_statistics'] = {
            'num_elements': model_features.get('num_elements', 0),
            'num_nodes': model_features.get('num_nodes', 0),
            'resource_usage': fea_results.get('resource_usage', 0.0),
            'allse': model_features.get('total_strain_energy', 0.0)
        }
        
        # 添加FEA运行时间
        if fea_runtime is not None:
            evaluation['fea_statistics']['runtime_seconds'] = fea_runtime
            evaluation['fea_statistics']['runtime_minutes'] = fea_runtime / 60.0
    
    # 与baseline比较
    if baseline_data and fea_results:
        baseline_allse = baseline_data.get('baseline_allse')
        current_allse = evaluation['fea_statistics'].get('allse')
        
        if baseline_allse is not None and current_allse is not None:
            delta_allse = current_allse - baseline_allse
            delta_allse_percent = (delta_allse / baseline_allse) * 100 if baseline_allse != 0 else 0
            abs_delta_percent = abs(delta_allse_percent)
            
            evaluation['baseline_comparison'] = {
                'baseline_allse': baseline_allse,
                'current_allse': current_allse,
                'delta_allse': delta_allse,
                'delta_allse_percent': delta_allse_percent,
                'abs_delta_percent': abs_delta_percent,
                'close_to_baseline': abs_delta_percent < 5.0  # 偏差小于5%认为接近baseline
            }
    
    # 质量指标
    if mesh_results and fea_results:
        mesh_stats = evaluation['mesh_statistics']
        fea_stats = evaluation['fea_statistics']
        
        # 元素密度（elements per cell）
        if mesh_stats.get('total_cells', 0) > 0:
            elements_per_cell = mesh_stats['total_elements'] / mesh_stats['total_cells']
            evaluation['quality_metrics']['elements_per_cell'] = elements_per_cell
        
        # 节点密度（nodes per element）
        if mesh_stats.get('total_elements', 0) > 0:
            nodes_per_element = mesh_stats['total_nodes'] / mesh_stats['total_elements']
            evaluation['quality_metrics']['nodes_per_element'] = nodes_per_element
        
        # 网格效率（有元素的cell占比）
        if mesh_stats.get('total_cells', 0) > 0:
            mesh_efficiency = mesh_stats['cells_with_elements'] / mesh_stats['total_cells']
            evaluation['quality_metrics']['mesh_efficiency'] = mesh_efficiency
    
    return evaluation

def print_evaluation_report(evaluation: dict):
    """
    打印评估报告
    
    :param evaluation: 评估结果字典
    """
    print("\n" + "="*80)
    print("网格质量评估报告 (MESH QUALITY EVALUATION REPORT)")
    print("="*80)
    
    # 网格统计
    print("\n[网格统计 MESH STATISTICS]")
    mesh_stats = evaluation.get('mesh_statistics', {})
    print(f"  总单元数 (Total Elements): {mesh_stats.get('total_elements', 'N/A')}")
    print(f"  总节点数 (Total Nodes): {mesh_stats.get('total_nodes', 'N/A')}")
    print(f"  总Cell数 (Total Cells): {mesh_stats.get('total_cells', 'N/A')}")
    print(f"  有单元的Cell数 (Cells with Elements): {mesh_stats.get('cells_with_elements', 'N/A')}")
    
    # FEA统计
    print("\n[有限元分析统计 FEA STATISTICS]")
    fea_stats = evaluation.get('fea_statistics', {})
    print(f"  单元数量 (Number of Elements): {fea_stats.get('num_elements', 'N/A')}")
    print(f"  节点数量 (Number of Nodes): {fea_stats.get('num_nodes', 'N/A')}")
    resource_usage = fea_stats.get('resource_usage')
    if resource_usage is not None:
        print(f"  资源使用率 (Resource Usage): {resource_usage:.4f} ({resource_usage*100:.1f}%)")
    allse = fea_stats.get('allse')
    if allse is not None:
        print(f"  全局应变能 (Global Strain Energy - ALLSE): {allse:.6e}")
    runtime_seconds = fea_stats.get('runtime_seconds')
    if runtime_seconds is not None:
        runtime_minutes = fea_stats.get('runtime_minutes', runtime_seconds / 60.0)
        print(f"  FEA运行时间 (FEA Runtime): {runtime_seconds:.2f} 秒 ({runtime_minutes:.2f} 分钟)")
    
    # Baseline比较
    baseline_comp = evaluation.get('baseline_comparison', {})
    if baseline_comp:
        print("\n[与Baseline比较 BASELINE COMPARISON]")
        print(f"  Baseline应变能 (Baseline ALLSE): {baseline_comp.get('baseline_allse', 'N/A'):.6e}")
        print(f"  当前应变能 (Current ALLSE): {baseline_comp.get('current_allse', 'N/A'):.6e}")
        delta = baseline_comp.get('delta_allse')
        if delta is not None:
            print(f"  应变能差值 (Delta ALLSE): {delta:.6e}")
            delta_percent = baseline_comp.get('delta_allse_percent', 0)
            abs_delta_percent = baseline_comp.get('abs_delta_percent', abs(delta_percent))
            print(f"  相对偏差 (Relative Deviation): {delta_percent:+.2f}%")
            print(f"  绝对偏差 (Absolute Deviation): {abs_delta_percent:.2f}%")
            
            if baseline_comp.get('close_to_baseline', False):
                print(f"  结果评价: ✓ 接近Baseline (偏差仅 {abs_delta_percent:.2f}%)")
            else:
                direction = "偏高" if delta > 0 else "偏低"
                print(f"  结果评价: ⚠ 偏离Baseline ({direction} {abs_delta_percent:.2f}%)")
    
    # 质量指标
    print("\n[质量指标 QUALITY METRICS]")
    quality_metrics = evaluation.get('quality_metrics', {})
    if 'elements_per_cell' in quality_metrics:
        print(f"  每Cell单元数 (Elements per Cell): {quality_metrics['elements_per_cell']:.2f}")
    if 'nodes_per_element' in quality_metrics:
        print(f"  每单元节点数 (Nodes per Element): {quality_metrics['nodes_per_element']:.2f}")
    if 'mesh_efficiency' in quality_metrics:
        print(f"  网格效率 (Mesh Efficiency): {quality_metrics['mesh_efficiency']:.4f}")
    
    print("\n" + "="*80)

def main():
    """
    主函数：执行完整的测试流程
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test script: Generate mesh, run FEA, and evaluate results based on cell_mesh_density.json'
    )
    parser.add_argument(
        '--mesh_density_file',
        type=str,
        help='Path to cell_mesh_density.json or last_valid_cell_mesh_density.json file',
        default='simulations/eval_002/last_valid_cell_mesh_density.json'
    )
    parser.add_argument(
        '--base-cae',
        type=str,
        default='DEMO.cae',
        help='Base CAE file path (default: example.cae)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_output',
        help='Output directory for test results (default: test_output)'
    )
    parser.add_argument(
        '--job-name',
        type=str,
        default='test_job',
        help='Job name for FEA analysis (default: test_job)'
    )
    parser.add_argument(
        '--baseline-cache-dir',
        type=str,
        default='checkpoints/baseline_cache',
        help='Baseline cache directory (default: checkpoints/baseline_cache)'
    )
    parser.add_argument(
        '--max-elements',
        type=int,
        default=30000,
        help='Maximum element count threshold for resource usage calculation (default: 50000)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.mesh_density_file):
        print(f"Error: Mesh density file not found: {args.mesh_density_file}")
        sys.exit(1)
    
    if not os.path.exists(args.base_cae):
        print(f"Error: Base CAE file not found: {args.base_cae}")
        sys.exit(1)
    
    # 创建工作目录
    work_dir = os.path.abspath(args.output_dir)
    ensure_dir(work_dir)
    
    print("="*80)
    print("网格密度测试脚本 (MESH DENSITY TEST SCRIPT)")
    print("="*80)
    print(f"网格密度文件 (Mesh Density File): {args.mesh_density_file}")
    print(f"基础CAE文件 (Base CAE File): {args.base_cae}")
    print(f"输出目录 (Output Directory): {work_dir}")
    print(f"作业名称 (Job Name): {args.job_name}")
    print(f"Baseline缓存目录 (Baseline Cache Dir): {args.baseline_cache_dir}")
    print("="*80)
    
    # 加载baseline数据
    print("\n[STEP 0] 加载Baseline数据...")
    baseline_data = load_baseline_data(args.baseline_cache_dir)
    if baseline_data:
        print(f"  ✓ 成功加载Baseline数据")
        print(f"  Baseline ALLSE: {baseline_data.get('baseline_allse', 'N/A'):.6e}")
    else:
        print(f"  ⚠ 未找到Baseline数据，将无法进行比较")
    
    # 步骤1: 读取mesh_density文件
    print("\n[STEP 1] 加载网格密度配置...")
    try:
        with open(args.mesh_density_file, 'r', encoding='utf-8') as f:
            mesh_density_data = json.load(f)
        cell_mesh_density = mesh_density_data.get('cell_mesh_density', {})
        edge_mesh_density = mesh_density_data.get('edge_mesh_density', {})
        print(f"  ✓ 已加载 {len(cell_mesh_density)} 个cell的网格密度")
        print(f"  ✓ 已加载 {len(edge_mesh_density)} 个edge的网格密度")
    except Exception as e:
        print(f"  ✗ 加载网格密度文件失败: {e}")
        sys.exit(1)
    
    # 步骤2: 生成网格
    print("\n[STEP 2] 生成网格...")
    output_cae_file = os.path.join(work_dir, f"{args.job_name}_mesh.cae")
    mesh_density_file_abs = os.path.abspath(args.mesh_density_file)
    
    if not run_mesh_generation(args.base_cae, mesh_density_file_abs, output_cae_file, work_dir):
        print("  ✗ 网格生成失败，退出。")
        sys.exit(1)
    
    # 检查生成的CAE文件是否存在
    if not os.path.exists(output_cae_file):
        print(f"  ✗ 生成的CAE文件未找到: {output_cae_file}")
        sys.exit(1)
    
    # 加载网格结果
    mesh_results_file = output_cae_file.replace('.cae', '_mesh_results.json')
    mesh_results = load_mesh_results(mesh_results_file)
    
    # 步骤3: 运行FEA分析
    print("\n[STEP 3] 运行FEA分析...")
    fea_success, fea_runtime = run_fea_analysis(output_cae_file, args.job_name, work_dir)
    if not fea_success:
        print("  ✗ FEA分析失败，退出。")
        sys.exit(1)
    
    # 检查ODB文件是否存在
    odb_file = os.path.join(work_dir, f"{args.job_name}.odb")
    if not os.path.exists(odb_file):
        print(f"  ✗ ODB文件未找到: {odb_file}")
        sys.exit(1)
    
    # 步骤4: 提取结果
    print("\n[STEP 4] 提取FEA结果...")
    fea_results = extract_results(odb_file, work_dir, args.max_elements)
    if fea_results is None:
        print("  ✗ 结果提取失败，退出。")
        sys.exit(1)
    
    # 步骤5: 评估网格质量
    print("\n[STEP 5] 评估网格质量...")
    evaluation = evaluate_mesh_quality(mesh_results, fea_results, baseline_data, fea_runtime)
    
    # 打印评估报告
    print_evaluation_report(evaluation)
    
    # 保存评估结果到JSON文件
    evaluation_file = os.path.join(work_dir, f"{args.job_name}_evaluation.json")
    try:
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        print(f"\n[保存] 评估结果已保存至: {evaluation_file}")
    except Exception as e:
        print(f"\n[保存] 警告: 保存评估结果失败: {e}")
    
    print("\n" + "="*80)
    print("✓ 测试成功完成!")
    print("="*80)
    print(f"输出文件 (Output Files):")
    print(f"  - CAE文件: {output_cae_file}")
    print(f"  - ODB文件: {odb_file}")
    print(f"  - 评估报告: {evaluation_file}")
    print("="*80)

if __name__ == '__main__':
    main()

