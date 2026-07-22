import os
import subprocess
import shutil
import json
import hashlib
from typing import Optional, List, Tuple
from typing import Dict, Any
import numpy as np


def format_action(action: int, refine_step_size: float = 0.05, coarsen_step_size: float = 0.05) -> str:
    """将action值转换为可读的字符串"""
    action_map = {
        0: f'加密 (refine, -{refine_step_size*100:.0f}%)',
        1: f'稀疏 (coarsen, +{coarsen_step_size*100:.0f}%)',
        2: 'no-op (保持不变)'
    }
    return action_map.get(action, f'unknown ({action})')

# Abaqus command, 确保 'abaqus' 在系统路径，或通过环境变量 'ABAQUS_CMD' 指定完整路径
ABAQUS_CMD = os.environ.get("ABAQUS_CMD", "F:/SIMULIA/Commands/abaqus.bat")

def ensure_clean_dir(directory_path: str) -> None:
    """
    确保工作目录存在并被清空：
    - 若不存在则创建
    - 若存在则删除其下所有文件与子目录
    - .json 文件会被保留
    注意：可能存在被占用的锁文件（如 .lck），删除失败将被忽略。
    """
    os.makedirs(directory_path, exist_ok=True)
    try:
        for entry in os.listdir(directory_path):
            full_path = os.path.join(directory_path, entry)
            try:
                if os.path.isdir(full_path) and not os.path.islink(full_path):
                    shutil.rmtree(full_path, ignore_errors=True)
                else:
                    # 跳过 .json 文件
                    if entry.endswith('.json'):
                        continue
                    try:
                        os.remove(full_path)
                    except Exception:
                        # 某些平台下的占用文件（如 .lck）可能无法删除，忽略即可
                        pass
            except Exception:
                # 单个条目清理失败不影响整体流程
                pass
    except FileNotFoundError:
        # 目录可能被并发删除，重新创建一次
        os.makedirs(directory_path, exist_ok=True)


def generate_baseline_cache_key(cae_file: str, baseline_mesh_size: float) -> str:
    """
    生成baseline缓存的唯一key，基于CAE文件名和baseline网格尺寸。
    
    :param cae_file: CAE文件路径
    :param baseline_mesh_size: baseline计算时使用的网格尺寸
    :return: 缓存key字符串
    """
    # 只使用文件名（不包含路径），避免路径变化导致缓存失效
    cae_basename = os.path.basename(cae_file)
    
    # 生成唯一的缓存key（只依赖CAE文件和baseline_mesh_size）
    cache_string = f"{cae_basename}_{baseline_mesh_size}"
    # 使用MD5生成较短的hash（也可以直接使用字符串，这里用hash避免特殊字符）
    cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
    
    return f"baseline_{cache_hash}"


def load_baseline_cache(cache_dir: str, cache_key: str) -> Optional[Dict[str, Any]]:
    """
    从缓存文件加载baseline数据（包括ALLSE值和每个cell的strain_energy）。
    
    :param cache_dir: 缓存目录
    :param cache_key: 缓存key
    :return: 包含baseline_allse和baseline_cell_strain_energy的字典，如果不存在则返回None
    """
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            baseline_allse = cache_data.get('baseline_allse')
            if baseline_allse is not None:
                print(f"[BASELINE CACHE] Loaded baseline from cache: {baseline_allse}")
                print(f"[BASELINE CACHE] Cache info: CAE={cache_data.get('cae_file')}, "
                      f"baseline_mesh_size={cache_data.get('baseline_mesh_size')}")
                
                # 加载cell_strain_energy（如果存在）
                baseline_cell_strain_energy_raw = cache_data.get('baseline_cell_strain_energy', {})
                # JSON中的key是字符串，需要转换为int
                baseline_cell_strain_energy = {int(k): v for k, v in baseline_cell_strain_energy_raw.items()}
                
                print(f"[BASELINE CACHE] Loaded baseline_cell_strain_energy for {len(baseline_cell_strain_energy)} cells")
                
                return {
                    'baseline_allse': float(baseline_allse),
                    'baseline_cell_strain_energy': baseline_cell_strain_energy
                }
        except Exception as e:
            print(f"[BASELINE CACHE] Warning: Failed to load cache file {cache_file}: {e}")
    return None


def save_baseline_cache(cache_dir: str, cache_key: str, baseline_allse: float, 
                       baseline_cell_strain_energy: Dict[int, float],
                       cae_file: str, baseline_mesh_size: float) -> None:
    """
    保存baseline数据（包括ALLSE值和每个cell的strain_energy）到缓存文件。
    
    :param cache_dir: 缓存目录
    :param cache_key: 缓存key
    :param baseline_allse: baseline ALLSE值
    :param baseline_cell_strain_energy: 每个cell的baseline strain_energy {cell_id: strain_energy}
    :param cae_file: CAE文件路径
    :param baseline_mesh_size: baseline计算时使用的网格尺寸
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    # JSON要求key为字符串，所以转换cell_id为字符串
    baseline_cell_strain_energy_str = {str(k): float(v) for k, v in baseline_cell_strain_energy.items()}
    
    cache_data = {
        'baseline_allse': float(baseline_allse),
        'baseline_cell_strain_energy': baseline_cell_strain_energy_str,
        'cae_file': os.path.basename(cae_file),
        'baseline_mesh_size': float(baseline_mesh_size),
        'timestamp': None  # 可以添加时间戳，这里暂不需要
    }
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"[BASELINE CACHE] Saved baseline to cache: {cache_file}")
        print(f"[BASELINE CACHE] Saved baseline_cell_strain_energy for {len(baseline_cell_strain_energy)} cells")
    except Exception as e:
        print(f"[BASELINE CACHE] Warning: Failed to save cache file {cache_file}: {e}")


class AbaqusEnv:
    """基于Abaqus的强化学习环境，支持cell级别的自适应网格优化"""

    def __init__(self,
                 template_cae_file: str = 'bridge.cae',
                 simulations_root: str = 'simulations',
                 cpus: int = 4,
                 max_elements: int = 50000,
                 min_elements: int = 1000,
                 cell_min_mesh_size: Optional[float] = None,
                 cell_max_mesh_size: Optional[float] = None,
                 baseline_on_reset: bool = True,
                 global_mesh_size: float = 10,
                 process_id: Optional[int] = None,
                 penalty_mesh_failure: float = -50.0,
                 penalty_fea_failure: float = -100.0,
                 penalty_file_missing: float = -200.0,
                 penalty_min_elements: float = -1.0,
                 penalty_max_elements: float = -5.0,
                 accuracy_weight: float = 1.0,
                 resource_weight: float = 1.0,
                 refine_step_size: float = 0.05,
                 coarsen_step_size: float = 0.05):
        self.template_cae_file = template_cae_file
        self.simulations_root = simulations_root
        self.cpus = cpus
        self.max_elements = max_elements
        self.min_elements = min_elements
        self.cell_min_mesh_size = cell_min_mesh_size if (cell_min_mesh_size and cell_min_mesh_size > 0) else None
        self.cell_max_mesh_size = cell_max_mesh_size if (cell_max_mesh_size and cell_max_mesh_size > 0) else None
        # 分级惩罚配置
        self.penalty_mesh_failure = penalty_mesh_failure
        self.penalty_fea_failure = penalty_fea_failure
        self.penalty_file_missing = penalty_file_missing
        self.penalty_min_elements = penalty_min_elements
        self.penalty_max_elements = penalty_max_elements
        
        # Reward平衡配置
        self.accuracy_weight = accuracy_weight
        self.resource_weight = resource_weight
        self.reward_metric = "accuracy_only"  # 奖励度量类型
        
        # 动作步长配置
        self.refine_step_size = refine_step_size
        self.coarsen_step_size = coarsen_step_size
        
        self.episode_index = 0
        self.step_index = 0
        self.baseline_allse = None
        self.initial_allse = None
        self.initial_num_elements = None  # 初始单元总数
        self.history_allse = 0  # 用于跟踪上一步的总应变能
        self.baseline_cell_strain_energy = {}
        self._last_obs = None
        self.global_mesh_size = global_mesh_size
        self.baseline_on_reset = bool(baseline_on_reset)
        self._prev_num_elements: Optional[int] = None
        
        # Cell-level data structures
        self.cell_mesh_density: Dict[int, float] = {}
        self.cell_adjacency: Dict[int, List[int]] = {}
        self.cell_to_elements_map: Dict[int, List[int]] = {}
        self.cell_geometric_features: Dict[int, List[float]] = {}
        self.edge_to_cells_map: Dict[int, List[int]] = {}
        self.cell_to_edges_map: Dict[int, List[int]] = {}
        self.edge_mesh_density_state: Dict[int, float] = {}
        
        # 保存上一步的网格统计信息，用于比较是否发生变化
        self._last_mesh_statistics: Optional[Dict[str, Any]] = None
        
        # 状态回退机制：保存操作前的状态
        self._backup_cell_mesh_density: Optional[Dict[int, float]] = None
        self._backup_edge_mesh_density: Optional[Dict[int, float]] = None
        self._consecutive_failures = 0  # 连续失败计数器
        self._max_consecutive_failures = 5  # 最大连续失败次数，超过则终止episode
        
        # 多进程支持：如果提供了process_id，创建进程专用的CAE文件副本
        self.process_id = process_id
        self._process_cae_file = None
        self._base_cae_file = None  # 保存base CAE文件路径（未网格化的原始文件）
        if process_id is not None:
            self._setup_process_cae_file()
        else:
            # 如果没有多进程，base CAE文件就是template_cae_file
            self._base_cae_file = os.path.abspath(self.template_cae_file)
    
    def _setup_process_cae_file(self):
        """为多进程环境创建CAE文件副本"""
        import os
        template_path = os.path.abspath(self.template_cae_file)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template CAE file not found: {template_path}")
        
        # 创建进程专用目录
        process_dir = os.path.join(self.simulations_root, f"_process_{self.process_id}")
        os.makedirs(process_dir, exist_ok=True)
        
        # 复制CAE文件到进程专用目录
        template_name = os.path.basename(template_path)
        process_cae_path = os.path.join(process_dir, template_name)
        shutil.copy2(template_path, process_cae_path)
        
        self._process_cae_file = process_cae_path
        self._base_cae_file = process_cae_path  # 保存base CAE文件路径
        print(f"[Process {self.process_id}] Created process-specific CAE file: {process_cae_path}")
    
    def _get_cae_file(self) -> str:
        """获取当前应该使用的CAE文件路径（可能是已网格化的文件）"""
        if self._process_cae_file is not None:
            return self._process_cae_file
        return os.path.abspath(self.template_cae_file)
    
    def _get_base_cae_file(self) -> str:
        """获取base CAE文件路径（未网格化的原始文件）"""
        if self._base_cae_file is not None:
            return self._base_cae_file
        return os.path.abspath(self.template_cae_file)
    
    def cleanup_process_files(self):
        """清理进程专用文件（可选，在进程结束时调用）"""
        if self._process_cae_file and os.path.exists(self._process_cae_file):
            try:
                process_dir = os.path.dirname(self._process_cae_file)
                # 可选：删除整个进程目录，或者只删除CAE文件
                # 为了安全，这里只删除CAE文件，保留其他可能需要的文件
                os.remove(self._process_cae_file)
                print(f"[Process {self.process_id}] Cleaned up process CAE file: {self._process_cae_file}")
            except Exception as e:
                print(f"[Process {self.process_id}] Warning: Failed to cleanup process files: {e}")

    def compute_baseline(self, cache_dir: str = 'baseline_cache', use_cache: bool = True, baseline_mesh_size: float = None):
        """计算baseline ALLSE值，支持缓存机制避免重复计算"""
        if self.baseline_allse is not None:
            print(f"[BASELINE] Baseline ALLSE already computed: {self.baseline_allse}, skipping computation.")
            return self.baseline_allse
        
        # 确定baseline mesh size
        if baseline_mesh_size is None:
            baseline_mesh_size = self.global_mesh_size
        
        # 生成缓存key（只依赖CAE文件和baseline_mesh_size）
        cae_file = self._get_base_cae_file()
        cache_key = generate_baseline_cache_key(cae_file, baseline_mesh_size)
        
        # 尝试从缓存加载
        if use_cache:
            cached_baseline_data = load_baseline_cache(cache_dir, cache_key)
            if cached_baseline_data is not None:
                self.baseline_allse = cached_baseline_data['baseline_allse']
                self.baseline_cell_strain_energy = cached_baseline_data['baseline_cell_strain_energy']
                print(f"[BASELINE] Using cached baseline ALLSE: {self.baseline_allse}")
                print(f"[BASELINE] Using cached baseline_cell_strain_energy for {len(self.baseline_cell_strain_energy)} cells")
                return self.baseline_allse
            else:
                print(f"[BASELINE] No cache found, will compute baseline and save to cache.")
        else:
            print(f"[BASELINE] Cache disabled (use_cache=False), computing baseline...")
        
        # 使用一个固定的baseline run_id
        baseline_run_id = "baseline_run"
        sim_dir = os.path.join(self.simulations_root, baseline_run_id)
        ensure_clean_dir(sim_dir)
        
        # 用baseline_mesh_size运行baseline，获取baseline的ALLSE值
        run_mesh_script = os.path.abspath("init_run_mesh.py")
        cae_file = os.path.abspath(cae_file)
        job_name_baseline = f"job_{baseline_run_id}_baseline"
        
        # 统一使用列表形式（跨平台兼容）
        command_baseline = [
            ABAQUS_CMD, 'cae',
            f'noGUI={run_mesh_script}',
            '--',
            cae_file,
            str(baseline_mesh_size),
            job_name_baseline
        ]

        print(f"[BASELINE] Computing baseline ALLSE with baseline_mesh_size={baseline_mesh_size}...")
        result_baseline = subprocess.run(
            command_baseline,
            shell=False,
            cwd=sim_dir,
            capture_output=True,
            text=True
        )

        # 临时设置run_id（如果尚未设置），以便run_static_analysis可以使用
        original_run_id = getattr(self, 'run_id', None)
        self.run_id = baseline_run_id

        # 在运行baseline分析前，需要先建立cell_to_elements_map
        # 这样才能在run_static_analysis中计算cell_strain_energy
        print(f"[BASELINE] Building cell mapping from baseline comprehensive data...")
        self._build_initial_action_record(sim_dir, job_name_baseline)

        # 运行并提取baseline状态（仅用于获取ALLSE）
        reward_dummy_baseline, info_baseline = self.run_static_analysis(sim_dir=sim_dir, job_name=job_name_baseline, input_filename=f"{job_name_baseline}.inp")
        
        # 恢复原始的run_id（如果存在）
        if original_run_id is not None:
            self.run_id = original_run_id
        else:
            # 如果原来没有run_id，清除它（但可能reset需要它，所以保留也没问题）
            pass
        
        # 从info中提取baseline的ALLSE值并保存
        if isinstance(info_baseline, dict) and 'allse' in info_baseline:
            self.baseline_allse = float(info_baseline['allse'])
            print(f"[BASELINE] Baseline ALLSE computed and saved: {self.baseline_allse}")
            
            # 保存每个cell的baseline strain_energy（从info中提取）
            if 'cell_strain_energy' in info_baseline:
                self.baseline_cell_strain_energy = info_baseline['cell_strain_energy']
                print(f"[BASELINE] Baseline cell strain_energy saved for {len(self.baseline_cell_strain_energy)} cells")
            else:
                print(f"[BASELINE] Warning: Could not find cell_strain_energy in info dict")
            
            # 保存到缓存（包括baseline_cell_strain_energy）
            if use_cache:
                save_baseline_cache(cache_dir, cache_key, self.baseline_allse, 
                                   self.baseline_cell_strain_energy,
                                   cae_file, baseline_mesh_size)
            
            return self.baseline_allse
        else:
            print(f"[BASELINE] Warning: Could not find ALLSE in info dict. Keys: {info_baseline.keys() if isinstance(info_baseline, dict) else 'N/A'}")
            return None

    def reset(self, run_id: Optional[str] = None):
        """开始新的episode"""
        self.episode_index += 1
        self.step_index = 0
        self.run_id = run_id or f"run_{self.episode_index:03d}"
        self._last_obs = None
        self._last_mesh_statistics = None
        self.history_allse = 0  # 重置历史应变能
        self._consecutive_failures = 0
        self._backup_cell_mesh_density = None
        self._backup_edge_mesh_density = None
        self.edge_mesh_density_state = {}

        # 在Episode开始时运行初始分析，获取初始观测（用于get_cell_observations）
        if self.baseline_on_reset:
            sim_dir = os.path.join(self.simulations_root, self.run_id)
            ensure_clean_dir(sim_dir)
            
            # 用global_mesh_size运行，获取初始观测（用于get_cell_observations）
            run_mesh_script = os.path.abspath("init_run_mesh.py")
            cae_file = self._get_base_cae_file()
            cae_file = os.path.abspath(cae_file)
            job_name_init = f"job_{self.run_id}_init"
            
            # 统一使用列表形式（跨平台兼容）
            command_init = [
                ABAQUS_CMD, 'cae',
                f'noGUI={run_mesh_script}',
                '--',
                cae_file,
                str(self.global_mesh_size),
                job_name_init
            ]

            print(f"[RESET] Running initial analysis with global_mesh_size={self.global_mesh_size} to get initial observations...")
            result_init = subprocess.run(
                command_init,
                shell=False,
                cwd=sim_dir,
                capture_output=True,   # 捕获输出，方便调试
                encoding='utf-8',       # <--- 告诉subprocess以utf-8解码捕获的输出
                errors='replace'       # 添加这行
            )

            # 基于生成的JSON文件构建初始动作记录（记录每个cell的网格密度）
            self._build_initial_action_record(sim_dir, job_name_init)
            
            # 运行并提取初始状态（用于观测）
            reward_dummy_init, info_init = self.run_static_analysis(sim_dir=sim_dir, job_name=job_name_init, input_filename=f"{job_name_init}.inp")
            
            # 从info中获取cell_features和global_features（用于初始观测）
            cell_features = info_init.get('cell_features', {}) if isinstance(info_init, dict) else {}
            global_features = info_init.get('global_features', {}) if isinstance(info_init, dict) else {}
            resource_usage = float(global_features.get('resource_usage', 0.0))

            # 保存 reset 时的初始 ALLSE 和 num_elements（用于后续归一化）
            try:
                if isinstance(info_init, dict) and 'allse' in info_init:
                    self.initial_allse = float(info_init.get('allse'))
                    print(f"[RESET] Saved initial ALLSE for run {self.run_id}: {self.initial_allse}")
                # 保存初始单元总数
                if isinstance(info_init, dict):
                    physical_info = info_init.get('physical_features', {})
                    if isinstance(physical_info, dict):
                        model_features = physical_info.get('model_features', {})
                        if isinstance(model_features, dict) and 'num_elements' in model_features:
                            self.initial_num_elements = int(model_features.get('num_elements', 0))
                            print(f"[RESET] Saved initial num_elements for run {self.run_id}: {self.initial_num_elements}")
                        self._prev_num_elements = self.initial_num_elements
            except Exception:
                # 不应阻塞 reset 流程，若无法解析则保留为 None
                pass
            
            if self._prev_num_elements is None:
                self._prev_num_elements = self.initial_num_elements if self.initial_num_elements is not None else 0
            
            # 构建初始观测（基于cell的观测框架）
            self._last_obs = {
                "last_reward": 0.0,
                "cell_features": cell_features,  # {cell_id: [feature_list]}
                "resource_usage": resource_usage,
                "global_features": global_features,
            }
        return self._last_obs

    def step(self, action_params: dict):
        """
        执行一步仿真。
        
        :param action_params: {cell_id: action_value} 字典，包含所有cell的动作
        :return: (obs, reward, done, info)
        """
        self.step_index += 1
        
        # 备份当前状态（用于错误时回退）
        self._backup_cell_mesh_density = self.cell_mesh_density.copy()
        self._backup_edge_mesh_density = self.edge_mesh_density_state.copy()

        # 1) 准备目录和文件名
        sim_dir = os.path.join(self.simulations_root, self.run_id)
        ensure_clean_dir(sim_dir)

        job_name = f"job_{self.run_id}"

        # action_params应该是 {cell_id: action_value} 字典
        actions_dict = action_params
        
        # 【修改】强制使用cell-based模式，因为每次只加密一个网格
        # 跳过edge转换逻辑，直接使用cell actions
        use_edge_based = False
        edge_actions = {}  # 不再使用，但保留以避免未定义变量警告
        print(f"  [ACTION RESOLUTION] Using cell-based mode (direct cell actions)")
        
        # 统计cell actions的分布
        cell_action_counts = {}
        for action in actions_dict.values():
            cell_action_counts[action] = cell_action_counts.get(action, 0) + 1
        action_summary = ", ".join([f"{format_action(a, self.refine_step_size, self.coarsen_step_size)}: {c}" for a, c in sorted(cell_action_counts.items())])
        print(f"  [CELL ACTIONS] {action_summary}")
        
        # 定义action到步长的映射
        def get_action_step_size(action_value: int) -> tuple:
            """
            返回 (action_type, step_size)
            action_type: 'refine', 'coarsen', 'no-op'
            step_size: 步长比例
            """
            if action_value == 0:
                return ('refine', self.refine_step_size)  # 加密
            elif action_value == 1:
                return ('coarsen', self.coarsen_step_size)  # 稀疏
            elif action_value == 2:
                return ('no-op', 0.0)  # 保持不变
            else:
                return ('no-op', 0.0)  # 未知动作，保持不变
        
        # 记录哪些cells选择了no-op（用于统计）
        no_op_cells = [cell_id for cell_id, action in actions_dict.items() if action == 2]
        affected_cells = [cell_id for cell_id, action in actions_dict.items() if action != 2]
        
        # 根据edge actions更新cell mesh density
        if use_edge_based:
            # 使用edge-based方法：仅对受影响cell的关联edge计算新的seed size
            edge_mesh_density: Dict[int, float] = {}
            
            for edge_id, action_value in edge_actions.items():
                # 获取这条edge当前的密度（使用相邻cells的平均密度作为参考）
                cell_ids_raw = self.edge_to_cells_map.get(edge_id, [])
                cell_ids: List[int] = []
                for cid in cell_ids_raw:
                    try:
                        cell_ids.append(int(cid))
                    except (TypeError, ValueError):
                        continue
                if not cell_ids:
                    continue
                
                # 当前密度：使用相邻cells的平均密度
                current_densities = [self.cell_mesh_density.get(cid, self.global_mesh_size) for cid in cell_ids]
                current_density = sum(current_densities) / len(current_densities)
                
                # 根据action计算新密度（使用新的步长）
                action_type, step_size = get_action_step_size(action_value)
                if action_type == 'refine':  # 加密（减小密度）
                    new_density = current_density * (1 - step_size)
                elif action_type == 'coarsen':  # 稀疏（增大密度）
                    new_density = current_density * (1 + step_size)
                else:  # no-op
                    new_density = current_density
                
                edge_mesh_density[edge_id] = new_density
            
            # 基于edge的新密度，仅更新受影响的cell
            for raw_cell_id in affected_cells:
                try:
                    cell_id = int(raw_cell_id)
                except (TypeError, ValueError):
                    continue
                edge_ids = self.cell_to_edges_map.get(cell_id, [])
                edge_densities = [edge_mesh_density[eid] for eid in edge_ids if eid in edge_mesh_density]
                if edge_densities:
                    self.cell_mesh_density[cell_id] = sum(edge_densities) / len(edge_densities)
                else:
                    # 若对应edge未发生变化，回退到直接根据action更新
                    current_density = self.cell_mesh_density.get(cell_id, self.global_mesh_size)
                    action_value = actions_dict.get(cell_id)
                    if action_value is None:
                        action_value = actions_dict.get(str(cell_id), 2)
                    action_type, step_size = get_action_step_size(action_value)
                    if action_type == 'refine':  # 加密（减小密度）
                        self.cell_mesh_density[cell_id] = current_density * (1 - step_size)
                    elif action_type == 'coarsen':  # 稀疏（增大密度）
                        self.cell_mesh_density[cell_id] = current_density * (1 + step_size)
                    else:  # no-op
                        self.cell_mesh_density[cell_id] = current_density
            
            if edge_mesh_density:
                if not isinstance(self.edge_mesh_density_state, dict):
                    self.edge_mesh_density_state = {}
                self.edge_mesh_density_state.update(edge_mesh_density)
        else:
            # 回退方法：直接使用cell actions（原来的方法）
            for cell_index_raw, action_value in actions_dict.items():
                try:
                    cell_index = int(cell_index_raw)
                except (TypeError, ValueError):
                    continue
                
                current_density = self.cell_mesh_density.get(cell_index, self.global_mesh_size)
                
                # 根据action计算新密度（使用新的步长）
                action_type, step_size = get_action_step_size(action_value)
                if action_type == 'refine':  # 加密（减小密度）
                    new_density = current_density * (1 - step_size)
                elif action_type == 'coarsen':  # 稀疏（增大密度）
                    new_density = current_density * (1 + step_size)
                else:  # no-op
                    new_density = current_density
                
                # 更新网格密度
                self.cell_mesh_density[cell_index] = new_density
            
        
        # 如果所有cell都是no-op，返回当前状态
        if len(no_op_cells) == len(actions_dict):
            print(f"  [NO-OP] All cells ({len(no_op_cells)}) selected no-op, skipping Abaqus call")
            obs = self._last_obs or {
                "last_reward": 0.0,
                "cell_features": {},
                "resource_usage": 0.0,
                "global_features": {},
            }
            info = {
                "reward_metric": self.reward_metric,
                "no_op": True,
                "no_op_cells": no_op_cells,
            }
            return obs, 0.0, False, info
        
        # 检查cell mesh size是否违反限制
        cell_mesh_violation = self._check_cell_mesh_size_constraints()
        if cell_mesh_violation:
            return self._handle_cell_mesh_size_violation(cell_mesh_violation)
        
        # 将cell_mesh_density和edge_mesh_density保存到本地JSON文件
        mesh_density_file = os.path.join(sim_dir, "cell_mesh_density.json")
        data_to_save = {
            'cell_mesh_density': self.cell_mesh_density,
            'use_edge_based': use_edge_based,
        }
        if use_edge_based and self.edge_mesh_density_state:
            data_to_save['edge_mesh_density'] = self.edge_mesh_density_state
        
        with open(mesh_density_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        # 第一步：生成网格并检查是否变化
        print(f"  [STEP 1] Running mesh generation...")
        mesh_changed = self._check_mesh_changed(sim_dir, mesh_density_file, job_name)
        
        # 验证网格生成是否成功（包括检查元素数量是否在允许范围内）
        print(f"  [STEP 1.5] Validating mesh and checking element count limits...")
        is_valid, error_message = self._validate_mesh_generation(sim_dir, job_name)
        if not is_valid:
            print(f"  [MESH/CONSTRAINT FAILURE] {error_message}")
            
            # 判断是网格生成失败还是元素数量超限或低于最小限制
            if "exceeds max limit" in error_message:
                # 超过最大限制：立即终止episode且不给负奖励
                print(f"  [HARD TERMINATE] Total elements exceeded max limit, stopping immediately without penalty")
                self._restore_mesh_state_from_backup()

                should_terminate = True
                self._consecutive_failures = self._max_consecutive_failures
                penalty_value = 0.0
                penalty_type = "max_elements_termination"

                obs = self._last_obs or {
                    "last_reward": penalty_value,
                    "cell_features": {},
                    "resource_usage": 0.0,
                    "global_features": {},
                }
                info = {
                    "reward_metric": self.reward_metric,
                    "max_elements_violation": True,
                    "error_message": error_message,
                    "penalty_type": penalty_type,
                    "penalty_value": penalty_value,
                    "consecutive_failures": self._consecutive_failures,
                    "terminated_immediately": True,
                    "state_rollback": True,
                }
                return obs, penalty_value, should_terminate, info
            elif "below min limit" in error_message:
                # 低于最小限制：触发软失败机制
                self._consecutive_failures += 1
                print(f"  [SOFT FAILURE] Consecutive failures: {self._consecutive_failures}/{self._max_consecutive_failures}")
                
                # 回退到操作前的状态
                self._restore_mesh_state_from_backup()
                
                # 检查是否超过最大连续失败次数
                should_terminate = self._consecutive_failures >= self._max_consecutive_failures
                if should_terminate:
                    print(f"  [HARD TERMINATE] Max consecutive failures reached, episode will terminate")
                else:
                    print(f"  [CONTINUE] Agent can retry with different actions")
                
                penalty_value = self.penalty_min_elements
                penalty_type = "min_elements_violation"
                
                obs = self._last_obs or {
                    "last_reward": penalty_value,
                    "cell_features": {},
                    "resource_usage": 0.0,
                    "global_features": {},
                }
                info = {
                    "reward_metric": self.reward_metric,
                    "min_elements_violation": True,
                    "error_message": error_message,
                    "penalty_type": penalty_type,
                    "penalty_value": penalty_value,
                    "consecutive_failures": self._consecutive_failures,
                    "state_rollback": True,
                }
                return obs, penalty_value, should_terminate, info
            else:
                # 网格生成失败：记录失败次数，可能回退状态
                self._consecutive_failures += 1
                print(f"  [SOFT FAILURE] Consecutive failures: {self._consecutive_failures}/{self._max_consecutive_failures}")
                
                # 回退到操作前的状态
                self._restore_mesh_state_from_backup()
                
                # 检查是否超过最大连续失败次数
                should_terminate = self._consecutive_failures >= self._max_consecutive_failures
                if should_terminate:
                    print(f"  [HARD TERMINATE] Max consecutive failures reached, episode will terminate")
                else:
                    print(f"  [CONTINUE] Agent can retry with different actions")
                
                penalty_value = self.penalty_mesh_failure
                penalty_type = "mesh_failure"
                
                obs = self._last_obs or {
                    "last_reward": penalty_value,
                    "cell_features": {},
                    "resource_usage": 0.0,
                    "global_features": {},
                }
                info = {
                    "reward_metric": self.reward_metric,
                    "mesh_generation_failed": True,
                    "mesh_error_message": error_message,
                    "penalty_type": penalty_type,
                    "penalty_value": penalty_value,
                    "consecutive_failures": self._consecutive_failures,
                    "state_rollback": True,
                }
                return obs, penalty_value, should_terminate, info
        
        if not mesh_changed:
            print(f"  [SKIP FEA] Mesh unchanged, skipping finite element analysis.")
            print(f"  [REUSE] Reusing previous FEA results if available.")
            # 网格没有变化，尝试复用之前的FEA结果（用于获取状态特征）
            # 但reward应该为0，因为这一步没有产生实际变化
            _, info = self._try_reuse_previous_results(sim_dir, job_name)
            if info is not None:
                # 成功复用结果，复用状态特征但reward设为0
                print(f"  [SUCCESS] Successfully reused previous FEA results (reward=0 for no change).")
                # 从info中获取cell_features和global_features
                cell_features = info.get('cell_features', {}) if isinstance(info, dict) else {}
                global_features = info.get('global_features', {}) if isinstance(info, dict) else {}
                resource_usage = float(global_features.get('resource_usage', 0.0))

                obs = {
                    "last_reward": 0.0,  # reward设为0，因为网格没有变化
                    "cell_features": cell_features,
                    "resource_usage": resource_usage,
                    "global_features": global_features,
                }
                self._last_obs = obs

                info = dict(info or {})
                info.update({
                    "reward_metric": self.reward_metric,
                    "mesh_unchanged": True,
                    "reused_results": True
                })
                return obs, 0.0, False, info  # reward设为0
            else:
                # 如果没有可复用的结果，仍然需要运行FEA
                print(f"  [FALLBACK] No previous results found, running FEA anyway.")
                mesh_changed = True
        
        # 第二步：如果网格变化了，运行FEA分析
        if mesh_changed:
            print(f"  [STEP 2] Mesh changed, running FEA analysis...")
            meshed_cae_file = os.path.abspath(os.path.join(sim_dir, f"{job_name}_mesh.cae"))
            
            if not os.path.exists(meshed_cae_file):
                print(f"  [ERROR] Meshed CAE file not found: {meshed_cae_file}")
                self._consecutive_failures += 1
                print(f"  [SOFT FAILURE] Consecutive failures: {self._consecutive_failures}/{self._max_consecutive_failures}")
                
                # 回退到操作前的状态
                self._restore_mesh_state_from_backup()
                
                should_terminate = self._consecutive_failures >= self._max_consecutive_failures
                if should_terminate:
                    print(f"  [HARD TERMINATE] Max consecutive failures reached, episode will terminate")
                else:
                    print(f"  [CONTINUE] Agent can retry with different actions")
                
                obs = self._last_obs or {
                    "last_reward": self.penalty_file_missing,
                    "cell_features": {},
                    "resource_usage": 0.0,
                    "global_features": {},
                }
                info = {
                    "reward_metric": self.reward_metric,
                    "meshed_cae_missing": True,
                    "error_message": f"Meshed CAE file not found: {meshed_cae_file}",
                    "penalty_type": "file_missing",
                    "penalty_value": self.penalty_file_missing,
                    "consecutive_failures": self._consecutive_failures,
                    "state_rollback": True,
                }
                return obs, self.penalty_file_missing, should_terminate, info
            
            run_fea_script = os.path.abspath("run_fea_analysis.py")
            command = [
                ABAQUS_CMD, 'cae',
                f'noGUI={run_fea_script}',
                '--',
                meshed_cae_file,
                job_name
            ]

            print(f"  [FEA] Running FEA analysis with meshed CAE: {meshed_cae_file}")
            result = subprocess.run(
                command,
                shell=False,
                cwd=sim_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"  [ERROR] FEA analysis failed!")
                print(f"  [ERROR] Return code: {result.returncode}")
                print(f"  [ERROR] STDERR: {result.stderr}")
                print(f"  [ERROR] STDOUT: {result.stdout}")
                self._consecutive_failures += 1
                print(f"  [SOFT FAILURE] Consecutive failures: {self._consecutive_failures}/{self._max_consecutive_failures}")
                
                # 回退到操作前的状态
                self._restore_mesh_state_from_backup()
                
                should_terminate = self._consecutive_failures >= self._max_consecutive_failures
                if should_terminate:
                    print(f"  [HARD TERMINATE] Max consecutive failures reached, episode will terminate")
                else:
                    print(f"  [CONTINUE] Agent can retry with different actions")
                
                obs = self._last_obs or {
                    "last_reward": self.penalty_fea_failure,
                    "cell_features": {},
                    "resource_usage": 0.0,
                    "global_features": {},
                }
                info = {
                    "reward_metric": self.reward_metric,
                    "fea_failed": True,
                    "error_message": result.stderr,
                    "error_stdout": result.stdout,
                    "return_code": result.returncode,
                    "penalty_type": "fea_failure",
                    "penalty_value": self.penalty_fea_failure,
                    "consecutive_failures": self._consecutive_failures,
                    "state_rollback": True,
                }
                return obs, self.penalty_fea_failure, should_terminate, info

            # 基于生成的JSON文件构建动作记录（记录每个cell的网格密度）
            self._build_iter_action_record(sim_dir, job_name)

        # 注意：元素数量检查已经在 _validate_mesh_generation 中完成
        # 这里直接运行并提取状态
        reward, info = self.run_static_analysis(sim_dir=sim_dir, job_name=job_name, input_filename=f"{job_name}.inp")
        
        # 成功执行，重置连续失败计数器
        self._consecutive_failures = 0
        
        # 保存最后一个成功的网格密度配置
        last_valid_mesh_density_file = os.path.join(sim_dir, "last_valid_cell_mesh_density.json")
        try:
            last_valid_data = {
                'cell_mesh_density': self.cell_mesh_density,
                'use_edge_based': use_edge_based,
                'episode_info': {
                    'run_id': self.run_id,
                    'step_index': self.step_index,
                    'timestamp': None  # 可以添加时间戳
                }
            }
            if use_edge_based and self.edge_mesh_density_state:
                last_valid_data['edge_mesh_density'] = self.edge_mesh_density_state
            
            with open(last_valid_mesh_density_file, 'w') as f:
                json.dump(last_valid_data, f, indent=2)
            print(f"  [SUCCESS] Saved last valid mesh density to: {last_valid_mesh_density_file}")
        except Exception as e:
            print(f"  [WARNING] Failed to save last valid mesh density: {e}")
        
        # 从info中获取cell_features和global_features
        cell_features = info.get('cell_features', {}) if isinstance(info, dict) else {}
        global_features = info.get('global_features', {}) if isinstance(info, dict) else {}
        resource_usage = float(global_features.get('resource_usage', 0.0))

        obs = {
            "last_reward": reward,
            "cell_features": cell_features,  # {cell_id: [feature_list]}
            "resource_usage": resource_usage,
            "global_features": global_features,
        }
        self._last_obs = obs

        # 单步情景：每次 step 后即结束
        done = False

        # 获取总元素数（用于info记录）
        total_elements = sum(len(elems) for elems in self.cell_to_elements_map.values())
        
        info = dict(info or {})
        info.update({
            "total_elements": total_elements,
            "consecutive_failures": 0,
        })

        return obs, reward, done, info

    def _check_cell_mesh_size_constraints(self) -> Optional[Dict[str, list]]:
        """
        检查所有cell的mesh size是否在允许范围内。
        返回字典 {'min': [...], 'max': [...]}，若没有违反则返回None。
        """
        if self.cell_min_mesh_size is None and self.cell_max_mesh_size is None:
            return None
        
        violating_min = []
        violating_max = []
        tolerance = 1e-6
        
        for cell_id, mesh_size in self.cell_mesh_density.items():
            if self.cell_min_mesh_size is not None and mesh_size < self.cell_min_mesh_size - tolerance:
                violating_min.append({
                    "cell_id": cell_id,
                    "mesh_size": float(mesh_size)
                })
            if self.cell_max_mesh_size is not None and mesh_size > self.cell_max_mesh_size + tolerance:
                violating_max.append({
                    "cell_id": cell_id,
                    "mesh_size": float(mesh_size)
                })
        
        if violating_min or violating_max:
            return {"min": violating_min, "max": violating_max}
        return None

    def _handle_cell_mesh_size_violation(self, violation_details: Dict[str, list]):
        """触发cell mesh size范围违规的软失败机制"""
        violation_type = 'max' if violation_details.get('max') else 'min'
        violating_cells = violation_details.get(violation_type, [])
        if not violating_cells:
            return self._last_obs, 0.0, False, {"warning": "Invalid violation data"}
        
        min_bound = self.cell_min_mesh_size if self.cell_min_mesh_size is not None else float('-inf')
        max_bound = self.cell_max_mesh_size if self.cell_max_mesh_size is not None else float('inf')
        sample_cell = violating_cells[0]
        comparison = "exceeds" if violation_type == 'max' else "falls below"
        bound_value = max_bound if violation_type == 'max' else min_bound
        error_message = (f"Cell mesh size violation: {len(violating_cells)} cell(s) {comparison} "
                         f"allowed range [{min_bound}, {max_bound}]. Example cell {sample_cell['cell_id']} "
                         f"has mesh size {sample_cell['mesh_size']:.3f} (limit {bound_value}).")
        print(f"  [CELL MESH RANGE] {error_message}")
        
        self._consecutive_failures += 1
        print(f"  [SOFT FAILURE] Consecutive failures: {self._consecutive_failures}/{self._max_consecutive_failures}")
        
        self._restore_mesh_state_from_backup()
        
        should_terminate = self._consecutive_failures >= self._max_consecutive_failures
        if should_terminate:
            print(f"  [HARD TERMINATE] Max consecutive failures reached, episode will terminate")
        else:
            print(f"  [CONTINUE] Agent can retry with different actions")
        
        penalty_value = self.penalty_max_elements if violation_type == 'max' else self.penalty_min_elements
        penalty_type = "cell_max_mesh_size_violation" if violation_type == 'max' else "cell_min_mesh_size_violation"
        
        obs = self._last_obs or {
            "last_reward": penalty_value,
            "cell_features": {},
            "resource_usage": 0.0,
            "global_features": {},
        }
        info = {
            "reward_metric": self.reward_metric,
            "cell_mesh_size_violation": True,
            "violation_type": penalty_type,
            "error_message": error_message,
            "penalty_value": penalty_value,
            "violating_cells": violating_cells[:10],  # 避免info过大，示例最多10个
            "consecutive_failures": self._consecutive_failures,
            "state_rollback": True,
        }
        return obs, penalty_value, should_terminate, info

    def _restore_mesh_state_from_backup(self):
        """恢复最近一次备份的cell/edge网格状态"""
        restored = False
        if self._backup_cell_mesh_density is not None:
            self.cell_mesh_density = self._backup_cell_mesh_density.copy()
            restored = True
        if self._backup_edge_mesh_density is not None:
            self.edge_mesh_density_state = self._backup_edge_mesh_density.copy()
            restored = True
        if restored:
            print(f"  [STATE ROLLBACK] Mesh density reverted to previous state")

    def get_cell_observations(self) -> dict:
        """获取每个cell的观测，包含自身与邻居cell的特征"""
        cell_features = self._last_obs.get('cell_features', {})
        global_features = {'resource_usage': self._last_obs.get('resource_usage', 0.0)}
        return self._build_cell_observations(cell_features, global_features)

    def _convert_cell_actions_to_edge_actions(self, cell_actions: Dict[int, int]) -> Dict[int, int]:
        """
        将cell级别的actions转换为edge级别的actions，使用众数策略解决冲突。
        如果有多个action得票相同，使用保守策略（优先no-op，或在refine/coarsen之间选择no-op）。
        """
        from collections import Counter
        
        if not self.edge_to_cells_map or not self.cell_to_edges_map:
            print(f"  [WARNING] edge_to_cells_map is empty, cannot convert to edge actions")
            return {}
        
        def get_action_type_and_size(action: int) -> tuple:
            """返回 (type, size): type='refine'/'coarsen'/'no-op', size=0表示小步长"""
            if action == 0:
                return ('refine', 0)  # 小幅加密
            elif action == 1:
                return ('coarsen', 0)  # 小幅稀疏
            elif action == 2:
                return ('no-op', 0)  # 保持不变
            else:
                return ('no-op', 0)
        
        def resolve_conflict(tied_actions: List[int]) -> int:
            """
            解决冲突：当多个action得票相同时，选择最保守的action
            策略：
            1. 如果有no-op (2)，选择no-op
            2. 如果既有refine又有coarsen，选择no-op (2) 作为折中
            3. 如果只有一种类型，直接选择该动作
            """
            # 策略1: 优先选择no-op
            if 2 in tied_actions:
                return 2
            
            # 按操作类型分组
            refine_actions = [a for a in tied_actions if a == 0]
            coarsen_actions = [a for a in tied_actions if a == 1]
            
            # 策略2: 如果既有refine又有coarsen，选择no-op作为折中
            if refine_actions and coarsen_actions:
                return 2  # no-op
            
            # 策略3: 只有一种类型，直接选择该动作
            if refine_actions:
                return 0  # 小幅加密
            elif coarsen_actions:
                return 1  # 小幅稀疏
            
            # 默认返回no-op（不应该到达这里）
            return 2
        
        edge_actions = {}
        edge_votes: Dict[int, List[int]] = {}
        
        for raw_cell_id, action in cell_actions.items():
            try:
                cell_id = int(raw_cell_id)
            except (TypeError, ValueError):
                continue
            if cell_id not in self.cell_to_edges_map:
                continue
            for edge_id in self.cell_to_edges_map[cell_id]:
                edge_votes.setdefault(edge_id, []).append(action)
        
        for edge_id, votes in edge_votes.items():
            if not votes:
                continue
            action_counts = Counter(votes)
            most_common = action_counts.most_common()
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                edge_actions[edge_id] = most_common[0][0]
            else:
                tied_actions = [action for action, count in most_common if count == most_common[0][1]]
                edge_actions[edge_id] = resolve_conflict(tied_actions)
        
        return edge_actions
    
    def _build_cell_observations(self, cell_features: dict, global_features: dict = None) -> dict:
        """构建每个cell的观测，包含自身与邻居cell的特征"""
        if global_features is None:
            global_features = {}
        
        cell_obs: dict = {}
        
        # 确保cell_features的键为整数
        cell_features_int = {}
        for k, v in cell_features.items():
            try:
                cell_features_int[int(k)] = list(v) if isinstance(v, (list, tuple)) else [float(v)]
            except (ValueError, TypeError):
                continue
        
        # 遍历所有cell构建观测
        for cell_id, self_features in cell_features_int.items():
            # 获取当前cell的自身特征
            self_feat = list(self_features)  # 确保是列表
            
            # 获取相邻cell的特征
            neighbor_feats = []
            neighbor_cell_ids = self.cell_adjacency.get(cell_id, [])
            
            for neighbor_id in neighbor_cell_ids:
                neighbor_id_int = int(neighbor_id) if not isinstance(neighbor_id, int) else neighbor_id
                if neighbor_id_int in cell_features_int:
                    neighbor_features = cell_features_int[neighbor_id_int]
                    neighbor_feats.append({
                        'cell_id': neighbor_id_int,
                        'features': list(neighbor_features),  # 相邻cell的特征向量
                    })
            
            cell_obs[int(cell_id)] = {
                'self': self_feat,  # 当前cell的特征向量
                'neighbors': neighbor_feats,  # 相邻cell的特征列表
            }
        
        return cell_obs

    def _normalize_cell_geometric_features(self):
        """归一化cell几何特征（前7列），保留第8列mesh_size不归一化"""
        if not self.cell_geometric_features:
            return
        
        data = np.array(list(self.cell_geometric_features.values()))
        if data.shape[0] > 0 and data.shape[1] >= 7:
            # 只对前7列做归一化
            min_vals = data[:, :7].min(axis=0)
            max_vals = data[:, :7].max(axis=0)
            denoms = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)
            normed = (data[:, :7] - min_vals) / denoms
            
            # 如果有第8列（mesh_size），拼接上；否则只保留归一化的7列
            if data.shape[1] >= 8:
                normed_full = np.concatenate([normed, data[:, 7:8]], axis=1)
            else:
                normed_full = normed
            
            # 写回
            for i, cell_index in enumerate(self.cell_geometric_features.keys()):
                self.cell_geometric_features[cell_index] = list(normed_full[i])

    def _build_initial_action_record(self, sim_dir: str, job_name: str):
        """构建初始动作记录，读取cell的相邻关系、element映射、几何特征和edge到cells的映射"""
        # 综合JSON文件路径：{job_name}_comprehensive_data.json
        comprehensive_json_path = os.path.join(sim_dir, f"{job_name}_comprehensive_data.json")
        
        if not os.path.exists(comprehensive_json_path):
            print(f'Error: Comprehensive data file not found: {comprehensive_json_path}')
            return
        
        # Edge到cells映射文件路径：{job_name}_edge_to_cells.json
        edge_to_cells_json_path = os.path.join(sim_dir, f"{job_name}_edge_to_cells.json")
        # 每次重建，避免遗留旧映射
        self.edge_to_cells_map = {}
        self.cell_to_edges_map = {}
        if not os.path.exists(edge_to_cells_json_path):
            print(f'Warning: Edge-to-cells mapping file not found: {edge_to_cells_json_path}')
            print(f'         Edge-based action resolution will not be available.')
        else:
            # 读取edge到cells的映射
            with open(edge_to_cells_json_path, 'r', encoding='utf-8') as f:
                edge_to_cells_raw = json.load(f)
            # JSON中的key是字符串，转换为int
            self.edge_to_cells_map = {int(k): v for k, v in edge_to_cells_raw.items()}
            print(f'Loaded edge-to-cells mapping for {len(self.edge_to_cells_map)} edges')
            # 构建cell到edges的反向映射，便于局部更新
            self.cell_to_edges_map = {}
            for edge_id, cell_ids in self.edge_to_cells_map.items():
                for cell_id in cell_ids:
                    cell_id_int = int(cell_id)
                    if cell_id_int not in self.cell_to_edges_map:
                        self.cell_to_edges_map[cell_id_int] = []
                    self.cell_to_edges_map[cell_id_int].append(edge_id)
        
        # 读取综合JSON文件
        with open(comprehensive_json_path, 'r', encoding='utf-8') as f:
            comprehensive_data = json.load(f)
        
        # 初始化字典
        self.cell_to_elements_map = {}
        self.cell_adjacency = {}
        self.cell_geometric_features = {}
        
        # 遍历综合数据，提取所有信息
        for cell_data in comprehensive_data:
            cell_index = int(cell_data['cell_index'])
            
            # 提取element映射
            element_labels = [int(elem) for elem in cell_data.get('element_labels', [])]
            self.cell_to_elements_map[cell_index] = element_labels
            
            # 提取相邻cell信息
            adjacent_cells = [int(adj) for adj in cell_data.get('adjacent_cell_indices', [])]
            self.cell_adjacency[cell_index] = adjacent_cells
            
            # 提取几何特征并组合成一行特征向量
            geometric_features = cell_data.get('geometric_features', {})
            
            # 组合成一行特征向量: [volume, bounding_box_aspect_ratio, max_edge_curvature, is_on_exterior, centroid_x, centroid_y, centroid_z, mesh_size]
            self.cell_geometric_features[cell_index] = [
                float(geometric_features.get('volume', 0.0)),
                float(geometric_features.get('bounding_box_aspect_ratio', 1.0)),
                float(geometric_features.get('max_edge_curvature', 0.0)),
                float(geometric_features.get('is_on_exterior', 0.0)),
                float(geometric_features.get('centroid_x', 0.0)),
                float(geometric_features.get('centroid_y', 0.0)),
                float(geometric_features.get('centroid_z', 0.0)),
            ]
            
            # 初始时所有cell都使用全局网格大小
            self.cell_mesh_density[cell_index] = float(self.global_mesh_size)
        
        # 归一化几何特征
        self._normalize_cell_geometric_features()
        
        print(f'Loaded comprehensive cell data: {len(self.cell_to_elements_map)} cells')
        print(f'  - Elements mapped: {sum(len(elems) for elems in self.cell_to_elements_map.values())} elements')
        print(f'  - Cells with neighbors: {len(self.cell_adjacency)} cells')
        print(f'  - Geometric features loaded: {len(self.cell_geometric_features)} cells')

    def _check_mesh_changed(self, sim_dir: str, mesh_density_file: str, job_name: str) -> bool:
        """调用mesh_generation.py生成网格，并比较与上一步的网格统计信息是否相同"""
        cae_file = os.path.abspath(self._get_base_cae_file())
        output_cae_file = os.path.abspath(os.path.join(sim_dir, f"{job_name}_mesh.cae"))
        mesh_density_file_abs = os.path.abspath(mesh_density_file)
        mesh_gen_script = os.path.abspath("mesh_generation.py")
        
        command = [
            ABAQUS_CMD, 'cae',
            f'noGUI={mesh_gen_script}',
            '--',
            cae_file,
            mesh_density_file_abs,
            output_cae_file
        ]
        
        print(f"    Running mesh generation script...")
        result = subprocess.run(
            command,
            shell=False,
            cwd=sim_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"    Warning: mesh_generation.py failed, assuming mesh changed.")
            print(f"    Error: {result.stderr}")
            return True  # 如果失败，保守地假设网格变化了
        
        # 更新进程CAE文件路径为带网格的CAE文件
        if os.path.exists(output_cae_file):
            self._process_cae_file = output_cae_file
            print(f"    Updated process CAE file to meshed CAE: {output_cae_file}")
        else:
            print(f"    Warning: Meshed CAE file not found: {output_cae_file}")
        
        # 读取生成的网格统计信息
        mesh_results_file = output_cae_file.replace('.cae', '_mesh_results.json')
        if not os.path.exists(mesh_results_file):
            print(f"    Warning: mesh results file not found: {mesh_results_file}, assuming mesh changed.")
            return True
        
        try:
            with open(mesh_results_file, 'r', encoding='utf-8') as f:
                current_mesh_results = json.load(f)
            
            current_stats = current_mesh_results.get('mesh_statistics', {})
            # JSON中的key是字符串，需要转换为int以匹配其他地方使用的key类型
            current_cell_counts_raw = current_mesh_results.get('cell_element_counts', {})
            current_cell_counts = {int(k): v for k, v in current_cell_counts_raw.items()}
            
            # 如果没有上一步的统计信息，认为网格变化了（首次运行）
            if self._last_mesh_statistics is None:
                print(f"    First run, mesh statistics saved for future comparison.")
                self._last_mesh_statistics = {
                    'mesh_statistics': current_stats,
                    'cell_element_counts': current_cell_counts
                }
                return True
            
            # 比较网格统计信息
            last_stats = self._last_mesh_statistics.get('mesh_statistics', {})
            last_cell_counts = self._last_mesh_statistics.get('cell_element_counts', {})
            
            # 比较总体统计和cell元素数量
            if (current_stats.get('total_elements') != last_stats.get('total_elements') or
                current_stats.get('total_nodes') != last_stats.get('total_nodes') or
                current_cell_counts != last_cell_counts):
                print(f"    Mesh changed.")
                self._last_mesh_statistics = {
                    'mesh_statistics': current_stats,
                    'cell_element_counts': current_cell_counts
                }
                return True
            
            # 网格没有变化
            if os.path.exists(output_cae_file):
                self._process_cae_file = output_cae_file
            print(f"    Mesh unchanged.")
            return False
            
        except Exception as e:
            print(f"    Error comparing mesh statistics: {e}, assuming mesh changed.")
            return True
    
    def _validate_mesh_generation(self, sim_dir: str, job_name: str) -> Tuple[bool, Optional[str]]:
        """验证网格生成是否成功，检查所有cell是否都有元素，并检查总元素数是否在允许范围内"""
        # 读取网格结果文件
        output_cae_file = os.path.abspath(os.path.join(sim_dir, f"{job_name}_mesh.cae"))
        mesh_results_file = output_cae_file.replace('.cae', '_mesh_results.json')
        
        if not os.path.exists(mesh_results_file):
            return False, f"Mesh results file not found: {mesh_results_file}"
        
        try:
            with open(mesh_results_file, 'r', encoding='utf-8') as f:
                mesh_results = json.load(f)
            
            # 获取每个cell的元素数量（JSON中的key是字符串，需要转换为int）
            cell_element_counts_raw = mesh_results.get('cell_element_counts', {})
            cell_element_counts = {int(k): v for k, v in cell_element_counts_raw.items()}
            
            # 检查所有应该生成网格的cell（在cell_mesh_density中的cell）是否都有元素
            failed_cells = []
            for cell_index in self.cell_mesh_density.keys():
                element_count = cell_element_counts.get(cell_index, 0)
                if element_count == 0:
                    failed_cells.append(cell_index)
            
            if failed_cells:
                error_message = f"Mesh generation failed for {len(failed_cells)} cell(s): {sorted(failed_cells)}. These cells have 0 elements."
                print(f"  [MESH VALIDATION] {error_message}")
                return False, error_message
            
            # 获取总元素数
            mesh_statistics = mesh_results.get('mesh_statistics', {})
            total_elements = mesh_statistics.get('total_elements', 0)
            
            # 检查总元素数是否达到或超过最大限制
            if total_elements >= self.max_elements:
                exceed_ratio = (total_elements - self.max_elements) / max(self.max_elements, 1)
                status_message = (f"Total elements {total_elements} exceeds max limit {self.max_elements} "
                                  f"(+{exceed_ratio*100:.1f}%). Max capacity reached.")
                print(f"  [MAX ELEMENTS REACHED] {status_message}")
                return False, status_message
            
            # 检查总元素数是否低于最小限制
            # 触发软失败机制：回退状态、累计失败、给予惩罚
            if total_elements < self.min_elements:
                deficit_ratio = (self.min_elements - total_elements) / self.min_elements
                error_message = (f"Total elements {total_elements} below min limit {self.min_elements} "
                                 f"(-{deficit_ratio*100:.1f}%). Triggering soft failure mechanism.")
                print(f"  [ELEMENT COUNT VIOLATION] {error_message}")
                # 返回 False 触发软失败机制
                return False, error_message
            
            print(f"  [MESH VALIDATION] All {len(self.cell_mesh_density)} cells successfully generated mesh.")
            print(f"  [MESH VALIDATION] Total elements: {total_elements} (within range [{self.min_elements}, {self.max_elements}])")
            return True, None
            
        except Exception as e:
            error_message = f"Error validating mesh generation: {e}"
            print(f"  [MESH VALIDATION] {error_message}")
            return False, error_message
    
    def _try_reuse_previous_results(self, sim_dir: str, job_name: str) -> Tuple[Optional[float], Optional[dict]]:
        """尝试复用之前的FEA结果"""
        odb_path = os.path.join(sim_dir, f"{job_name}.odb")
        if os.path.exists(odb_path):
            print(f"    Found existing ODB file, extracting state features...")
            try:
                reward, info = self.run_static_analysis(sim_dir=sim_dir, job_name=job_name, input_filename=f"{job_name}.inp")
                return reward, info
            except Exception as e:
                print(f"    Error extracting results: {e}")
        
        # 复用上一次的状态特征
        if self._last_obs is not None:
            print(f"    Reusing previous state features.")
            info = {
                "cell_features": self._last_obs.get('cell_features', {}),
                "global_features": {"resource_usage": self._last_obs.get('resource_usage', 0.0)},
                "reused_from_last_obs": True
            }
            return None, info
        
        print(f"    No previous results available.")
        return None, None

    def _build_iter_action_record(self, sim_dir: str, job_name: str):
        """构建迭代动作记录，更新element映射和几何特征"""
        # 综合JSON文件路径：{job_name}_comprehensive_data.json
        comprehensive_json_path = os.path.join(sim_dir, f"{job_name}_comprehensive_data.json")
        
        if not os.path.exists(comprehensive_json_path):
            print(f'Error: Comprehensive data file not found: {comprehensive_json_path}')
            return
        
        # 读取综合JSON文件
        with open(comprehensive_json_path, 'r', encoding='utf-8') as f:
            comprehensive_data = json.load(f)
        
        # 保存旧的element映射用于比较
        old_cell_to_elements_map = self.cell_to_elements_map.copy() if self.cell_to_elements_map else {}
        
        # 初始化字典
        self.cell_to_elements_map = {}
        self.cell_geometric_features = {}
        
        # 遍历综合数据，提取所有信息
        for cell_data in comprehensive_data:
            cell_index = int(cell_data['cell_index'])
            
            # 提取element映射
            element_labels = [int(elem) for elem in cell_data.get('element_labels', [])]
            self.cell_to_elements_map[cell_index] = element_labels
            
            # 提取几何特征并组合成一行特征向量
            geometric_features = cell_data.get('geometric_features', {})
            
            # 组合成一行特征向量: [volume, bounding_box_aspect_ratio, max_edge_curvature, is_on_exterior, centroid_x, centroid_y, centroid_z, mesh_size]
            self.cell_geometric_features[cell_index] = [
                float(geometric_features.get('volume', 0.0)),
                float(geometric_features.get('bounding_box_aspect_ratio', 1.0)),
                float(geometric_features.get('max_edge_curvature', 0.0)),
                float(geometric_features.get('is_on_exterior', 0.0)),
                float(geometric_features.get('centroid_x', 0.0)),
                float(geometric_features.get('centroid_y', 0.0)),
                float(geometric_features.get('centroid_z', 0.0)),
            ]
        
        # 归一化几何特征
        self._normalize_cell_geometric_features()
        
        # 比较新旧element数量，找出发生变化的cell
        changed_cells = []
        all_cell_ids = set(old_cell_to_elements_map.keys()) | set(self.cell_to_elements_map.keys())
        
        for cell_index in all_cell_ids:
            old_count = len(old_cell_to_elements_map.get(cell_index, []))
            new_count = len(self.cell_to_elements_map.get(cell_index, []))
            
            if old_count != new_count:
                changed_cells.append({
                    'cell_index': cell_index,
                    'old_count': old_count,
                    'new_count': new_count,
                    'change': new_count - old_count
                })
        
        # 打印发生变化的cell信息
        if changed_cells:
            print(f'\n[Element Count Changes] Found {len(changed_cells)} cells with changed element counts:')
            for change_info in sorted(changed_cells, key=lambda x: x['cell_index']):
                print(f'  Cell {change_info["cell_index"]}: {change_info["old_count"]} -> {change_info["new_count"]} '
                      f'(change: {change_info["change"]:+d})')
        else:
            print(f'\n[Element Count Changes] No cells with changed element counts.')
        
        print(f'Loaded comprehensive cell data: {len(self.cell_to_elements_map)} cells')
        print(f'  - Elements: {sum(len(elems) for elems in self.cell_to_elements_map.values())}')
        print(f'  - Features loaded for {len(self.cell_geometric_features)} cells')

    def run_static_analysis(self, sim_dir: str, job_name: str, input_filename: str) -> Tuple[float, dict]:
        """运行Abaqus静力学分析并提取奖励"""
        
        # 检查 ODB 是否生成
        odb_path = os.path.join(sim_dir, f"{job_name}.odb")
        if not os.path.exists(odb_path):
            print(f"[{self.run_id}] Error: ODB file not found at {odb_path}")
            return self.penalty_file_missing, {"error": "odb_missing", "penalty_type": "file_missing"}

        # 调用提取脚本（扩展状态特征JSON）
        project_root = os.path.dirname(os.path.abspath(__file__))
        extract_command = f'{ABAQUS_CMD} python "{os.path.join(project_root, "extract_results.py")}" "{odb_path}"'
        print(f"[{self.run_id}] Extracting extended state features... Command: {extract_command}")

        extract_process = subprocess.run(
            extract_command,
            shell=True,
            cwd=project_root,
            capture_output=True,
            text=True
        )

        if extract_process.returncode != 0:
            print(f"[{self.run_id}] Result extraction script failed!")
            print(extract_process.stderr)
            return self.penalty_fea_failure, {"error": "extract_failed", "stderr": extract_process.stderr, "penalty_type": "fea_failure"}

        try:
            # 构造输出文件路径
            import json
            odb_dir = os.path.dirname(odb_path)
            odb_basename = os.path.splitext(os.path.basename(odb_path))[0]
            output_filename = f"{odb_basename}_physical_features.json"
            output_path = os.path.join(odb_dir, output_filename)
            
            # 读取JSON文件
            if not os.path.exists(output_path):
                print(f"[{self.run_id}] Error: Physical features file not found at {output_path}")
                return self.penalty_file_missing, {"error": "features_file_missing", "penalty_type": "file_missing"}
            
            with open(output_path, 'r') as f:
                physical_features = json.load(f)
            
            # 将新格式转换为代码期望的格式
            state_features = {
                'model_features': physical_features.get('model_features', {}),
                'element_features': physical_features.get('element_features', {}),
                'num_elements': physical_features.get('model_features', {}).get('num_elements', 0),
                'num_nodes': physical_features.get('model_features', {}).get('num_nodes', 0),
            }

            # 获取当前ALLSE值（total_strain_energy）
            current_allse = state_features['model_features'].get('total_strain_energy')

            # 从新格式的 element_features 提取特征列表
            # 新格式: {"1": {"mises": ..., "strain_energy_density": ..., "s11": ..., ...}, ...}
            # 需要转换为: {1: [feature1, feature2, ...], ...}
            element_features = {}
            raw_element_features = state_features.get('element_features', {})
            
            # ========== 全局级别奖励计算 ==========
            # 所有cell共享同一个基于精度的全局reward值
            
            # 获取当前总单元数
            current_num_elements = state_features.get('num_elements', 0)
            
            # 计算全局归一化因子
            num_cells = len(self.cell_to_elements_map)
            normalization_factor_accuracy = 1.0
            
            # 全局精度归一化因子：使用整体的应变能差异（不除以num_cells）
            if self.baseline_allse is not None and self.initial_allse is not None:
                gap = abs(self.baseline_allse - self.initial_allse)
                if gap > 1e-8:
                    normalization_factor_accuracy = gap
                else:
                    # 如果gap太小（baseline和initial几乎相同），回退到使用baseline值
                    normalization_factor_accuracy = self.baseline_allse if self.baseline_allse > 1e-8 else 1.0
            elif self.baseline_allse is not None and self.baseline_allse > 1e-8:
                # 如果没有initial_allse，回退到使用baseline值
                normalization_factor_accuracy = self.baseline_allse
            
            # 归一化因子平均到每个cell
            if num_cells > 0:
                normalization_factor_accuracy /= num_cells
            
            # 计算全局精度奖励
            if self.baseline_allse is None:
                # 如果baseline还没有计算，跳过精度奖励计算
                global_accuracy_reward = 0.0
            elif self.history_allse != 0:  # 非首次step
                # 计算应变能差异的改进
                last_allse_diff = abs(self.history_allse - self.baseline_allse)
                current_allse_diff = abs(current_allse - self.baseline_allse)
                # 如果差异减小，说明精度提高了
                global_accuracy_reward = last_allse_diff - current_allse_diff
            else:
                # 第一次step，基于与baseline的绝对差异
                current_allse_diff = abs(current_allse - self.baseline_allse)
                global_accuracy_reward = -current_allse_diff
            
            # 归一化全局精度奖励
            normalized_global_accuracy = global_accuracy_reward / normalization_factor_accuracy

            # 计算资源增量：若当前单元数比上一状态更多，则产生惩罚
            max_elements = getattr(self, 'max_elements', 50000)
            prev_num_elements = self._prev_num_elements if self._prev_num_elements is not None else current_num_elements
            delta_num_elements = max(0.0, float(current_num_elements - prev_num_elements))
            normalized_resource_delta = delta_num_elements / max(max_elements, 1)
            current_resource_usage = float(current_num_elements) / max(max_elements, 1)

            # 计算精度reward（带权重，使用log变换）
            log_component = np.log(abs(normalized_global_accuracy) + 1e-4) - np.log(1e-4)
            if normalized_global_accuracy < 0:
                log_component = -log_component
            accuracy_component = self.accuracy_weight * log_component

            # 资源组件：只要当前比上一状态使用更多网格，就给负值；否则为0（使用log变换）
            resource_log_component = np.log(normalized_resource_delta + 1e-4) - np.log(1e-4)
            resource_component = -self.resource_weight * resource_log_component
            resource_penalty = self.resource_weight * resource_log_component
            resource_reward_bonus = 0.0

            # 合成全局reward
            global_reward = accuracy_component + resource_component
            
            # 所有cell共享相同的全局reward
            cell_rewards = {cell_id: global_reward for cell_id in self.cell_to_elements_map.keys()}
            
            # 打印全局奖励信息（方便调参和监控）- 在更新history_allse之前打印
            print(f"\n[奖励组件分析 - 全局级别]")
            print(f"  Cell数量: {num_cells}")
            print(f"  当前总单元数: {current_num_elements}")
            print(f"  归一化因子:")
            print(f"    精度归一化因子 (全局): {normalization_factor_accuracy:.6f}")
            print(f"  权重配置: accuracy_weight={self.accuracy_weight:.3f}, resource_weight={self.resource_weight:.3f}")
            print(f"  原始奖励组件:")
            print(f"    全局精度奖励 (原始): {global_accuracy_reward:.6f}")
            print(f"  归一化奖励组件:")
            print(f"    归一化精度奖励: {normalized_global_accuracy:.6f}")
            prev_usage_pct = float(prev_num_elements) / max(max_elements, 1) * 100.0 if max_elements > 0 else 0.0
            print(f"  资源使用率: 当前={current_resource_usage*100:.2f}% | 上一步={prev_usage_pct:.2f}%")
            print(f"  资源增量: +{delta_num_elements:.0f} 元素 ({normalized_resource_delta*100:.2f}% of max)")
            print(f"  资源奖励组件: {resource_component:.6f} (penalty={resource_penalty:.6f})")
            print(f"  最终全局奖励: {global_reward:.6f}")
            print(f"  ALLSE变化: {self.history_allse:.6f} -> {current_allse:.6f}")
            if self.baseline_allse is not None:
                print(f"  与Baseline ALLSE差异: {abs(current_allse - self.baseline_allse):.6f}")
            
            # 保存当前状态用于下次计算（在打印之后更新，这样下次打印才能看到变化）
            self.history_allse = current_allse if current_allse is not None else 0
            
            # 设置全局reward作为返回值
            reward = global_reward
            self._prev_num_elements = current_num_elements
            
            # 定义特征字段的顺序（保持一致性）
            feature_fields = ['mises', 'strain_energy_density', 
                            's11', 's22', 's33', 's12', 's13', 's23',
                            'e11', 'e22', 'e33', 'e12', 'e13', 'e23']
            
            # 同时提取element_strain_energy（用于计算cell_strain_energy，仅用于诊断）
            element_strain_energy = {}
            for label_str, feat_dict in raw_element_features.items():
                label = int(label_str)
                feature_list = []
                for field in feature_fields:
                    feature_list.append(float(feat_dict.get(field, 0.0)))
                element_features[label] = feature_list
                # 提取strain_energy用于诊断信息
                element_strain_energy[label] = float(feat_dict.get('strain_energy', 0.0))
            
            # 对element_features中的所有维度特征进行归一化处理
            # 首先统计特征总数（每个cell的特征列表长度一致, 取第一个cell的特征数）
            if element_features:
                n_features = len(next(iter(element_features.values())))
                # 收集每个特征维度的所有值
                feature_matrix = np.array([v for v in element_features.values()])  # shape: [n_cells, n_features]
                mins = np.min(feature_matrix, axis=0)
                maxs = np.max(feature_matrix, axis=0)
                ranges = maxs - mins
                # 防止除0
                ranges[ranges == 0] = 1.0
                normalized_matrix = (feature_matrix - mins) / ranges
                # 更新element_features为归一化后的值
                for idx, label in enumerate(element_features.keys()):
                    element_features[int(label)] = list(normalized_matrix[idx])


            # 根据 self.cell_to_elements_map, 将 element_features 聚合为 cell_features
            cell_features = {}
            for cell_id, elem_labels in self.cell_to_elements_map.items():
                # 为该cell收集所有element的特征列表
                elem_feats_list = []
                for label in elem_labels:
                    feat = element_features.get(label)
                    if feat is not None:
                        elem_feats_list.append(feat)
                if not elem_feats_list:
                    continue  # 该cell无有效元素，跳过或可置为0
                # 聚合每个特征维度: 求 mean, std, max
                # 转置为各个特征的列表
                elem_feats_array = list(zip(*elem_feats_list))  # shape: [n_features, n_elem]
                agg_features = []
                for feat_vals in elem_feats_array:
                    vals = list(feat_vals)
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals))
                    max_val = float(np.max(vals))
                    agg_features.extend([mean_val, std_val, max_val])
                
                # 添加几何特征
                geometric_feat = self.cell_geometric_features.get(cell_id, [])
                if geometric_feat:
                    agg_features.extend(geometric_feat)
                
                # 添加资源消耗特征 - 让agent能看到每个cell的网格数量
                # 这对学习资源分配策略至关重要
                cell_element_count = len(elem_labels)
                total_elements = sum(len(elems) for elems in self.cell_to_elements_map.values())
                max_elements = getattr(self, 'max_elements', 50000)
                
                # 添加三个资源相关特征：
                # 1. 该cell的元素数量（归一化到0-1，使用max_elements作为归一化因子）
                # 2. 该cell占总元素的比例
                # 3. 当前总资源使用率
                agg_features.extend([
                    float(cell_element_count) / max_elements,  # cell网格数量（归一化）
                    float(cell_element_count) / max(total_elements, 1),  # cell占比
                    float(total_elements) / max_elements  # 全局资源使用率
                ])
                
                cell_features[cell_id] = agg_features

            # 计算每个cell的strain_energy总和（仅用于诊断信息）
            cell_strain_energy = {}
            for cell_id, elem_labels in self.cell_to_elements_map.items():
                total_se = sum(
                    element_strain_energy.get(label, 0.0) 
                    for label in elem_labels
                )
                cell_strain_energy[cell_id] = total_se
            
            # 将完整的状态特征存储在info中
            info = {
                "sim_dir": sim_dir, 
                "job_name": job_name,
                "cell_features": cell_features,
                "resource_usage": current_resource_usage,
                "allse": current_allse,  # 添加ALLSE值到info中
                "cell_strain_energy": cell_strain_energy,  # 添加每个cell的strain_energy（仅诊断用）
                "cell_rewards": cell_rewards,  # 所有cell共享相同的全局reward
                # 奖励诊断信息（全局奖励）
                "reward_components": {
                    "global_reward": global_reward,
                    "global_accuracy_reward": global_accuracy_reward,
                    "accuracy_reward": accuracy_component,
                    "raw_accuracy_reward": global_accuracy_reward,
                    "normalized_global_accuracy": normalized_global_accuracy,
                    "resource_component": resource_component,
                    "resource_penalty": resource_penalty,
                    "resource_reward_bonus": resource_reward_bonus,
                    "resource_usage": current_resource_usage,
                    "resource_delta_elements": delta_num_elements,
                    "resource_delta_ratio": normalized_resource_delta,
                    "prev_num_elements": prev_num_elements,
                    "resource_weight": self.resource_weight,
                    "num_elements": current_num_elements,
                    "baseline_allse": self.baseline_allse,
                    "current_allse": current_allse,
                    "history_allse": self.history_allse,
                    "accuracy_weight": self.accuracy_weight,
                    # 归一化因子
                    "normalization_factors": {
                        "accuracy": normalization_factor_accuracy,
                        "num_cells": num_cells,
                    },
                }
            }

        except Exception as e:
            print(f"[{self.run_id}] Could not parse state features from file")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return self.penalty_fea_failure, {"error": "parse_failed", "exception": str(e), "penalty_type": "fea_failure"}

        return reward, info