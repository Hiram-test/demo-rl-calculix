from pathlib import Path  # 导入跨平台路径处理模块

runner_path = Path("experiments/cross_passage_torsion_benchmark/run_benchmark.py")  # 定义需要修正的实验主程序路径
source = runner_path.read_text(encoding="utf-8")  # 读取已经应用位移兼容补丁的完整主程序
source = source.replace("T3D2", "B31")  # 统一修正遗留注释和报告中的单元类型名称
old_metric = '''        energy_error = abs(solution.strain_energy - self.reference.strain_energy) / max(abs(self.reference.strain_energy), 1.0e-18)  # 计算应变能相对误差
        probe_error = float(np.linalg.norm(solution.probe_vector - self.reference.probe_vector) / max(np.linalg.norm(self.reference.probe_vector), 1.0e-18))  # 计算中部位移场相对误差
        candidate_hotspots = set(int(index) for index in self._top_regions(solution.region_energy, TOP_HOTSPOT_COUNT))  # 提取候选结果最强热点区域
        hotspot_recall = len(candidate_hotspots.intersection(self.reference_hotspots)) / float(TOP_HOTSPOT_COUNT)  # 计算参考热点召回率
        resource_fraction = solution.element_count / float(self.element_cap)  # 计算最终网格资源占用比例
        objective = 0.35 * torque_error + 0.25 * energy_error + 0.20 * probe_error + 0.18 * (1.0 - hotspot_recall) + 0.02 * resource_fraction  # 形成统一综合目标函数
'''
new_metric = '''        solution_energy_distribution = solution.region_energy / max(float(np.sum(solution.region_energy)), 1.0e-18)  # 将候选区域能量转换为无量纲分布
        reference_energy_distribution = self.reference.region_energy / max(float(np.sum(self.reference.region_energy)), 1.0e-18)  # 将参考区域能量转换为无量纲分布
        energy_error = 0.5 * float(np.sum(np.abs(solution_energy_distribution - reference_energy_distribution)))  # 以总变差距离计算独立的区域能量分布误差
        probe_error = float(np.linalg.norm(solution.probe_vector - self.reference.probe_vector) / max(np.linalg.norm(self.reference.probe_vector), 1.0e-18))  # 计算中部位移场相对误差
        candidate_hotspots = set(int(index) for index in self._top_regions(solution.region_energy, TOP_HOTSPOT_COUNT))  # 提取候选结果最强热点区域
        hotspot_recall = len(candidate_hotspots.intersection(self.reference_hotspots)) / float(TOP_HOTSPOT_COUNT)  # 计算参考热点召回率
        resource_fraction = solution.element_count / float(self.element_cap)  # 计算最终网格资源占用比例
        objective = 0.40 * torque_error + 0.25 * energy_error + 0.20 * probe_error + 0.13 * (1.0 - hotspot_recall) + 0.02 * resource_fraction  # 使用互相独立的整体刚度、能量分布、位移场、热点和资源指标形成目标
'''
if old_metric not in source:  # 检查原始重复计权指标是否仍与已审查版本一致
    raise RuntimeError("metric anchor not found")  # 锚点变化时拒绝盲目修改
source = source.replace(old_metric, new_metric, 1)  # 将总应变能误差替换为独立的区域能量分布误差
source = source.replace("应变能误差↓", "区域能量分布误差↓")  # 修正 Markdown 表头以准确描述新指标
old_conclusion = '''f"均匀网格的扭矩误差为 **{100.0 * uniform.torque_error:.3f}%**，最优方法为 **{100.0 * best.torque_error:.3f}%**；均匀细化能够稳定整体刚度，但在固定单元预算下对局部扭转热点的资源利用率较低。", "热点 PSO 与纯 PSO 的差异直接反映候选区域降维是否有效；图多智能体反映局部状态和邻域消息能否在不建立全局价值函数时形成有效资源协商；DQN+GCN 的结果同时包含其图表示能力和仅有三十二次真实求解在线训练造成的样本效率限制。"'''
new_conclusion = '''f"均匀网格的扭矩误差为 **{100.0 * uniform.torque_error:.3f}%**，最优方法为 **{100.0 * best.torque_error:.3f}%**；均匀细化能够稳定整体位移场，但在固定单元预算下对主导扭转刚度的斜撑和连接区资源利用率较低。", "固定转角线弹性分析中总应变能与反力矩满足 U=Tθ/2，因此总能量误差没有作为独立目标重复计权，本实验改用十六个图区域归一化能量分布的总变差距离。", "热点 PSO 与纯 PSO 的差异直接反映候选区域降维是否有效；图多智能体反映局部状态和邻域消息能否在不建立全局价值函数时形成有效资源协商；DQN+GCN 的结果同时包含其图表示能力和仅有三十二次真实求解在线训练造成的样本效率限制。"'''
if old_conclusion not in source:  # 检查结论文本锚点是否仍与已审查版本一致
    raise RuntimeError("conclusion anchor not found")  # 锚点变化时拒绝盲目修改
source = source.replace(old_conclusion, new_conclusion, 1)  # 在最终报告中记录指标独立性修正和物理原因
runner_path.write_text(source, encoding="utf-8")  # 写回修正后的完整实验主程序
