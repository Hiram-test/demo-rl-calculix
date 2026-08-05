from pathlib import Path  # 导入跨平台路径处理模块

runner_path = Path("experiments/cross_passage_torsion_benchmark/run_benchmark.py")  # 定义需要修正的实验主程序路径
source = runner_path.read_text(encoding="utf-8")  # 读取完整实验主程序文本
old_output = '''        lines.append("*NODE FILE,NSET=NALL")  # 请求将全部节点位移写入 ASCII FRD 文件
        lines.append("U")  # 指定位移输出变量
        lines.append("*NODE PRINT,NSET=RIGHT")  # 请求将右端节点外力写入 ASCII DAT 文件
'''
new_output = '''        lines.append("*NODE FILE,NSET=NALL")  # 请求将全部节点位移写入 ASCII FRD 文件
        lines.append("U")  # 指定位移输出变量
        lines.append("*NODE PRINT,NSET=NALL")  # 同时将全部原始梁节点位移写入稳健的 ASCII DAT 文件
        lines.append("U")  # 请求 DAT 位移表以兼容 B31 展开后空 FRD 位移数据集
        lines.append("*NODE PRINT,NSET=RIGHT")  # 请求将右端节点外力写入 ASCII DAT 文件
'''
if old_output not in source:  # 检查输出请求锚点是否仍与已审查版本一致
    raise RuntimeError("node-output anchor not found")  # 锚点变化时拒绝盲目修改
source = source.replace(old_output, new_output, 1)  # 增加原始梁节点 DAT 位移输出请求
old_fallback = '''        if not result:  # 检查是否成功读取位移结果
            raise RuntimeError(f"no displacement dataset found in {filepath}")  # 无位移时拒绝发布候选结果
        return result  # 返回节点编号到三维位移的映射
    def _parse_dat_reactions_and_energy(self, filepath: Path, element_count: int) -> tuple[dict[int, np.ndarray], np.ndarray]:  # 解析右端反力和逐梁单元内部能量
'''
new_fallback = '''        if not result:  # 检查 FRD 是否成功输出原始梁节点位移
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
'''
if old_fallback not in source:  # 检查 FRD 回退锚点是否仍与已审查版本一致
    raise RuntimeError("displacement-parser anchor not found")  # 锚点变化时拒绝盲目修改
source = source.replace(old_fallback, new_fallback, 1)  # 增加 DAT 位移解析回退并保留 FRD 优先路径
runner_path.write_text(source, encoding="utf-8")  # 写回修正后的完整实验主程序
