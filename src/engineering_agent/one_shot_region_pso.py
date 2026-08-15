from __future__ import annotations  # 启用延迟类型注解以避免前向引用问题。
import math  # 提供对数、指数和有限性检查。
import random  # 提供带固定种子的确定性PSO随机系数。
from dataclasses import asdict, dataclass  # 提供不可变数据合同和结果序列化。
from typing import Any, Callable, Mapping, Sequence  # 提供公共接口所需的类型注解。
NumberMapping = Mapping[str, float]  # 定义区域尺寸映射的公共别名。
CertificationCallable = Callable[[NumberMapping], Mapping[str, Any]]  # 定义终局高保真认证器接口。
@dataclass(frozen=True)  # 将区域输入定义为不可变合同。
class RegionSpec:  # 描述一个区域智能体能够看到的全部先验与约束。
    region_id: str  # 保存区域唯一标识。
    reference_size: float  # 保存粗解或参考网格上的区域尺寸。
    delegated_size: float  # 保存全局智能体一次性委派的区域尺寸。
    minimum_size: float  # 保存区域允许的最小尺寸。
    maximum_size: float  # 保存区域允许的最大尺寸。
    reference_error: float  # 保存参考尺寸下的区域误差贡献。
    reference_resource: float  # 保存参考尺寸下的区域资源贡献。
    error_order: float  # 保存误差对尺寸的局部幂律阶次。
    resource_dimension: float  # 保存资源对尺寸的局部幂律维数。
    target_error: float  # 保存全局智能体分配给该区域的误差份额。
    target_resource: float  # 保存全局智能体分配给该区域的资源份额。
    confidence: float = 1.0  # 保存区域响应估计的置信度。
    neighbors: tuple[str, ...] = ()  # 保存一次通信所使用的相邻区域标识。
@dataclass(frozen=True)  # 将全局约束定义为不可变合同。
class GlobalTargets:  # 描述总智能体评估时使用的误差与资源目标。
    error_limit: float  # 保存全局允许的误差上限。
    resource_budget: float  # 保存全局允许的资源预算。
@dataclass(frozen=True)  # 将算法超参数定义为不可变合同。
class AlgorithmConfig:  # 集中管理一次区域修正与微型PSO的全部参数。
    local_error_weight: float = 4.0  # 设置区域误差残差的权重。
    local_resource_weight: float = 1.0  # 设置区域资源残差的权重。
    local_regularization: float = 0.5  # 设置区域一步修正的信赖正则项。
    global_pressure_share: float = 0.6  # 设置一次通信中全局压力进入各区域的比例。
    efficiency_coupling: float = 0.05  # 设置区域边际效率竞争的通信修正强度。
    neighbor_coupling: float = 0.08  # 设置相邻区域尺寸协调的通信修正强度。
    max_region_log_step: float = 0.30  # 限制每个区域智能体仅有的一次对数尺寸调整幅度。
    max_neighbor_ratio: float = 1.60  # 设置相邻区域允许的最大尺寸比。
    fitness_error_weight: float = 200.0  # 设置PSO误差超限罚项权重。
    fitness_resource_over_weight: float = 200.0  # 设置PSO资源超限罚项权重。
    fitness_resource_under_weight: float = 2.0  # 设置PSO资源闲置罚项权重。
    fitness_quality_weight: float = 50.0  # 设置PSO尺寸过渡违规罚项权重。
    fitness_deviation_weight: float = 0.05  # 设置PSO偏离母粒子的正则权重。
    pso_generations: int = 1  # 设置严格意义上的PSO更新代数且只允许一代或两代。
    pso_scale_radius: float = 0.04  # 设置母粒子周围整体尺度方向的初始扰动半径。
    pso_transfer_radius: float = 0.04  # 设置母粒子周围资源转移方向的初始扰动半径。
    pso_position_bound: float = 0.12  # 设置两个低维校准坐标的绝对边界。
    pso_velocity_bound: float = 0.04  # 设置两个低维校准坐标的单代速度边界。
    pso_inertia: float = 0.35  # 设置微型PSO惯性系数。
    pso_cognitive: float = 0.80  # 设置微型PSO认知系数。
    pso_social: float = 1.20  # 设置微型PSO社会系数。
    random_seed: int = 17  # 设置可重复的PSO随机种子。
@dataclass(frozen=True)  # 将一次区域估计定义为不可变记录。
class LocalEstimate:  # 保存区域智能体在委派尺寸上唯一一次误差资源估计。
    region_id: str  # 保存区域标识。
    size: float  # 保存本次估计使用的委派尺寸。
    error: float  # 保存估计的区域误差贡献。
    resource: float  # 保存估计的区域资源贡献。
    error_log_sensitivity: float  # 保存误差对对数尺寸的局部导数。
    resource_log_sensitivity: float  # 保存资源对对数尺寸的局部导数绝对值。
    marginal_efficiency: float  # 保存单位资源变化对应的误差收益尺度。
@dataclass(frozen=True)  # 将一次区域通信消息定义为不可变记录。
class RegionMessage:  # 保存每个区域智能体在唯一通信轮次中广播的内容。
    region_id: str  # 保存发送消息的区域标识。
    estimate: LocalEstimate  # 保存该区域唯一一次局部估计。
    error_log_residual: float  # 保存区域误差相对目标的对数残差。
    resource_log_residual: float  # 保存区域资源相对目标的对数残差。
    raw_log_step: float  # 保存通信前由一维线性化得到的原始尺寸修正。
    candidate_log_size: float  # 保存通信前候选尺寸的对数值。
    confidence: float  # 保存消息置信度。
@dataclass(frozen=True)  # 将唯一通信轮次的公共摘要定义为不可变记录。
class CommunicationSummary:  # 保存区域智能体通信后共同获得的全局压力信息。
    total_estimated_error: float  # 保存委派粒子的总估计误差。
    total_estimated_resource: float  # 保存委派粒子的总估计资源。
    global_error_log_residual: float  # 保存总误差相对目标的对数残差。
    global_resource_log_residual: float  # 保存总资源相对预算的对数残差。
    global_log_step: float  # 保存一次通信得到的共同对数尺寸压力。
    mean_log_efficiency: float  # 保存区域边际效率的置信加权对数均值。
@dataclass(frozen=True)  # 将区域一次调整结果定义为不可变记录。
class RegionDecision:  # 保存每个区域智能体通信后唯一一次尺寸调整。
    region_id: str  # 保存区域标识。
    delegated_size: float  # 保存全局智能体委派的原尺寸。
    adjusted_size: float  # 保存区域智能体调整一次后的尺寸。
    raw_log_step: float  # 保存区域本地残差产生的修正量。
    global_log_shift: float  # 保存全局压力产生的修正量。
    efficiency_log_shift: float  # 保存边际效率通信产生的修正量。
    neighbor_log_shift: float  # 保存相邻尺寸通信产生的修正量。
    applied_log_step: float  # 保存投影与限幅后实际采用的总修正量。
@dataclass(frozen=True)  # 将总智能体的一次候选评估定义为不可变记录。
class GlobalEvaluation:  # 保存完整区域尺寸粒子的误差资源质量与适应度。
    sizes: dict[str, float]  # 保存完整区域尺寸向量。
    predicted_error: float  # 保存代理预测的全局误差。
    predicted_resource: float  # 保存代理预测的全局资源。
    quality_penalty: float  # 保存相邻尺寸过渡违规量。
    error_log_ratio: float  # 保存总误差相对目标的对数比。
    resource_log_ratio: float  # 保存总资源相对预算的对数比。
    fitness: float  # 保存PSO使用的归一化约束残差适应度。
    error_feasible: bool  # 标记误差约束是否满足。
    resource_feasible: bool  # 标记资源约束是否满足。
    quality_feasible: bool  # 标记尺寸过渡约束是否满足。
@dataclass(frozen=True)  # 将一个微型PSO粒子定义为不可变记录。
class CalibrationParticle:  # 保存母粒子在两个残余校准坐标上的一个位置。
    scale_coordinate: float  # 保存整体粗细尺度校准量。
    transfer_coordinate: float  # 保存区域间资源转移校准量。
    evaluation: GlobalEvaluation  # 保存该低维位置解码后的完整尺寸粒子评估。
@dataclass(frozen=True)  # 将严格状态机调用计数定义为不可变记录。
class RunTrace:  # 证明算法没有偷偷增加智能体轮次或高保真求解。
    global_delegations: int  # 记录全局智能体委派次数。
    regional_estimates: int  # 记录全部区域智能体局部估计次数。
    communication_rounds: int  # 记录区域消息通信轮次。
    regional_adjustments: int  # 记录全部区域智能体尺寸调整次数。
    global_base_reviews: int  # 记录总智能体在PSO前评估母粒子的次数。
    pso_coordinate_dimension: int  # 记录PSO实际开放的残余自由度数量。
    pso_generations: int  # 记录PSO严格更新代数。
    pso_surrogate_evaluations: int  # 记录PSO额外调用代理模型的次数。
    total_surrogate_evaluations: int  # 记录母粒子评估与PSO评估总次数。
    high_fidelity_certifications: int  # 记录终局高保真认证调用次数。
@dataclass(frozen=True)  # 将完整运行结果定义为不可变记录。
class RunResult:  # 汇总全局委派、区域决策、母粒子与PSO终局粒子。
    delegated_sizes: dict[str, float]  # 保存全局智能体一次委派的尺寸向量。
    regional_messages: tuple[RegionMessage, ...]  # 保存唯一通信轮次中的全部消息。
    communication: CommunicationSummary  # 保存唯一通信轮次的全局摘要。
    regional_decisions: tuple[RegionDecision, ...]  # 保存各区域唯一一次调整结果。
    base_particle: CalibrationParticle  # 保存总智能体评估后的唯一母粒子。
    transfer_direction: dict[str, float]  # 保存由区域消息编译得到的资源转移方向。
    selected_particle: CalibrationParticle  # 保存一到两代微型PSO选中的终局粒子。
    pso_history: tuple[dict[str, Any], ...]  # 保存每次代理评估的可审计轨迹。
    certification: Mapping[str, Any] | None  # 保存可选的一次高保真终局认证结果。
    trace: RunTrace  # 保存严格状态机调用计数。
    def to_dict(self) -> dict[str, Any]:  # 将结果转换为可直接写入JSON的字典。
        return asdict(self)  # 使用数据类递归序列化所有嵌套记录。
def _positive(value: float, name: str) -> None:  # 验证一个量必须严格为正。
    if not math.isfinite(value) or value <= 0.0:  # 拒绝非有限值、零值和负值。
        raise ValueError(f"{name} must be finite and positive")  # 报告具体字段错误。
def _clip(value: float, lower: float, upper: float) -> float:  # 将标量投影到闭区间。
    return min(max(value, lower), upper)  # 返回完成上下界投影的值。
def _safe_log_ratio(numerator: float, denominator: float) -> float:  # 计算经过正值验证的对数比。
    _positive(numerator, "log numerator")  # 验证分子能够取对数。
    _positive(denominator, "log denominator")  # 验证分母能够取对数。
    return math.log(numerator / denominator)  # 返回无量纲对数残差。
def validate_problem(regions: Sequence[RegionSpec], targets: GlobalTargets, config: AlgorithmConfig) -> None:  # 验证全部输入合同。
    if not regions:  # 拒绝没有区域智能体的空问题。
        raise ValueError("at least one region is required")  # 报告区域列表为空。
    _positive(targets.error_limit, "error_limit")  # 验证全局误差目标。
    _positive(targets.resource_budget, "resource_budget")  # 验证全局资源预算。
    if config.pso_generations not in (1, 2):  # 强制PSO只能校准一代或两代。
        raise ValueError("pso_generations must be exactly 1 or 2")  # 阻止长程PSO重新接管决策。
    _positive(config.local_error_weight, "local_error_weight")  # 验证区域误差权重。
    _positive(config.local_resource_weight, "local_resource_weight")  # 验证区域资源权重。
    _positive(config.local_regularization, "local_regularization")  # 验证区域信赖正则项。
    _positive(config.max_region_log_step, "max_region_log_step")  # 验证区域一步调整上限。
    _positive(config.max_neighbor_ratio, "max_neighbor_ratio")  # 验证相邻尺寸比上限。
    _positive(config.pso_position_bound, "pso_position_bound")  # 验证PSO位置边界。
    _positive(config.pso_velocity_bound, "pso_velocity_bound")  # 验证PSO速度边界。
    ids = [region.region_id for region in regions]  # 收集全部区域标识。
    if len(ids) != len(set(ids)):  # 检查区域标识是否重复。
        raise ValueError("region_id values must be unique")  # 阻止通信映射发生覆盖。
    known_ids = set(ids)  # 构造邻接关系验证集合。
    for region in regions:  # 逐个验证区域合同。
        if not region.region_id:  # 拒绝空区域标识。
            raise ValueError("region_id must not be empty")  # 报告区域标识错误。
        _positive(region.reference_size, f"{region.region_id}.reference_size")  # 验证参考尺寸。
        _positive(region.delegated_size, f"{region.region_id}.delegated_size")  # 验证委派尺寸。
        _positive(region.minimum_size, f"{region.region_id}.minimum_size")  # 验证尺寸下界。
        _positive(region.maximum_size, f"{region.region_id}.maximum_size")  # 验证尺寸上界。
        _positive(region.reference_error, f"{region.region_id}.reference_error")  # 验证参考误差。
        _positive(region.reference_resource, f"{region.region_id}.reference_resource")  # 验证参考资源。
        _positive(region.error_order, f"{region.region_id}.error_order")  # 验证误差阶次。
        _positive(region.resource_dimension, f"{region.region_id}.resource_dimension")  # 验证资源维数。
        _positive(region.target_error, f"{region.region_id}.target_error")  # 验证区域误差份额。
        _positive(region.target_resource, f"{region.region_id}.target_resource")  # 验证区域资源份额。
        _positive(region.confidence, f"{region.region_id}.confidence")  # 验证区域置信度。
        if region.minimum_size > region.maximum_size:  # 检查尺寸上下界顺序。
            raise ValueError(f"{region.region_id} has inverted size bounds")  # 报告非法尺寸区间。
        if region.delegated_size < region.minimum_size or region.delegated_size > region.maximum_size:  # 检查全局委派是否可行。
            raise ValueError(f"{region.region_id}.delegated_size is outside bounds")  # 阻止不可行母提案进入区域智能体。
        for neighbor in region.neighbors:  # 逐个检查邻接标识。
            if neighbor not in known_ids:  # 检查邻居是否存在。
                raise ValueError(f"unknown neighbor {neighbor} for {region.region_id}")  # 报告悬空邻接边。
            if neighbor == region.region_id:  # 拒绝区域自环。
                raise ValueError(f"self-neighbor is not allowed for {region.region_id}")  # 报告无效通信边。
class PowerLawSurrogate:  # 用粗解锚点和局部阶次提供零高保真调用的响应模型。
    def __init__(self, regions: Sequence[RegionSpec], background_error: float = 0.0, background_resource: float = 0.0) -> None:  # 初始化代理模型。
        if not math.isfinite(background_error) or background_error < 0.0:  # 验证不可调背景误差。
            raise ValueError("background_error must be finite and nonnegative")  # 报告背景误差错误。
        if not math.isfinite(background_resource) or background_resource < 0.0:  # 验证不可调背景资源。
            raise ValueError("background_resource must be finite and nonnegative")  # 报告背景资源错误。
        self._regions = {region.region_id: region for region in regions}  # 建立区域响应参数索引。
        self.background_error = float(background_error)  # 保存不可调背景误差。
        self.background_resource = float(background_resource)  # 保存不可调背景资源。
    def predict_region(self, region_id: str, size: float) -> LocalEstimate:  # 在任意候选尺寸上进行纯代数预测。
        region = self._regions[region_id]  # 读取目标区域的响应参数。
        _positive(size, f"{region_id}.candidate_size")  # 验证候选尺寸。
        error = region.reference_error * (size / region.reference_size) ** region.error_order  # 计算局部误差幂律响应。
        resource = region.reference_resource * (region.reference_size / size) ** region.resource_dimension  # 计算局部资源幂律响应。
        marginal = region.error_order * error / (region.resource_dimension * resource)  # 计算正的边际误差资源效率尺度。
        return LocalEstimate(region_id, size, error, resource, region.error_order, region.resource_dimension, marginal)  # 返回完整局部预测。
    def aggregate(self, sizes: NumberMapping) -> tuple[float, float, dict[str, LocalEstimate]]:  # 汇总完整区域尺寸粒子的响应。
        estimates = {region_id: self.predict_region(region_id, float(size)) for region_id, size in sizes.items()}  # 对每个区域执行代数预测。
        total_error = self.background_error + sum(estimate.error for estimate in estimates.values())  # 汇总全局预测误差。
        total_resource = self.background_resource + sum(estimate.resource for estimate in estimates.values())  # 汇总全局预测资源。
        return total_error, total_resource, estimates  # 返回总量与各区域明细。
class GlobalAgent:  # 实现一次委派和PSO前一次母粒子评估的总智能体。
    def __init__(self, regions: Sequence[RegionSpec], targets: GlobalTargets, config: AlgorithmConfig, surrogate: PowerLawSurrogate) -> None:  # 初始化总智能体。
        self.regions = tuple(regions)  # 保存稳定的区域顺序。
        self.targets = targets  # 保存全局误差资源目标。
        self.config = config  # 保存算法配置。
        self.surrogate = surrogate  # 保存零高保真调用的响应代理。
        self.delegation_calls = 0  # 初始化委派调用计数。
        self.evaluation_calls = 0  # 初始化完整粒子代理评估计数。
    def delegate_once(self) -> dict[str, float]:  # 发出全局智能体唯一一次区域尺寸委派。
        if self.delegation_calls != 0:  # 阻止总智能体二次改写区域尺寸。
            raise RuntimeError("global delegation may be called only once")  # 报告状态机违规。
        self.delegation_calls += 1  # 记录唯一一次委派。
        return {region.region_id: region.delegated_size for region in self.regions}  # 返回完整委派尺寸向量。
    def _quality_penalty(self, sizes: NumberMapping) -> float:  # 计算相邻区域尺寸过渡违规量。
        region_map = {region.region_id: region for region in self.regions}  # 建立区域索引。
        visited: set[tuple[str, str]] = set()  # 防止无向邻接边重复计数。
        penalty = 0.0  # 初始化质量罚项。
        for region in self.regions:  # 遍历每个区域的邻接边。
            for neighbor in region.neighbors:  # 遍历当前区域的每个邻居。
                edge = tuple(sorted((region.region_id, neighbor)))  # 将邻接边规范化为无向键。
                if edge in visited:  # 检查该边是否已经计入。
                    continue  # 跳过重复方向。
                visited.add(edge)  # 标记该边已经处理。
                left = float(sizes[region.region_id])  # 读取当前区域尺寸。
                right = float(sizes[neighbor])  # 读取相邻区域尺寸。
                ratio = max(left, right) / min(left, right)  # 计算无量纲尺寸比。
                violation = max(math.log(ratio / self.config.max_neighbor_ratio), 0.0)  # 计算超过允许比值的对数违规量。
                penalty += violation * violation  # 累加平方质量罚项。
        if len(region_map) != len(self.regions):  # 保留区域索引一致性检查。
            raise RuntimeError("region index corruption detected")  # 报告内部状态损坏。
        return penalty  # 返回总质量罚项。
    def evaluate(self, sizes: NumberMapping, scale_coordinate: float = 0.0, transfer_coordinate: float = 0.0) -> GlobalEvaluation:  # 评估一个完整尺寸粒子。
        self.evaluation_calls += 1  # 记录一次代理评估。
        total_error, total_resource, _ = self.surrogate.aggregate(sizes)  # 汇总误差与资源预测。
        error_log_ratio = _safe_log_ratio(total_error, self.targets.error_limit)  # 计算误差约束对数残差。
        resource_log_ratio = _safe_log_ratio(total_resource, self.targets.resource_budget)  # 计算资源约束对数残差。
        quality_penalty = self._quality_penalty(sizes)  # 计算尺寸过渡违规。
        error_excess = max(error_log_ratio, 0.0)  # 仅提取误差超限部分。
        resource_excess = max(resource_log_ratio, 0.0)  # 仅提取资源超限部分。
        resource_underuse = max(-resource_log_ratio, 0.0)  # 提取预算闲置部分。
        deviation = scale_coordinate * scale_coordinate + transfer_coordinate * transfer_coordinate  # 计算偏离母粒子的低维正则量。
        fitness = self.config.fitness_error_weight * error_excess * error_excess  # 加入误差超限罚项。
        fitness += self.config.fitness_resource_over_weight * resource_excess * resource_excess  # 加入资源超限罚项。
        fitness += self.config.fitness_resource_under_weight * resource_underuse * resource_underuse  # 加入资源闲置罚项。
        fitness += self.config.fitness_quality_weight * quality_penalty  # 加入尺寸过渡质量罚项。
        fitness += self.config.fitness_deviation_weight * deviation  # 加入偏离母粒子的正则项。
        copied_sizes = {region.region_id: float(sizes[region.region_id]) for region in self.regions}  # 按稳定顺序复制完整尺寸向量。
        return GlobalEvaluation(copied_sizes, total_error, total_resource, quality_penalty, error_log_ratio, resource_log_ratio, fitness, error_log_ratio <= 0.0, resource_log_ratio <= 0.0, quality_penalty <= 1.0e-14)  # 返回完整总智能体评估。
class RegionalAgent:  # 实现每个区域一次估计、一次发消息和一次调整。
    def __init__(self, region: RegionSpec, config: AlgorithmConfig, surrogate: PowerLawSurrogate) -> None:  # 初始化区域智能体。
        self.region = region  # 保存该智能体负责的唯一区域。
        self.config = config  # 保存算法配置。
        self.surrogate = surrogate  # 保存区域响应代理。
        self.estimate_calls = 0  # 初始化局部估计调用计数。
        self.message_calls = 0  # 初始化消息构造调用计数。
        self.adjustment_calls = 0  # 初始化区域尺寸调整调用计数。
        self._estimate: LocalEstimate | None = None  # 预留唯一一次估计结果。
        self._message: RegionMessage | None = None  # 预留唯一一次通信消息。
    def estimate_once(self, delegated_size: float) -> LocalEstimate:  # 在全局委派尺寸上执行唯一一次局部估计。
        if self.estimate_calls != 0:  # 阻止区域智能体二次探测响应。
            raise RuntimeError(f"{self.region.region_id} may estimate only once")  # 报告状态机违规。
        self.estimate_calls += 1  # 记录唯一一次局部估计。
        self._estimate = self.surrogate.predict_region(self.region.region_id, delegated_size)  # 计算委派尺寸上的误差资源响应。
        return self._estimate  # 返回局部估计。
    def build_message_once(self) -> RegionMessage:  # 根据唯一估计构造唯一通信消息。
        if self.message_calls != 0:  # 阻止区域智能体重复广播。
            raise RuntimeError(f"{self.region.region_id} may message only once")  # 报告状态机违规。
        if self._estimate is None:  # 检查消息是否建立在真实局部估计上。
            raise RuntimeError("estimate_once must precede build_message_once")  # 报告非法调用顺序。
        self.message_calls += 1  # 记录唯一一次消息构造。
        error_residual = _safe_log_ratio(self._estimate.error, self.region.target_error)  # 计算区域误差残差。
        resource_residual = _safe_log_ratio(self._estimate.resource, self.region.target_resource)  # 计算区域资源残差。
        p_value = self._estimate.error_log_sensitivity  # 读取误差对数灵敏度。
        d_value = self._estimate.resource_log_sensitivity  # 读取资源对数灵敏度绝对值。
        numerator = self.config.local_resource_weight * d_value * resource_residual  # 构造资源推动区域变粗的线性项。
        numerator -= self.config.local_error_weight * p_value * error_residual  # 加入误差推动区域变细的线性项。
        denominator = self.config.local_error_weight * p_value * p_value  # 构造误差曲率项。
        denominator += self.config.local_resource_weight * d_value * d_value  # 加入资源曲率项。
        denominator += self.config.local_regularization  # 加入一步信赖正则项。
        raw_step = _clip(numerator / denominator, -self.config.max_region_log_step, self.config.max_region_log_step)  # 求解并限幅一维局部校正。
        candidate_log_size = math.log(self._estimate.size) + raw_step  # 形成通信前候选尺寸。
        self._message = RegionMessage(self.region.region_id, self._estimate, error_residual, resource_residual, raw_step, candidate_log_size, self.region.confidence)  # 保存唯一通信消息。
        return self._message  # 返回通信消息。
    def adjust_once(self, communication: CommunicationSummary, messages: Mapping[str, RegionMessage]) -> RegionDecision:  # 通信后仅调整一次自身尺寸。
        if self.adjustment_calls != 0:  # 阻止区域智能体二次修正尺寸。
            raise RuntimeError(f"{self.region.region_id} may adjust only once")  # 报告状态机违规。
        if self._message is None:  # 检查调整是否建立在唯一通信消息上。
            raise RuntimeError("build_message_once must precede adjust_once")  # 报告非法调用顺序。
        self.adjustment_calls += 1  # 记录唯一一次区域尺寸调整。
        global_shift = self.config.global_pressure_share * communication.global_log_step  # 计算全局误差资源压力分量。
        efficiency_shift = -self.config.efficiency_coupling * (math.log(self._message.estimate.marginal_efficiency) - communication.mean_log_efficiency)  # 计算高收益区域变细低收益区域变粗的通信分量。
        neighbor_messages = [messages[neighbor] for neighbor in self.region.neighbors]  # 收集相邻区域在唯一通信轮次中的消息。
        if neighbor_messages:  # 判断当前区域是否存在邻居。
            total_weight = sum(message.confidence for message in neighbor_messages)  # 计算邻居置信权重和。
            neighbor_mean = sum(message.confidence * message.candidate_log_size for message in neighbor_messages) / total_weight  # 计算邻居候选对数尺寸均值。
            neighbor_shift = self.config.neighbor_coupling * (neighbor_mean - self._message.candidate_log_size)  # 计算一次尺寸协调分量。
        else:  # 处理孤立区域。
            neighbor_shift = 0.0  # 孤立区域不施加邻接修正。
        combined_step = self._message.raw_log_step + global_shift + efficiency_shift + neighbor_shift  # 合成区域唯一一次调整。
        applied_step = _clip(combined_step, -self.config.max_region_log_step, self.config.max_region_log_step)  # 对总调整执行信赖域限幅。
        adjusted_size = self._message.estimate.size * math.exp(applied_step)  # 将对数修正映射回物理尺寸。
        adjusted_size = _clip(adjusted_size, self.region.minimum_size, self.region.maximum_size)  # 投影到区域尺寸硬边界。
        actual_step = math.log(adjusted_size / self._message.estimate.size)  # 记录投影后真实采用的修正量。
        return RegionDecision(self.region.region_id, self._message.estimate.size, adjusted_size, self._message.raw_log_step, global_shift, efficiency_shift, neighbor_shift, actual_step)  # 返回区域唯一决策。
class CommunicationHub:  # 实现全部区域消息的一次同步通信。
    def __init__(self, regions: Sequence[RegionSpec], targets: GlobalTargets, config: AlgorithmConfig, surrogate: PowerLawSurrogate) -> None:  # 初始化通信枢纽。
        self.regions = tuple(regions)  # 保存稳定区域顺序。
        self.targets = targets  # 保存全局约束。
        self.config = config  # 保存算法配置。
        self.surrogate = surrogate  # 保存背景误差资源常数。
        self.exchange_calls = 0  # 初始化通信轮次计数。
    def exchange_once(self, messages: Sequence[RegionMessage]) -> CommunicationSummary:  # 汇总并广播唯一一轮消息。
        if self.exchange_calls != 0:  # 阻止第二轮区域通信。
            raise RuntimeError("communication may occur only once")  # 报告状态机违规。
        if len(messages) != len(self.regions):  # 检查每个区域是否恰好发送一条消息。
            raise ValueError("exactly one message per region is required")  # 报告消息数量错误。
        self.exchange_calls += 1  # 记录唯一通信轮次。
        total_error = self.surrogate.background_error + sum(message.estimate.error for message in messages)  # 汇总委派粒子的总估计误差。
        total_resource = self.surrogate.background_resource + sum(message.estimate.resource for message in messages)  # 汇总委派粒子的总估计资源。
        global_error_residual = _safe_log_ratio(total_error, self.targets.error_limit)  # 计算全局误差残差。
        global_resource_residual = _safe_log_ratio(total_resource, self.targets.resource_budget)  # 计算全局资源残差。
        adjustable_error = sum(message.estimate.error for message in messages)  # 计算可调区域误差总量。
        adjustable_resource = sum(message.estimate.resource for message in messages)  # 计算可调区域资源总量。
        p_bar = sum(message.estimate.error * message.estimate.error_log_sensitivity for message in messages) / adjustable_error  # 计算误差加权平均阶次。
        d_bar = sum(message.estimate.resource * message.estimate.resource_log_sensitivity for message in messages) / adjustable_resource  # 计算资源加权平均维数。
        numerator = self.config.local_resource_weight * d_bar * global_resource_residual  # 构造全局资源压力线性项。
        numerator -= self.config.local_error_weight * p_bar * global_error_residual  # 加入全局误差压力线性项。
        denominator = self.config.local_error_weight * p_bar * p_bar  # 构造全局误差曲率项。
        denominator += self.config.local_resource_weight * d_bar * d_bar  # 加入全局资源曲率项。
        denominator += self.config.local_regularization  # 加入全局一步信赖正则项。
        global_step = _clip(numerator / denominator, -self.config.max_region_log_step, self.config.max_region_log_step)  # 求解并限幅共同压力修正。
        efficiency_weights = [message.confidence * message.estimate.resource_log_sensitivity * message.estimate.resource for message in messages]  # 构造边际效率通信权重。
        weight_sum = sum(efficiency_weights)  # 汇总边际效率权重。
        mean_log_efficiency = sum(weight * math.log(message.estimate.marginal_efficiency) for weight, message in zip(efficiency_weights, messages)) / weight_sum  # 计算置信资源加权效率均值。
        return CommunicationSummary(total_error, total_resource, global_error_residual, global_resource_residual, global_step, mean_log_efficiency)  # 返回唯一通信摘要。
class MicroPSO:  # 实现只围绕一个母粒子的二维微型PSO校准。
    def __init__(self, regions: Sequence[RegionSpec], config: AlgorithmConfig, global_agent: GlobalAgent, surrogate: PowerLawSurrogate) -> None:  # 初始化微型PSO。
        self.regions = tuple(regions)  # 保存稳定的区域顺序。
        self.config = config  # 保存PSO配置。
        self.global_agent = global_agent  # 保存完整粒子评估器。
        self.surrogate = surrogate  # 保存响应代理以构造资源中性方向。
        self.extra_evaluations = 0  # 初始化除母粒子缓存外的PSO评估计数。
    def build_transfer_direction(self, base_sizes: NumberMapping, messages: Mapping[str, RegionMessage]) -> dict[str, float]:  # 由区域通信证据编译唯一资源转移方向。
        resource_weights: list[float] = []  # 初始化一阶资源灵敏度权重。
        raw_values: list[float] = []  # 初始化未中心化的边际效率方向。
        for region in self.regions:  # 按稳定顺序遍历区域。
            estimate = self.surrogate.predict_region(region.region_id, float(base_sizes[region.region_id]))  # 计算母粒子处的代数资源灵敏度。
            weight = estimate.resource * estimate.resource_log_sensitivity * region.confidence  # 构造资源一阶中性条件权重。
            resource_weights.append(weight)  # 保存当前区域资源权重。
            raw_values.append(-math.log(messages[region.region_id].estimate.marginal_efficiency))  # 让高边际效率区域在正转移坐标下变细。
        weighted_mean = sum(weight * value for weight, value in zip(resource_weights, raw_values)) / sum(resource_weights)  # 计算资源权重下的方向均值。
        centered = [value - weighted_mean for value in raw_values]  # 消除整体尺度分量以保持一阶资源中性。
        maximum = max(abs(value) for value in centered)  # 计算方向归一化尺度。
        if maximum <= 1.0e-14:  # 检查所有区域边际效率是否近似一致。
            normalized = [0.0 for _ in centered]  # 退化时关闭资源转移坐标。
        else:  # 处理非退化边际效率差异。
            normalized = [value / maximum for value in centered]  # 将方向归一化到最大绝对值为一。
        return {region.region_id: value for region, value in zip(self.regions, normalized)}  # 返回按区域索引的资源转移方向。
    def _decode(self, base_sizes: NumberMapping, transfer_direction: NumberMapping, scale_coordinate: float, transfer_coordinate: float) -> dict[str, float]:  # 将二维坐标解码为完整区域尺寸粒子。
        decoded: dict[str, float] = {}  # 初始化完整尺寸向量。
        for region in self.regions:  # 按稳定顺序解码每个区域。
            log_shift = scale_coordinate + transfer_coordinate * float(transfer_direction[region.region_id])  # 合成整体尺度和资源转移修正。
            candidate = float(base_sizes[region.region_id]) * math.exp(log_shift)  # 将低维修正映射到物理尺寸。
            decoded[region.region_id] = _clip(candidate, region.minimum_size, region.maximum_size)  # 投影到区域尺寸硬边界。
        return decoded  # 返回完整候选尺寸向量。
    def _evaluate(self, base_sizes: NumberMapping, transfer_direction: NumberMapping, position: Sequence[float], history: list[dict[str, Any]], stage: str, generation: int, cached_base: GlobalEvaluation | None = None) -> CalibrationParticle:  # 评估一个二维PSO位置。
        scale_coordinate = float(position[0])  # 读取整体尺度坐标。
        transfer_coordinate = float(position[1])  # 读取资源转移坐标。
        if cached_base is not None and abs(scale_coordinate) <= 1.0e-15 and abs(transfer_coordinate) <= 1.0e-15:  # 检查是否可以复用总智能体母粒子评估。
            evaluation = cached_base  # 复用已评估的唯一母粒子。
            cached = True  # 标记本次没有新增代理调用。
        else:  # 处理真正需要代理评估的扰动粒子。
            sizes = self._decode(base_sizes, transfer_direction, scale_coordinate, transfer_coordinate)  # 解码完整尺寸向量。
            evaluation = self.global_agent.evaluate(sizes, scale_coordinate, transfer_coordinate)  # 调用零高保真代理评估完整粒子。
            self.extra_evaluations += 1  # 记录PSO新增代理调用。
            cached = False  # 标记本次执行了代理调用。
        history.append({"stage": stage, "generation": generation, "scale": scale_coordinate, "transfer": transfer_coordinate, "fitness": evaluation.fitness, "predicted_error": evaluation.predicted_error, "predicted_resource": evaluation.predicted_resource, "cached_base": cached})  # 写入可审计PSO轨迹。
        return CalibrationParticle(scale_coordinate, transfer_coordinate, evaluation)  # 返回粒子及评估。
    def calibrate(self, base_sizes: NumberMapping, transfer_direction: NumberMapping, base_evaluation: GlobalEvaluation) -> tuple[CalibrationParticle, tuple[dict[str, Any], ...]]:  # 围绕母粒子执行一代或两代二维PSO。
        rng = random.Random(self.config.random_seed)  # 创建确定性随机数发生器。
        positions = [[0.0, 0.0], [self.config.pso_scale_radius, 0.0], [-self.config.pso_scale_radius, 0.0], [0.0, self.config.pso_transfer_radius], [0.0, -self.config.pso_transfer_radius]]  # 由一个母粒子生成五个确定性局部粒子。
        velocities = [[0.0, 0.0] for _ in positions]  # 将所有初始速度设为零。
        history: list[dict[str, Any]] = []  # 初始化PSO评估轨迹。
        current = [self._evaluate(base_sizes, transfer_direction, position, history, "initial", 0, base_evaluation) for position in positions]  # 评估母粒子及四个轴向扰动。
        personal_positions = [position.copy() for position in positions]  # 初始化每个粒子的个体最优位置。
        personal_best = list(current)  # 初始化每个粒子的个体最优评估。
        global_index = min(range(len(personal_best)), key=lambda index: personal_best[index].evaluation.fitness)  # 找到初始全局最优粒子。
        global_position = personal_positions[global_index].copy()  # 保存初始全局最优位置。
        for generation in range(1, self.config.pso_generations + 1):  # 严格执行配置指定的一代或两代更新。
            for particle_index, position in enumerate(positions):  # 遍历五个局部粒子。
                for coordinate_index in range(2):  # 只更新整体尺度与资源转移两个坐标。
                    cognitive_random = rng.random()  # 生成认知项随机系数。
                    social_random = rng.random()  # 生成社会项随机系数。
                    velocity = self.config.pso_inertia * velocities[particle_index][coordinate_index]  # 计算惯性速度分量。
                    velocity += self.config.pso_cognitive * cognitive_random * (personal_positions[particle_index][coordinate_index] - position[coordinate_index])  # 加入个体最优吸引项。
                    velocity += self.config.pso_social * social_random * (global_position[coordinate_index] - position[coordinate_index])  # 加入全局最优吸引项。
                    velocity = _clip(velocity, -self.config.pso_velocity_bound, self.config.pso_velocity_bound)  # 限制单代校准速度。
                    velocities[particle_index][coordinate_index] = velocity  # 保存更新后的速度。
                    position[coordinate_index] = _clip(position[coordinate_index] + velocity, -self.config.pso_position_bound, self.config.pso_position_bound)  # 更新并投影低维位置。
            current = [self._evaluate(base_sizes, transfer_direction, position, history, "update", generation) for position in positions]  # 重新评价更新后的全部局部粒子。
            for particle_index, particle in enumerate(current):  # 更新每个粒子的个体最优。
                if particle.evaluation.fitness < personal_best[particle_index].evaluation.fitness:  # 比较新旧个体最优适应度。
                    personal_best[particle_index] = particle  # 保存更优评估。
                    personal_positions[particle_index] = positions[particle_index].copy()  # 保存更优位置。
            global_index = min(range(len(personal_best)), key=lambda index: personal_best[index].evaluation.fitness)  # 重新寻找全局最优粒子。
            global_position = personal_positions[global_index].copy()  # 保存当前全局最优位置。
        selected_index = min(range(len(personal_best)), key=lambda index: personal_best[index].evaluation.fitness)  # 确定终局选中的已评价粒子。
        return personal_best[selected_index], tuple(history)  # 返回终局粒子和完整评估轨迹。
class OneShotDelegatedRegionPSO:  # 编排用户指定的完整单次多智能体与微型PSO链路。
    def __init__(self, regions: Sequence[RegionSpec], targets: GlobalTargets, config: AlgorithmConfig | None = None, background_error: float = 0.0, background_resource: float = 0.0) -> None:  # 初始化完整算法。
        self.regions = tuple(regions)  # 固化区域顺序与区域数量。
        self.targets = targets  # 保存全局误差资源目标。
        self.config = config or AlgorithmConfig()  # 使用显式配置或默认配置。
        validate_problem(self.regions, self.targets, self.config)  # 验证全部输入和一到两代限制。
        self.surrogate = PowerLawSurrogate(self.regions, background_error, background_resource)  # 构造粗解锚定响应代理。
        self.global_agent = GlobalAgent(self.regions, self.targets, self.config, self.surrogate)  # 构造总智能体。
        self.regional_agents = tuple(RegionalAgent(region, self.config, self.surrogate) for region in self.regions)  # 为每个区域构造一个且仅一个子智能体。
        self.communication_hub = CommunicationHub(self.regions, self.targets, self.config, self.surrogate)  # 构造一次同步通信枢纽。
        self.micro_pso = MicroPSO(self.regions, self.config, self.global_agent, self.surrogate)  # 构造二维微型PSO校准器。
        self.run_calls = 0  # 初始化完整算法运行计数。
    def run(self, certifier: CertificationCallable | None = None) -> RunResult:  # 按固定状态机执行一次完整流程。
        if self.run_calls != 0:  # 阻止同一编排器重复运行而破坏一次性计数。
            raise RuntimeError("OneShotDelegatedRegionPSO instances may run only once")  # 报告状态机违规。
        self.run_calls += 1  # 记录唯一一次完整运行。
        delegated_sizes = self.global_agent.delegate_once()  # 执行总智能体唯一一次区域尺寸委派。
        messages: list[RegionMessage] = []  # 初始化区域通信消息集合。
        for agent in self.regional_agents:  # 让每个区域智能体独立处理自己的委派尺寸。
            agent.estimate_once(delegated_sizes[agent.region.region_id])  # 执行该区域唯一一次误差资源估计。
            messages.append(agent.build_message_once())  # 形成该区域唯一一次通信消息。
        communication = self.communication_hub.exchange_once(messages)  # 执行全部区域唯一一轮同步通信。
        message_map = {message.region_id: message for message in messages}  # 建立消息索引供各区域读取邻居信息。
        decisions = tuple(agent.adjust_once(communication, message_map) for agent in self.regional_agents)  # 让每个区域智能体通信后仅调整一次尺寸。
        base_sizes = {decision.region_id: decision.adjusted_size for decision in decisions}  # 将全部区域调整结果编译成唯一母粒子。
        base_evaluation = self.global_agent.evaluate(base_sizes, 0.0, 0.0)  # 执行总智能体在PSO前唯一一次母粒子评估。
        base_particle = CalibrationParticle(0.0, 0.0, base_evaluation)  # 封装唯一母粒子。
        transfer_direction = self.micro_pso.build_transfer_direction(base_sizes, message_map)  # 由区域消息编译唯一资源转移方向。
        selected_particle, pso_history = self.micro_pso.calibrate(base_sizes, transfer_direction, base_evaluation)  # 仅围绕母粒子执行一到两代二维PSO。
        certification: Mapping[str, Any] | None = None  # 初始化终局高保真认证结果。
        high_fidelity_calls = 0  # 初始化终局高保真调用计数。
        if certifier is not None:  # 检查是否提供真实求解器认证接口。
            certification = certifier(selected_particle.evaluation.sizes)  # 仅对PSO选中的一个终局粒子调用高保真求解器。
            high_fidelity_calls = 1  # 记录唯一一次终局高保真认证。
        trace = RunTrace(self.global_agent.delegation_calls, sum(agent.estimate_calls for agent in self.regional_agents), self.communication_hub.exchange_calls, sum(agent.adjustment_calls for agent in self.regional_agents), 1, 2, self.config.pso_generations, self.micro_pso.extra_evaluations, self.global_agent.evaluation_calls, high_fidelity_calls)  # 汇总严格状态机计数。
        expected_regions = len(self.regions)  # 计算预期区域智能体数量。
        if trace.global_delegations != 1 or trace.communication_rounds != 1 or trace.global_base_reviews != 1:  # 检查三个全局一次性环节。
            raise RuntimeError("global one-shot state machine invariant failed")  # 报告全局状态机违规。
        if trace.regional_estimates != expected_regions or trace.regional_adjustments != expected_regions:  # 检查每个区域是否恰好估计和调整一次。
            raise RuntimeError("regional one-shot state machine invariant failed")  # 报告区域状态机违规。
        expected_pso_evaluations = 4 + 5 * self.config.pso_generations  # 计算复用母粒子缓存后的PSO额外评估次数。
        if trace.pso_surrogate_evaluations != expected_pso_evaluations:  # 检查微型PSO是否偷偷增加粒子或迭代。
            raise RuntimeError("micro-PSO evaluation budget invariant failed")  # 报告PSO预算违规。
        if trace.total_surrogate_evaluations != 1 + expected_pso_evaluations:  # 检查母粒子评估与PSO评估总数。
            raise RuntimeError("surrogate evaluation accounting invariant failed")  # 报告代理调用计数错误。
        return RunResult(dict(delegated_sizes), tuple(messages), communication, decisions, base_particle, dict(transfer_direction), selected_particle, pso_history, certification, trace)  # 返回完整可审计结果。
def region_from_mapping(payload: Mapping[str, Any]) -> RegionSpec:  # 从JSON映射构造区域合同。
    return RegionSpec(region_id=str(payload["region_id"]), reference_size=float(payload["reference_size"]), delegated_size=float(payload["delegated_size"]), minimum_size=float(payload["minimum_size"]), maximum_size=float(payload["maximum_size"]), reference_error=float(payload["reference_error"]), reference_resource=float(payload["reference_resource"]), error_order=float(payload["error_order"]), resource_dimension=float(payload["resource_dimension"]), target_error=float(payload["target_error"]), target_resource=float(payload["target_resource"]), confidence=float(payload.get("confidence", 1.0)), neighbors=tuple(str(value) for value in payload.get("neighbors", ())))  # 显式转换全部区域字段。
def config_from_mapping(payload: Mapping[str, Any] | None) -> AlgorithmConfig:  # 从JSON映射构造算法配置。
    if payload is None:  # 处理未提供配置的情况。
        return AlgorithmConfig()  # 返回默认的一代微型PSO配置。
    allowed = set(AlgorithmConfig.__dataclass_fields__)  # 读取配置合同允许的字段集合。
    unknown = set(payload) - allowed  # 查找未声明的配置字段。
    if unknown:  # 检查是否存在拼写错误或隐式新参数。
        raise ValueError(f"unknown config fields: {sorted(unknown)}")  # 拒绝静默忽略未知配置。
    return AlgorithmConfig(**dict(payload))  # 构造强类型算法配置。
def algorithm_from_mapping(payload: Mapping[str, Any]) -> OneShotDelegatedRegionPSO:  # 从完整JSON载荷构造算法实例。
    regions = tuple(region_from_mapping(item) for item in payload["regions"])  # 构造全部区域合同。
    target_payload = payload["targets"]  # 读取全局目标字段。
    targets = GlobalTargets(float(target_payload["error_limit"]), float(target_payload["resource_budget"]))  # 构造全局目标合同。
    config = config_from_mapping(payload.get("config"))  # 构造一到两代算法配置。
    background_error = float(payload.get("background_error", 0.0))  # 读取不可调背景误差。
    background_resource = float(payload.get("background_resource", 0.0))  # 读取不可调背景资源。
    return OneShotDelegatedRegionPSO(regions, targets, config, background_error, background_resource)  # 返回可运行编排器。
