from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

from dataclasses import dataclass  # 定义不可变Skill合同和运行上下文。
from typing import Any, Callable  # 表示动态参数、结果和Skill处理函数。

from experiments.hidden_executor.contracts import canonical_json  # 生成稳定Skill目录文本。
from experiments.hidden_executor.contracts import sha256_text  # 计算Skill目录版本摘要。


@dataclass(frozen=True)  # 使用不可变结构防止规划期间修改Skill合同。
class SkillContext:  # 定义Skill运行时可以读取的真实证据上下文。
    initial_evidence: dict[str, Any]  # 保存用户问题、模型事实和初始有限元证据。
    public_history: list[dict[str, Any]]  # 保存第一API已经看到的逐轮公开历史。
    prior_skill_outputs: dict[str, dict[str, Any]]  # 保存当前执行图中前序Skill输出。


SkillHandler = Callable[[dict[str, Any], SkillContext], dict[str, Any]]  # 定义所有Skill处理函数统一签名。


@dataclass(frozen=True)  # 使用不可变结构保存单项Skill能力合同。
class SkillDefinition:  # 描述一个可由第二API选择的隐藏Skill。
    skill_id: str  # 保存稳定且只对第二API可见的Skill标识。
    description: str  # 说明该Skill忠实完成的工程或数值任务。
    input_schema: dict[str, dict[str, Any]]  # 定义参数类型、必需性和语义说明。
    output_fields: list[str]  # 列出Skill能够返回的公开物理字段。
    effects: list[str]  # 列出Skill会改变或读取的模型对象。
    limitations: list[str]  # 列出该能力固有的工程边界。
    handler: SkillHandler  # 保存确定性执行函数。

    def public_descriptor(self) -> dict[str, Any]:  # 构造仅发送给第二API的Skill目录条目。
        return {"skill_id": self.skill_id, "description": self.description, "input_schema": self.input_schema, "output_fields": self.output_fields, "effects": self.effects, "limitations": self.limitations}  # 排除Python函数和内部对象。


class SkillRegistry:  # 管理隐藏Skill目录、参数合同和确定性处理函数。
    def __init__(self) -> None:  # 初始化空Skill目录。
        self._skills: dict[str, SkillDefinition] = {}  # 使用Skill标识索引不可变能力定义。

    def register(self, skill: SkillDefinition) -> None:  # 注册一项新的隐藏Skill。
        if skill.skill_id in self._skills:  # 检查Skill标识是否已经存在。
            raise ValueError(f"duplicate skill id: {skill.skill_id}")  # 拒绝覆盖既有能力合同。
        self._skills[skill.skill_id] = skill  # 保存通过唯一性检查的Skill定义。

    def get(self, skill_id: str) -> SkillDefinition:  # 按稳定标识读取Skill定义。
        if skill_id not in self._skills:  # 检查第二API是否引用未知能力。
            raise KeyError(f"unknown skill id: {skill_id}")  # 拒绝不存在的Skill调用。
        return self._skills[skill_id]  # 返回不可变Skill合同。

    def catalog(self) -> list[dict[str, Any]]:  # 构造发送给第二API的排序Skill目录。
        return [self._skills[key].public_descriptor() for key in sorted(self._skills)]  # 固定排序保证目录哈希可复现。

    def catalog_hash(self) -> str:  # 计算当前Skill目录版本摘要。
        return sha256_text(canonical_json(self.catalog()))  # 对不含处理函数的公开描述生成SHA256。

    def ids(self) -> set[str]:  # 返回全部已注册Skill标识集合。
        return set(self._skills)  # 复制键集合避免调用方修改内部目录。

    def validate_arguments(self, skill_id: str, arguments: dict[str, Any]) -> list[str]:  # 按Skill合同校验第二API生成的参数。
        errors: list[str] = []  # 初始化参数错误列表。
        skill = self.get(skill_id)  # 读取目标Skill定义并拒绝未知标识。
        schema = skill.input_schema  # 读取该Skill参数合同。
        unknown = set(arguments) - set(schema)  # 找出Skill合同未声明的参数。
        if unknown:  # 检查第二API是否发明了参数。
            errors.append(f"unknown arguments for {skill_id}: {sorted(unknown)}")  # 记录未知参数。
        for name, contract in schema.items():  # 遍历Skill声明的全部参数。
            required = bool(contract.get("required", False))  # 读取参数是否必需。
            if required and name not in arguments:  # 检查必需参数是否缺失。
                errors.append(f"missing required argument {name} for {skill_id}")  # 记录缺失参数。
                continue  # 跳过不存在参数的类型检查。
            if name not in arguments:  # 检查可选参数是否未提供。
                continue  # 未提供可选参数时使用Skill内部明确默认值。
            expected = str(contract.get("type", "any"))  # 读取简化类型描述。
            if not _matches_type(arguments[name], expected):  # 检查实际值是否满足声明类型。
                errors.append(f"argument {name} for {skill_id} must be {expected}")  # 记录类型错误。
        return errors  # 返回全部参数合同错误。


def _matches_type(value: Any, expected: str) -> bool:  # 按简化Skill参数类型检查动态JSON值。
    if expected == "any":  # 允许Skill显式接受任意JSON值。
        return True  # 直接通过任意类型。
    if expected == "number":  # 检查浮点或整数数值但排除布尔值。
        return isinstance(value, (int, float)) and not isinstance(value, bool)  # 返回数值类型判断。
    if expected == "integer":  # 检查整数但排除布尔值。
        return isinstance(value, int) and not isinstance(value, bool)  # 返回整数类型判断。
    if expected == "boolean":  # 检查布尔参数。
        return isinstance(value, bool)  # 返回布尔类型判断。
    if expected == "string":  # 检查非空文本参数。
        return isinstance(value, str) and bool(value.strip())  # 只接受非空字符串。
    if expected == "array[number]":  # 检查数值数组。
        return isinstance(value, list) and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)  # 验证每个元素为数值。
    if expected == "array[string]":  # 检查非空文本数组。
        return isinstance(value, list) and all(isinstance(item, str) and bool(item.strip()) for item in value)  # 验证每个元素为非空字符串。
    if expected == "object":  # 检查JSON对象。
        return isinstance(value, dict)  # 返回对象类型判断。
    return False  # 对未声明的类型描述采取拒绝策略。
