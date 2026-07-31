from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

import ast  # 解析受限数学表达式并禁止任意代码执行。
import math  # 提供安全公式Skill允许的数学函数和常数。
from typing import Any  # 表示动态公式变量、结果和Skill上下文。

from experiments.skill_planner.registry import SkillContext  # 使用统一Skill运行上下文。
from experiments.skill_planner.registry import SkillDefinition  # 定义新增通用公式Skill合同。
from experiments.skill_planner.registry import SkillRegistry  # 扩展既有隐藏Skill目录。
from experiments.skill_planner.skills import build_registry as build_base_registry  # 复用第一阶段已经通过真实运行的有限元Skill。

_ALLOWED_FUNCTIONS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "log": math.log, "abs": abs, "min": min, "max": max}  # 定义安全公式可以调用的纯数学函数。
_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}  # 定义安全公式可以使用的数学常数。
_ALLOWED_BINARY_NODES = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)  # 定义允许的二元算术运算。
_ALLOWED_UNARY_NODES = (ast.UAdd, ast.USub)  # 定义允许的一元正负号。

SKILL_CAPABILITY_TAGS = {  # 定义确定性保真校验使用的Skill能力标签。
    "fracture.energy_sequence": {"method.energy_difference", "quantity.energy_release_rate", "quantity.stress_intensity", "mesh.sequence_comparison"},  # 标记多档能量差Skill的方法和输出能力。
    "fracture.refine_and_energy": {"method.energy_difference", "quantity.energy_release_rate", "quantity.stress_intensity", "mesh.refinement", "mesh.sequence_comparison"},  # 标记目标加密联合Skill。
    "mesh.refine": {"mesh.refinement", "quantity.displacement", "quantity.strain_energy", "quantity.local_stress"},  # 标记普通网格求解能力。
    "fracture.crack_face_displacement": {"method.crack_face_displacement", "method.displacement_extrapolation", "quantity.crack_opening", "quantity.stress_intensity", "mesh.sequence_comparison"},  # 标记真实裂纹面位移法能力。
    "postprocess.richardson": {"method.richardson", "quantity.extrapolated_limit", "mesh.sequence_comparison"},  # 标记网格外推能力。
    "material.request": {"data.external_material"},  # 标记外部材料事实请求能力。
    "postprocess.formula_table": {"method.safe_formula", "method.closed_form_reference", "method.irwin_plastic_zone", "quantity.derived_scalar"},  # 标记受限公式和常见断裂后处理能力。
}  # 完成Skill能力标签目录。


def _validate_expression_node(node: ast.AST, variable_names: set[str]) -> None:  # 递归验证公式AST只包含白名单数学结构。
    if isinstance(node, ast.Expression):  # 检查表达式根节点。
        _validate_expression_node(node.body, variable_names)  # 递归验证根表达式内容。
        return  # 完成根节点验证。
    if isinstance(node, ast.Constant):  # 检查字面常数。
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):  # 检查常数必须是普通数值。
            raise ValueError("formula constants must be numeric")  # 拒绝字符串、布尔值和其他对象。
        return  # 完成数值常量验证。
    if isinstance(node, ast.Name):  # 检查变量或数学常数名称。
        if node.id not in variable_names and node.id not in _ALLOWED_CONSTANTS:  # 检查名称是否来自显式变量或白名单常数。
            raise ValueError(f"unknown formula name: {node.id}")  # 拒绝未声明变量和任意全局名称。
        return  # 完成名称验证。
    if isinstance(node, ast.BinOp):  # 检查二元算术表达式。
        if not isinstance(node.op, _ALLOWED_BINARY_NODES):  # 检查运算符是否在白名单。
            raise ValueError("formula binary operator is not allowed")  # 拒绝位运算和矩阵运算等结构。
        _validate_expression_node(node.left, variable_names)  # 验证左操作数。
        _validate_expression_node(node.right, variable_names)  # 验证右操作数。
        return  # 完成二元表达式验证。
    if isinstance(node, ast.UnaryOp):  # 检查一元正负号。
        if not isinstance(node.op, _ALLOWED_UNARY_NODES):  # 检查一元运算符是否允许。
            raise ValueError("formula unary operator is not allowed")  # 拒绝逻辑非和位反转。
        _validate_expression_node(node.operand, variable_names)  # 验证一元操作数。
        return  # 完成一元表达式验证。
    if isinstance(node, ast.Call):  # 检查纯数学函数调用。
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:  # 检查函数必须是白名单中的直接名称。
            raise ValueError("formula function is not allowed")  # 拒绝属性访问、导入和任意函数调用。
        if node.keywords:  # 检查是否使用关键字参数。
            raise ValueError("formula keyword arguments are not allowed")  # 简化执行语义并拒绝复杂调用。
        for argument in node.args:  # 遍历函数位置参数。
            _validate_expression_node(argument, variable_names)  # 递归验证每个参数。
        return  # 完成函数调用验证。
    raise ValueError(f"formula node is not allowed: {type(node).__name__}")  # 拒绝属性、下标、推导式、条件表达式和其他Python语法。


def _evaluate_expression(expression: str, variables: dict[str, float]) -> float:  # 在白名单环境中计算单行安全数学表达式。
    tree = ast.parse(expression, mode="eval")  # 把公式解析为表达式AST。
    _validate_expression_node(tree, set(variables))  # 在编译前验证所有节点和名称。
    environment = dict(_ALLOWED_CONSTANTS)  # 构造只含数学常数的执行环境。
    environment.update(_ALLOWED_FUNCTIONS)  # 加入白名单纯数学函数。
    environment.update(variables)  # 加入当前行经过数值检查的显式变量。
    value = eval(compile(tree, "<safe_formula>", "eval"), {"__builtins__": {}}, environment)  # 在无内置函数环境中执行已验证表达式。
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):  # 检查结果必须是有限实数。
        raise ValueError("formula result must be a finite number")  # 拒绝无穷、NaN和非数值结果。
    return float(value)  # 返回标准浮点结果。


def _row_count(variables: dict[str, Any]) -> int:  # 确定标量和数组变量共同形成的表格行数。
    lengths = {len(value) for value in variables.values() if isinstance(value, list)}  # 收集全部数组变量长度。
    if len(lengths) > 1:  # 检查数组变量长度是否一致。
        raise ValueError("all array variables must have the same length")  # 拒绝无法逐行对齐的变量表。
    return next(iter(lengths), 1)  # 没有数组时使用单行，有数组时使用共同长度。


def _formula_table(arguments: dict[str, Any], context: SkillContext) -> dict[str, Any]:  # 对冻结提案给出的变量表执行一个或多个安全公式。
    del context  # 公式Skill只使用显式参数，不读取隐藏目录外事实。
    variables = arguments.get("variables")  # 读取变量对象。
    formulas = arguments.get("formulas")  # 读取输出名称到表达式的映射。
    if not isinstance(variables, dict) or not variables:  # 检查至少提供一个显式变量。
        raise ValueError("variables must be a non-empty object")  # 拒绝无输入公式。
    if not isinstance(formulas, dict) or not formulas:  # 检查至少提供一个派生公式。
        raise ValueError("formulas must be a non-empty object")  # 拒绝无输出计划。
    row_count = _row_count(variables)  # 确定逐行计算规模。
    rows: list[dict[str, Any]] = []  # 初始化公开公式结果表。
    for index in range(row_count):  # 遍历每一组变量取值。
        row_variables: dict[str, float] = {}  # 初始化当前行数值变量。
        for name, raw_value in variables.items():  # 遍历全部输入变量。
            value = raw_value[index] if isinstance(raw_value, list) else raw_value  # 对数组取当前行，对标量广播。
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):  # 检查变量必须是有限数值。
                raise ValueError(f"variable {name} must contain finite numbers")  # 拒绝文本、布尔和无穷值。
            row_variables[str(name)] = float(value)  # 保存标准化当前行变量。
        outputs: dict[str, float] = {}  # 初始化当前行派生结果。
        for output_name, expression in formulas.items():  # 遍历全部派生公式。
            if not isinstance(expression, str) or not expression.strip():  # 检查表达式必须是非空文本。
                raise ValueError(f"formula {output_name} must be a non-empty string")  # 拒绝无效公式。
            outputs[str(output_name)] = _evaluate_expression(expression, row_variables)  # 安全计算当前派生量。
        rows.append({"inputs": row_variables, "outputs": outputs})  # 保存当前行完整输入和输出。
    return {"status": "completed", "executed_change": "不运行新的有限元模型，使用冻结提案和既有证据中的数值执行受限数学公式表", "actual_parameters": {"row_count": row_count, "formula_names": [str(name) for name in formulas]}, "observations": {"rows": rows}, "limitations": ["公式Skill只执行白名单算术和纯数学函数，不验证公式对应的物理假设是否适用", "输入数值和公式必须来自冻结提案或既有证据，不能由Skill自行补造", "派生结果不能替代缺失的材料试验或非线性有限元证据"]}  # 返回不泄露内部求值器的公开后处理结果。


def build_registry() -> SkillRegistry:  # 构造Phase 2隐藏Skill目录。
    registry = build_base_registry()  # 复用已经通过真实双API运行的六项基础Skill。
    registry.register(SkillDefinition(skill_id="postprocess.formula_table", description="对冻结提案或既有证据中的标量和等长数组执行受限数学公式，支持解析参照、塑性区估计、比值和其他纯后处理派生量。", input_schema={"variables": {"type": "object", "required": True, "description": "变量名到有限数值或等长数值数组的映射。"}, "formulas": {"type": "object", "required": True, "description": "输出名称到白名单数学表达式的映射。"}}, output_fields=["rows", "inputs", "outputs"], effects=["只执行已有数据的数学后处理", "不运行有限元模型", "不生成外部材料事实"], limitations=["不验证物理公式适用性", "禁止任意Python代码和未声明变量"], handler=_formula_table))  # 注册通用安全公式Skill。
    return registry  # 返回扩展后的不可变Skill注册表。
