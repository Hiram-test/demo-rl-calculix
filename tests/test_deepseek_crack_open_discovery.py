from __future__ import annotations  # 允许测试使用现代类型注解。

import ast  # 在不执行 API 调用的情况下读取脚本常量。
import unittest  # 使用标准库测试框架运行静态合同检查。
from pathlib import Path  # 定位实验脚本与协议文档。

ROOT = Path(__file__).resolve().parents[1]  # 定位仓库根目录。
SCRIPT_PATH = ROOT / "scripts" / "run_deepseek_crack_open_discovery.py"  # 定位开放发现运行脚本。
DOC_PATH = ROOT / "docs" / "DEEPSEEK_CRACK_OPEN_DISCOVERY.md"  # 定位实验协议文档。


def _assignment_value(name: str):  # 从脚本 AST 中安全读取字面量常量。
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))  # 解析脚本而不导入第三方依赖。
    for node in tree.body:  # 遍历模块顶层语句。
        if isinstance(node, ast.Assign):  # 只检查普通赋值语句。
            for target in node.targets:  # 检查赋值的所有目标。
                if isinstance(target, ast.Name) and target.id == name:  # 匹配所需常量名。
                    return ast.literal_eval(node.value)  # 安全求值字符串、集合或字典字面量。
    raise AssertionError(f"assignment not found: {name}")  # 在常量缺失时给出明确错误。


class CrackOpenDiscoveryContractTests(unittest.TestCase):  # 定义开放发现实验的静态合同测试。
    def test_script_compiles(self) -> None:  # 验证运行脚本具有合法 Python 语法。
        ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))  # 解析完整脚本并在语法错误时失败。

    def test_guided_prompt_has_no_forced_crack_route(self) -> None:  # 验证高层指南没有写死裂纹答案。
        guided = _assignment_value("GUIDED_PRINCIPLES")  # 读取模型实际使用的工程师指南。
        forbidden = ("必须调用", "必须先", "J积分", "能量释放率", "应力强度因子", "弹塑性", "compute_linear_J_G", "run_elastoplastic_crack_sequence")  # 定义不能出现在运行时指南中的路线词。
        self.assertTrue(all(token not in guided for token in forbidden))  # 确保指南只保留通用工程原则。

    def test_toolbox_allows_real_alternatives_and_free_finish(self) -> None:  # 验证工具目录不是单一路线。
        actions = _assignment_value("ALLOWED_ACTIONS")  # 读取运行时允许动作集合。
        self.assertIn("finish", actions)  # 确保模型可以自主结束。
        self.assertIn("refine_current_model", actions)  # 确保模型仍可选择继续加密。
        self.assertIn("probe_fixed_physical_locations", actions)  # 确保模型可选择改变取样协议。
        self.assertIn("compare_region_average", actions)  # 确保模型可选择区域平均。
        self.assertIn("compare_nearby_geometry_energy", actions)  # 确保模型可选择几何扰动能量比较。
        self.assertIn("request_material_curve", actions)  # 确保模型可请求缺失材料信息。
        self.assertGreaterEqual(len(actions), 7)  # 确保工具目录具有足够分支而非伪选择。

    def test_runtime_contract_does_not_require_action_names(self) -> None:  # 验证输出合同只约束结构。
        contract = _assignment_value("OUTPUT_CONTRACT")  # 读取模型实际使用的输出合同。
        actions = _assignment_value("ALLOWED_ACTIONS")  # 读取全部工具名称。
        self.assertTrue(all(action not in contract for action in actions))  # 确保合同没有点名任何必选动作。

    def test_protocol_explicitly_separates_guidance_from_route(self) -> None:  # 验证协议文件记录实验边界。
        text = DOC_PATH.read_text(encoding="utf-8")  # 读取实验协议全文。
        self.assertIn("高层工作原则", text)  # 确认允许提供工程师知识框架。
        self.assertIn("不得列出任何场景专用完成要求", text)  # 确认禁止场景专用完成门。
        self.assertIn("评分器不参与运行时决策", text)  # 确认事后评分不会泄漏给模型。

    def test_each_nonblank_script_line_has_comment_marker(self) -> None:  # 验证实验代码每一行均带解释性注释。
        lines = SCRIPT_PATH.read_text(encoding="utf-8").splitlines()  # 读取运行脚本的全部行。
        missing = [index for index, line in enumerate(lines, start=1) if line.strip() and "#" not in line]  # 找出没有注释标记的非空行。
        self.assertEqual([], missing, f"uncommented lines: {missing}")  # 在任何代码行缺少注释时失败。


if __name__ == "__main__":  # 允许直接执行本测试文件。
    unittest.main()  # 启动标准库测试运行器。
