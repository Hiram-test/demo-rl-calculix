"""验证三案例 T0 就绪性活动保持正确执行技术族、缺失事实和零执行边界。"""  # 限定测试只检查 fail-closed 软件行为。
from __future__ import annotations  # 启用现代类型注解行为，兼容 GitHub Actions Python。

import copy  # 复制 profile 以测试错误执行技术族会被拒绝。
import json  # 读取 runner 在临时 artifact 中写出的 receipt 和 manifest。
import sys  # 将当前仓库根加入模块搜索路径。
import tempfile  # 为每项测试创建自动回收的独立 artifact 目录。
import unittest  # 使用 Python 标准库执行轻量确定性测试。
from pathlib import Path  # 用跨平台路径对象定位仓库和测试输出。

REPO_ROOT = Path(__file__).resolve().parents[1]  # 以 tests 目录的父目录作为当前仓库根。
if str(REPO_ROOT) not in sys.path:  # 只在仓库根尚未注册时修改导入路径。
    sys.path.insert(0, str(REPO_ROOT))  # 允许导入当前提交的 scripts 命名空间。

from scripts import run_v5_high_layer_readiness as readiness  # 导入当前提交的就绪性 runner 供确定性测试。

PROFILES_DIR = REPO_ROOT / "examples/v5_high_layer"  # 指向三个当前提高层 profile。
SKILLS_DIR = REPO_ROOT / "skills/engineering"  # 指向当前三项通用工程 Skill。
EXPECTED_FAMILIES = {  # 冻结用户明确确认的案例与执行技术族映射。
    "FEN-003": "FEniCS/FEniCSx",  # FEN-003 必须走 FEniCS/FEniCSx。
    "FEN-014": "FEniCS/FEniCSx",  # FEN-014 必须走 FEniCS/FEniCSx。
    "CCX-015": "CalculiX",  # CCX-015 必须走 CalculiX。
}  # 结束执行技术族映射。
EXPECTED_MISSING_COUNTS = {  # 冻结 T0 必须完整保留的当前缺失事实数量。
    "FEN-003": 5,  # FEN-003 包含运行时、模型、QoI、网格序列和判据五项缺口。
    "FEN-014": 7,  # FEN-014 包含运行时、外部网格、格式、参考、映射、单位和失败日志七项缺口。
    "CCX-015": 8,  # CCX-015 包含运行时、细化工具链和六项当前细化证据缺口。
}  # 结束缺失事实数量映射。


class HighLayerReadinessTests(unittest.TestCase):  # 定义三案例 fail-closed 软件合同测试组。
    def test_current_campaign_blocks_all_cases_before_execution(self) -> None:  # 验证当前无案例输入时三案都正确阻断且工作流整体成功。
        with tempfile.TemporaryDirectory(prefix="v5-readiness-") as temporary_name:  # 创建独立且自动回收的测试 artifact 目录。
            output_dir = Path(temporary_name) / "artifact"  # 在临时目录内定义 runner 输出根。
            campaign = readiness.run_campaign(PROFILES_DIR, SKILLS_DIR, output_dir)  # 执行当前三个 profile 的真实 T0 来源审计。
            self.assertEqual(campaign["status"], "success")  # T0 正确阻断应视为软件测试成功。
            self.assertEqual(campaign["case_count"], 3)  # 活动必须恰好覆盖三个冻结案例。
            self.assertEqual(campaign["blocked_case_count"], 3)  # 三案都应因缺少当前证据而阻断。
            self.assertFalse(campaign["engineering_results_generated"])  # T0 不得生成任何工程结果。
            self.assertTrue(all(value == 0 for value in campaign["execution_counts"].values()))  # 所有模型、FEniCS、CalculiX、网格和优化调用必须为零。
            for case_id, expected_family in EXPECTED_FAMILIES.items():  # 逐案检查机器 receipt 的映射和边界。
                receipt_path = output_dir / case_id.lower() / "readiness_receipt.json"  # 定位当前案例主 receipt。
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))  # 解析 runner 真实写出的 UTF-8 证据。
                self.assertEqual(receipt["execution_family"], expected_family)  # 要求 FEN 与 CCX 不再接错执行后端。
                self.assertEqual(receipt["test_outcome"], "pass")  # 当前 profile 必须通过 T0 合同测试。
                self.assertEqual(receipt["case_status"], "blocked_missing_current_evidence")  # 工程案例仍必须保持阻断。
                self.assertEqual(receipt["stop_stage"], "source_audit")  # 必须在网格、弱式和求解之前停止。
                self.assertEqual(len(receipt["missing_facts"]), EXPECTED_MISSING_COUNTS[case_id])  # 缺失事实不得被静默删除。
                self.assertTrue(all(value == 0 for value in receipt["execution_counts"].values()))  # 当前案例的全部执行调用必须为零。

    def test_swapped_execution_family_is_rejected(self) -> None:  # 验证把 FEN-003 错接到 CalculiX 会产生明确失败检查。
        profile_path = PROFILES_DIR / "fen-003.json"  # 定位当前 FEN-003 profile。
        profile = json.loads(profile_path.read_text(encoding="utf-8"))  # 读取当前权威 profile 内容。
        swapped = copy.deepcopy(profile)  # 创建不会修改仓库文件的独立测试副本。
        swapped["execution_family"] = "CalculiX"  # 注入用户刚刚纠正的错误后端映射。
        library = readiness.SkillLibrary.load_json_directory(SKILLS_DIR)  # 使用当前 Skill 合同加载真实 Skill 库。
        checks = readiness.validate_profile(swapped, "FEN-003", readiness.CASE_CONTRACTS["FEN-003"], library)  # 对错误副本运行同一入口验证。
        family_check = next(row for row in checks if row["name"] == "execution_family_exact")  # 取得专门检查执行技术族的结果。
        self.assertFalse(family_check["passed"])  # 错接到 CalculiX 必须被确定性拒绝。

    def test_manifest_questions_match_profile_missing_facts(self) -> None:  # 验证 ProblemManifest 问题清单与 profile 缺口一一对应。
        for case_id, contract in readiness.CASE_CONTRACTS.items():  # 按三个冻结案例逐一核对。
            profile_path = PROFILES_DIR / str(contract["profile_name"])  # 根据 runner 冻结路径定位 profile。
            profile = json.loads(profile_path.read_text(encoding="utf-8"))  # 解析当前 profile。
            profile_record = readiness.source_record(profile_path)  # 为当前 profile 生成真实哈希来源记录。
            skill_paths = [SKILLS_DIR / (str(skill_id) + ".json") for skill_id in contract["expected_skills"]]  # 定位当前案例引用的真实 Skill。
            skill_records = [readiness.source_record(path) for path in skill_paths]  # 为每个 Skill 生成真实来源记录。
            manifest = readiness.build_manifest(profile, profile_record, skill_records)  # 用当前 V5 合同生成无物理事实 manifest。
            manifest_payload = manifest.to_dict()  # 转换为机器可读对象供断言。
            expected_paths = [str(row["path"]) for row in profile["missing_facts"]]  # 读取 profile 原始缺失事实顺序。
            actual_paths = [str(row["path"]) for row in manifest_payload["missing_facts"]]  # 读取 manifest 保留后的缺失事实顺序。
            question_paths = [str(row["field"]) for row in manifest.questions_for_user()]  # 读取面向用户的问题字段顺序。
            self.assertEqual(actual_paths, expected_paths)  # manifest 不得删除、重排或改名任何缺口。
            self.assertEqual(question_paths, expected_paths)  # 用户问题必须与当前缺口逐项对应。
            self.assertEqual(manifest_payload["facts"], {})  # 没有当前输入时物理事实集合必须为空。
            self.assertEqual(manifest_payload["algorithm_configuration"], {})  # T0 不得用算法配置夹带默认值。


if __name__ == "__main__":  # 仅在测试文件直接运行时启动标准库测试器。
    unittest.main()  # 执行全部三项 fail-closed 单元测试。
