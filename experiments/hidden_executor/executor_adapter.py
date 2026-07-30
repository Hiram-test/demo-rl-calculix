from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 更新隐藏映射收据和脱敏反馈文件。
from pathlib import Path  # 管理冻结轮次目录。
from typing import Any  # 表示动态提案、映射和反馈结构。

from .contracts import verify_frozen_proposal  # 读取并验证冻结提案内容。
from .executor import _load_backend  # 复用既有只读有限元后端加载器。
from .executor import execute_mapping as _base_execute_mapping  # 复用已经实现的真实执行逻辑。
from .executor import map_frozen_proposal as _base_map_frozen_proposal  # 复用基础确定性映射逻辑。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 对自然语言语序差异增加保守兼容层。
    mapping = _base_map_frozen_proposal(round_dir, proposal_hash)  # 首先运行基础隐藏映射。
    if mapping.get("operation") != "unsupported":  # 在基础映射已经明确时保持原结果。
        return mapping  # 返回基础忠实映射。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 重新读取并验证冻结提案。
    text = json.dumps(proposal, ensure_ascii=False).lower()  # 组合提案全部自然语言字段。
    has_crack_subject = "裂纹" in text  # 检查实验对象是否明确是裂纹。
    has_extension_change = any(token in text for token in ("延长", "扩展", "增长", "增量", "变长"))  # 检查是否明确改变裂纹长度。
    has_energy_measure = any(token in text for token in ("能量", "势能", "应变能", "释放"))  # 检查是否明确比较能量量。
    if has_crack_subject and has_extension_change and has_energy_measure:  # 只在三个物理条件同时满足时补充语序无关映射。
        mapping = {"proposal_sha256": proposal_hash, "operation": "geometry_energy", "mapping_reason": "proposal changes crack length and compares an energy quantity regardless of word order", "proposal_unchanged": True}  # 生成忠实内部映射收据。
        (round_dir / "mapping_receipt.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 覆盖基础不支持收据并保留审计记录。
    return mapping  # 返回最终隐藏映射结果。


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 对公开结束状态进行中性化处理。
    feedback = _base_execute_mapping(round_dir, proposal_hash, mapping)  # 执行基础真实求解或停止逻辑。
    if mapping.get("operation") != "finish":  # 在非结束操作时保持原始物理反馈。
        return feedback  # 返回基础脱敏反馈。
    feedback = dict(feedback)  # 复制反馈以避免修改调用方持有的对象。
    feedback["status"] = "analysis_complete"  # 使用不含内部操作词的中性公开状态。
    feedback["observations"] = {"analysis_complete": True}  # 使用中性布尔观测替代内部状态字符串。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 重写下一轮模型可见反馈。
    audit_path = round_dir / "execution_audit.json"  # 定位完整内部审计文件。
    audit = json.loads(audit_path.read_text(encoding="utf-8"))  # 读取基础执行审计。
    audit["public_feedback"] = feedback  # 让审计文件引用最终中性公开反馈。
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存更新后的完整审计记录。
    return feedback  # 返回中性化的公开结束反馈。
