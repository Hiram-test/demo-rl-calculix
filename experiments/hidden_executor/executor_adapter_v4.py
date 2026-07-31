from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 保存忠实性拒绝映射收据。
from pathlib import Path  # 管理冻结轮次目录。
from typing import Any  # 表示动态提案、映射和反馈结构。

from . import executor_adapter_v3 as previous_adapter  # 复用目标尺寸忠实执行和公开反馈净化层。
from .contracts import canonical_json  # 规范序列化提案用于完整性检查。
from .contracts import sha256_text  # 计算提案摘要以验证执行器没有改写内容。
from .contracts import verify_frozen_proposal  # 读取并验证冻结提案。


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:  # 对中英文自然语言执行确定性包含检查。
    return any(term in text for term in terms)  # 任一语义短语出现时返回真。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 在所有映射前阻止只执行联合实验的一部分。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 读取并验证冻结提案。
    before_hash = sha256_text(canonical_json(proposal))  # 保存分类前提案摘要。
    if proposal.get("proposal_type") == "experiment":  # 只检查模型明确提出的控制实验。
        experiment = proposal.get("experiment", {})  # 读取冻结实验设计对象。
        change_text = str(experiment.get("change", "")).lower()  # 读取准备改变的变量。
        measure_text = " ".join(str(item) for item in experiment.get("measure", [])).lower()  # 读取模型要求的测量输出。
        requests_refinement = _contains_any(change_text, ("加密", "细化", "减小", "refine", "refinement", "reduce", "smaller")) and _contains_any(change_text, ("网格", "单元", "mesh", "element"))  # 检查是否明确改变网格分辨率。
        requests_fracture_quantity = _contains_any(measure_text, ("应力强度因子", "能量释放率", "j积分", "j 积分", "stress intensity factor", "energy release rate", "j-integral", "j integral", "k_i", "ki"))  # 检查是否要求断裂参量。
        requests_supported_energy_method = _contains_any(measure_text, ("总应变能", "能量差", "新增裂纹", "strain energy", "energy difference", "newly created crack", "k =", "sqrt(e", "√(e"))  # 检查是否明确指定当前后端可实现的能量差协议。
        if requests_refinement and requests_fracture_quantity and not requests_supported_energy_method:  # 检查是否为当前无法完整执行的联合实验。
            after_hash = sha256_text(canonical_json(proposal))  # 重新计算分类后提案摘要。
            if after_hash != before_hash:  # 检查分类过程中是否改写冻结提案。
                raise RuntimeError("executor mutated the frozen proposal")  # 发现改写时拒绝继续。
            mapping = {"proposal_sha256": proposal_hash, "operation": "unsupported", "mapping_reason": "executor cannot preserve the requested fracture-parameter extraction method while refining the mesh", "proposal_unchanged": True}  # 组织诚实不支持映射收据。
            (round_dir / "mapping_receipt.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存模型不可见的拒绝理由。
            return mapping  # 返回不支持而不执行普通加密替代实验。
    return previous_adapter.map_frozen_proposal(round_dir, proposal_hash)  # 对能够忠实执行的其他提案复用v3映射。


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 委托v3执行并保留统一公开反馈净化。
    return previous_adapter.execute_mapping(round_dir, proposal_hash, mapping)  # 对不支持映射返回诚实限制，对支持映射执行真实计算。
