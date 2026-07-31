from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 组合本轮可执行字段用于作用域严格的语义分类。
from pathlib import Path  # 管理冻结轮次目录。
from typing import Any  # 表示动态提案、映射和公开反馈结构。

from . import executor_adapter_v6 as previous_adapter  # 复用真实位移提取、纯后处理和任务控制实现。
from .contracts import canonical_json  # 规范序列化冻结提案用于完整性检查。
from .contracts import sha256_text  # 计算提案摘要以验证分类过程没有改写内容。
from .contracts import verify_frozen_proposal  # 读取并验证已经冻结的模型提案。

DISPLACEMENT_PROBE_OPERATION = previous_adapter.DISPLACEMENT_PROBE_OPERATION  # 复用v6原始裂纹面位移操作标识。
DISPLACEMENT_K_OPERATION = previous_adapter.DISPLACEMENT_K_OPERATION  # 复用v6位移法K联合操作标识。
DISPLACEMENT_MATERIAL_OPERATION = previous_adapter.DISPLACEMENT_MATERIAL_OPERATION  # 复用v6位移和材料联合操作标识。
STRESS_POSTPROCESS_OPERATION = previous_adapter.STRESS_POSTPROCESS_OPERATION  # 复用v6纯应力后处理操作标识。
_V6_MAP_FROZEN_PROPOSAL = previous_adapter.map_frozen_proposal  # 保存未修补的v6映射入口以避免兼容别名产生递归。


def _actionable_text(proposal: dict[str, Any]) -> str:  # 只读取本轮实际要求执行的字段而排除背景不确定性。
    proposal_type = str(proposal.get("proposal_type", ""))  # 读取模型选择的提案类型。
    if proposal_type == "request_information":  # 检查本轮是否是信息请求。
        request = proposal.get("information_request", {})  # 读取信息请求对象。
        return str(request.get("question", "")).lower() if isinstance(request, dict) else ""  # 只使用本轮问题文本进行分类。
    if proposal_type == "experiment":  # 检查本轮是否是受控实验。
        experiment = proposal.get("experiment", {})  # 读取实验设计对象。
        if not isinstance(experiment, dict):  # 检查实验结构是否有效。
            return ""  # 在无效结构时返回空文本并交给既有合同处理。
        payload = {"purpose": experiment.get("purpose", ""), "change": experiment.get("change", ""), "measure": experiment.get("measure", [])}  # 只组合本轮目的、改变和测量项。
        return json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()  # 返回不含uncertainties的可执行语义文本。
    return ""  # 对路线控制和任务完成提案不执行位移分类。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 修正背景不确定性污染本轮执行目的的问题。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在分类前验证冻结提案完整性。
    before_hash = sha256_text(canonical_json(proposal))  # 保存分类前提案摘要。
    proposal_type = str(proposal.get("proposal_type", ""))  # 读取本轮提案类型。
    actionable = _actionable_text(proposal)  # 提取仅代表本轮执行目的的自然语言。
    asks_displacement = previous_adapter._contains_any(actionable, previous_adapter._DISPLACEMENT_TERMS)  # 检查本轮是否明确要求位移数据。
    if proposal_type in {"experiment", "request_information"} and asks_displacement:  # 在v6全提案扫描前优先完成作用域严格分类。
        asks_fracture = previous_adapter._contains_any(actionable, previous_adapter._FRACTURE_TERMS)  # 检查本轮是否要求由位移形成断裂评价量。
        asks_material = previous_adapter._contains_any(actionable, previous_adapter._MATERIAL_TERMS)  # 检查本轮动作是否主动请求材料事实。
        after_hash = sha256_text(canonical_json(proposal))  # 重新计算分类后提案摘要。
        if after_hash != before_hash:  # 检查分类过程是否改写冻结提案。
            raise RuntimeError("executor mutated the frozen proposal")  # 发现内容改变时拒绝执行。
        if asks_material:  # 检查本轮是否确实同时要求材料数据。
            return previous_adapter._write_mapping(round_dir, proposal_hash, DISPLACEMENT_MATERIAL_OPERATION, "actionable fields request crack-face displacement evidence together with external material facts")  # 保留位移和材料联合目的。
        if asks_fracture:  # 检查本轮是否要求位移法断裂评价量。
            return previous_adapter._write_mapping(round_dir, proposal_hash, DISPLACEMENT_K_OPERATION, "actionable fields request crack-face displacement extraction and a displacement-derived fracture quantity")  # 映射为完整位移法K实验。
        return previous_adapter._write_mapping(round_dir, proposal_hash, DISPLACEMENT_PROBE_OPERATION, "actionable fields request raw crack-face or nodal displacement evidence")  # 映射为原始位移数据提取。
    return _V6_MAP_FROZEN_PROPOSAL(round_dir, proposal_hash)  # 对其他提案复用原始v6全部忠实映射合同。


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 复用v6真实执行路径。
    return previous_adapter.execute_mapping(round_dir, proposal_hash, mapping)  # 返回位移、纯后处理、路线控制或其他隐藏实验结果。


previous_adapter.map_frozen_proposal = map_frozen_proposal  # 为旧v6测试和兼容导入暴露相同的作用域严格映射入口。
