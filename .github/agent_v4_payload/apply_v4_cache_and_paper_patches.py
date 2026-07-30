# 启用延迟类型注解，保持补丁脚本在 Python 3.12 下的声明清晰。
from __future__ import annotations

# 导入文件复制模块，用于把受版本控制的新模块安装到哈希校验后的载荷目录。
import shutil
# 导入命令行参数模块，用于接收载荷展开根目录。
import sys
# 导入路径类型，用于以跨平台方式定位源资产和目标文件。
from pathlib import Path


# 定位本补丁脚本所在的受版本控制资产目录。
ASSET_ROOT = Path(__file__).resolve().parent


# 定义 UTF-8 精确替换函数，使上游载荷漂移时立即给出明确错误。
def _replace_once(path: Path, old: str, new: str) -> None:
    """对目标文件执行一次可重复运行的精确文本替换。"""
    # 读取目标文件的 UTF-8 正文。
    text = path.read_text(encoding="utf-8")
    # 新片段已存在时视为补丁已经应用并直接返回。
    if new in text:
        # 保持文件不变以实现幂等。
        return
    # 统计旧片段出现次数，必须恰好为一才能安全修改。
    occurrence_count = text.count(old)
    # 旧片段数量异常时拒绝猜测替换位置。
    if occurrence_count != 1:
        # 抛出带路径和计数的错误，便于 Actions 日志定位载荷漂移。
        raise RuntimeError(f"expected one patch marker in {path}, found {occurrence_count}")
    # 写回只替换一次的新正文。
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# 定义受控模块复制函数，避免在工作流 YAML 中重复文件操作。
def _install(asset_name: str, target: Path) -> None:
    """把一个版本控制资产复制到展开后的载荷目录。"""
    # 定位必须存在的源资产。
    source = ASSET_ROOT / asset_name
    # 源资产缺失时立即失败，避免静默使用旧实现。
    if not source.is_file():
        # 抛出具体源路径。
        raise FileNotFoundError(source)
    # 创建目标父目录，支持新增测试目录或模块目录。
    target.parent.mkdir(parents=True, exist_ok=True)
    # 复制原始字节并保留 UTF-8 源码内容。
    shutil.copyfile(source, target)


# 定义主补丁流程。
def main() -> int:
    """安装缓存提供器、几何纠错和 DeepSeek 决策论文接入。"""
    # 要求唯一位置参数为载荷展开根目录。
    if len(sys.argv) != 2:
        # 参数数量错误时给出最小用法说明。
        raise SystemExit("usage: apply_v4_cache_and_paper_patches.py <payload-root>")
    # 解析并规范化载荷根目录。
    root = Path(sys.argv[1]).resolve()
    # 要求载荷根目录已经由工作流完成哈希校验和解包。
    if not (root / "bridge_agent" / "provider.py").is_file():
        # 缺少核心文件时拒绝在错误目录写入。
        raise FileNotFoundError(root / "bridge_agent" / "provider.py")
    # 安装跨四场景复用同一消息历史的 DeepSeek 提供器。
    _install("cache_aware_provider.py", root / "bridge_agent" / "cache_aware_provider.py")
    # 安装把最终 DeepSeek 决策写入论文的确定性后处理器。
    _install("decision_paper.py", root / "bridge_agent" / "decision_paper.py")
    # 安装只补算两个截断场景的续跑入口，保留上一轮已完成的多孔和裂纹产物。
    _install("resume_v4_suite.py", root / "scripts" / "resume_v4_suite.py")
    # 安装不联网的缓存、重试和区域合同测试。
    _install("test_cache_aware_provider.py", root / "tests" / "test_cache_aware_provider.py")
    # 定位隔离执行单个 case 的入口脚本。
    run_one_case_path = root / "scripts" / "run_one_case.py"
    # 将旧的逐轮新客户端提供器替换为持久历史提供器。
    _replace_once(
        # 指定单 case 入口文件。
        run_one_case_path,
        # 匹配原载荷导入。
        "from bridge_agent.provider import DeepSeekProvider",
        # 写入带中文用途说明的新导入。
        "# 使用跨四场景持久化消息历史的提供器，以复用 DeepSeek 前缀缓存。\nfrom bridge_agent.cache_aware_provider import DeepSeekProvider",
    )
    # 定位原始系统提示词文件。
    provider_path = root / "bridge_agent" / "provider.py"
    # 强化区域多样性与少调用约束，使模型在提交前主动满足历史失败项。
    _replace_once(
        # 指定原始提供器文件。
        provider_path,
        # 匹配原来的区域规则。
        "你可以合并、旋转、扩大或替换候选，也必须显式指定至少一个可稀疏的低重要区域；",
        # 要求至少两种形状并禁止重复已完成 Skill，以减少模型调用。
        "你可以合并、旋转、扩大或替换候选，也必须显式指定至少一个可稀疏的低重要区域；一次区域提交必须使用至少两种不同 shape。不要重复调用已经成功完成的 Skill，required artifact 全部满足后立即 finish；",
    )
    # 定位区域 Skill 实现。
    region_skills_path = root / "bridge_agent" / "region_skills.py"
    # 把几何多样性检查前移到区域提交 Skill，使 DeepSeek 可在下一轮直接纠正。
    _replace_once(
        # 指定区域 Skill 文件。
        region_skills_path,
        # 匹配区域对象刚完成解析的位置。
        "    regions = regions_from_payload(raw)\n    m = ctx.case.model; width = float(m.get(\"width_mm\", m.get(\"length_mm\"))); height = float(m[\"height_mm\"])",
        # 插入形状集合与最少两种形状合同。
        "    regions = regions_from_payload(raw)\n    # 收集本次 AI 分区使用的几何类型，确保不是同一模板的重复平移。\n    shape_kinds = {region.shape for region in regions}\n    # 少于两种形状时立即把可纠正错误返回给下一轮 DeepSeek。\n    if len(shape_kinds) < 2:\n        # 拒绝历史上导致最终 validation 失败的单一形状分区。\n        raise ValueError(\"AI partition must use at least two distinct shapes before optimization\")\n    m = ctx.case.model; width = float(m.get(\"width_mm\", m.get(\"length_mm\"))); height = float(m[\"height_mm\"])",
    )
    # 强化提供给模型的区域证据指令。
    _replace_once(
        # 指定区域 Skill 文件。
        region_skills_path,
        # 匹配原始 AI 指令字面值。
        "\"ai_instruction\": \"根据用户QoI、载荷路径、几何锚点、图像派生候选和量化场图，自主提出任意形状区域。候选不是固定模板；至少提交一个明确可稀疏区域。\",",
        # 增加两种形状的明确要求。
        "\"ai_instruction\": \"根据用户QoI、载荷路径、几何锚点、图像派生候选和量化场图，自主提出任意形状区域。候选不是固定模板；至少提交一个明确可稀疏区域，并在同一次提交中使用至少两种不同shape。\",",
    )
    # 强化 Skill catalog 描述，避免模型只在最终验证时才知道多样性要求。
    _replace_once(
        # 指定区域 Skill 文件。
        region_skills_path,
        # 匹配 catalog 中原始说明。
        "并至少指定一个可稀疏区域。\", \"schema\": _schema({\"regions\":",
        # 在 schema 前的自然语言说明中加入两种形状约束。
        "并至少指定一个可稀疏区域；一次提交必须使用至少两种不同shape。\", \"schema\": _schema({\"regions\":",
    )
    # 定位 Elsevier 论文构建脚本。
    paper_builder_path = root / "scripts" / "build_elsevier_paper.py"
    # 导入决策论文后处理器。
    _replace_once(
        # 指定论文构建脚本。
        paper_builder_path,
        # 匹配现有区域模块导入。
        "from bridge_agent.regions import regions_from_payload, sample_region_boundary",
        # 在相邻位置加入决策注入函数。
        "from bridge_agent.regions import regions_from_payload, sample_region_boundary\n# 将同一 live run 的最终 DeepSeek 决策确定性写入论文，不新增模型调用。\nfrom bridge_agent.decision_paper import inject_deepseek_decisions",
    )
    # 在 Discussion 前放置唯一显式决策插槽。
    _replace_once(
        # 指定论文构建脚本。
        paper_builder_path,
        # 匹配唯一 Discussion 标题。
        "\\section{Discussion}",
        # 插入不会渲染的 LaTeX 注释哨兵。
        "% DEEPSEEK_FINAL_DECISIONS\n\\section{Discussion}",
    )
    # 在固定论文正文生成后立即注入四个最终决策并保存来源收据。
    _replace_once(
        # 指定论文构建脚本。
        paper_builder_path,
        # 匹配唯一正文写入调用。
        "    write_tex(out/'paper.tex',results,receipt,has_plastic)\n    (out/'highlights.txt').write_text(",
        # 注入决策并写 JSON，但不设置额外门禁阻断本轮论文编译。
        "    write_tex(out/'paper.tex',results,receipt,has_plastic)\n    # 从四份 agent trace 提取最终 DeepSeek 决策并写入现有论文正文。\n    decision_receipt=inject_deepseek_decisions(out/'paper.tex',suite,CASE_ORDER,CASE_SHORT)\n    # 保存机器可读来源收据；本轮按用户要求不把它设置为额外阻断门禁。\n    (out/'decision_provenance_receipt.json').write_text(json.dumps(decision_receipt,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')\n    (out/'highlights.txt').write_text(",
    )
    # 打印补丁完成位置，供 Actions 日志快速核查。
    print(f"Applied DeepSeek cache and decision-paper patches to {root}")
    # 返回成功状态。
    return 0


# 仅在脚本直接运行时执行主入口。
if __name__ == "__main__":
    # 把主入口返回码交给操作系统。
    raise SystemExit(main())
