"""把本地 CalculiX 三项反事实复核 trace 渲染为可审阅 PDF。"""  # 说明脚本只读取已冻结证据并生成论文，不触发求解或模型调用。

# 启用延迟注解解析，避免运行时为类型注解引入额外依赖。  # 说明该 future 开关只影响类型注解求值方式。
from __future__ import annotations  # 使用现代类型注解语法并保持 Python 兼容性。

# 导入 HTML 转义工具，防止证据文本中的特殊字符被 ReportLab 当作标签。  # 说明该模块只用于文本安全处理。
import html  # 提供 escape 函数以构造安全 Paragraph 内容。
# 导入 JSON 模块，用于读取机器可读决策路径并输出渲染收据。  # 说明输入输出均采用标准 JSON。
import json  # 解析 trace 文件并序列化最终元数据。
# 导入环境变量访问能力，使用户可显式指定中文字体。  # 说明只读取 PAPER_CJK_FONT，不修改环境。
import os  # 读取可选 PAPER_CJK_FONT 配置。
# 导入 SHA-256 算法，用于把 PDF 与来源 trace 绑定。  # 说明哈希用于证据完整性，不用于安全认证。
import hashlib  # 计算文件内容的 SHA-256 摘要。
# 导入路径类型，以跨平台方式定位仓库、trace 和输出目录。  # 说明所有路径都由脚本位置推导。
from pathlib import Path  # 提供路径拼接、目录创建和文件读取能力。
# 导入通用类型标记，用于 ReportLab 动态对象的类型注解。  # 说明 Any 只用于第三方库对象边界。
from typing import Any  # 标注 canvas、document、style 和 flowable 等动态对象。


# 定义文件哈希函数，把任意工件转换为稳定的 SHA-256 十六进制摘要。  # 说明输入必须是存在的普通文件。
def _sha256_file(path: Path) -> str:  # 接收文件路径并返回大写 SHA-256 字符串。
    # 创建新的 SHA-256 累加器，使用标准 256 位摘要算法。  # 说明算法与 trace 内其余哈希保持一致。
    digest = hashlib.sha256()  # 初始化空摘要状态。
    # 以二进制只读模式打开文件，避免文本换行和编码转换改变字节。  # 说明文件句柄在上下文结束时自动关闭。
    with path.open("rb") as handle:  # 打开待哈希文件并绑定为 handle。
        # 循环读取固定大小分块，以支持未来更大的 PDF 或证据文件。  # 说明循环遇到空字节串时结束。
        while True:  # 持续读取直到到达文件末尾。
            # 每次读取一兆字节，兼顾内存占用与哈希吞吐。  # 数值 1024*1024 表示 1 MiB 分块。
            block = handle.read(1024 * 1024)  # 从当前文件位置读取最多 1 MiB。
            # 当读取结果为空时说明已经到达文件末尾。  # 说明该分支是正常结束路径。
            if not block:  # 检查当前分块是否为空。
                # 退出分块读取循环并进入摘要返回。  # 说明不会跳过任何已读取字节。
                break  # 结束 while 循环。
            # 把当前原始字节块加入摘要状态。  # 说明分块顺序与文件顺序完全一致。
            digest.update(block)  # 更新 SHA-256 累加器。
    # 返回大写十六进制字符串，与仓库现有收据格式一致。  # 输出长度固定为 64 个十六进制字符。
    return digest.hexdigest().upper()  # 完成摘要并规范化大小写。


# 定义中文字体注册函数，优先选择用户配置和常见系统 CJK 字体。  # 说明无法获得中文字体时显式失败。
def _register_pdf_font() -> str:  # 返回已在 ReportLab 注册的字体名称。
    # 延迟导入 ReportLab 字体注册器，使导入本脚本时不会立即要求 PDF 依赖。  # 说明渲染调用时才加载第三方库。
    from reportlab.pdfbase import pdfmetrics  # 提供全局字体注册表。
    # 导入内置中文 CID 字体类型，作为本地字体文件不可用时的后备。  # 说明后备仍能渲染简体中文。
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # 提供 STSong-Light 支持。
    # 导入 TrueType 字体类型，以注册 TTC 或 TTF 中文字体。  # 说明 subfontIndex 选择集合中的第一个字体。
    from reportlab.pdfbase.ttfonts import TTFont  # 提供 TrueType/TTC 字体加载器。
    # 读取用户可选字体路径；未设置时得到 None。  # 字符串 PAPER_CJK_FONT 是仓库已有 PDF 约定。
    configured_font = os.environ.get("PAPER_CJK_FONT")  # 获取显式 CJK 字体覆盖值。
    # 建立候选路径列表，顺序体现用户配置、Linux CI、Windows 本机的优先级。  # 每个字面路径都指向常见中文字体安装位置。
    candidates = [Path(configured_font) if configured_font else None, Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"), Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simsun.ttc")]  # 汇集五个按优先级排列的字体候选。
    # 逐个检查候选，找到第一个存在且可注册的字体。  # 说明失败候选不会终止整个注册过程。
    for candidate in candidates:  # 遍历显式路径和四个常见系统路径。
        # 跳过空路径与不存在的路径，防止传入无效字体文件。  # 说明 is_file 同时排除目录。
        if candidate is None or not candidate.is_file():  # 判断当前候选是否可作为文件读取。
            # 继续检查下一个候选，不把缺少某个系统字体当作错误。  # 说明这是预期兼容路径。
            continue  # 进入下一轮候选检查。
        # 捕获字体集合兼容性差异，以便尝试后续候选。  # 说明只包围单次字体注册。
        try:  # 尝试把当前候选注册为统一名称。
            # 使用集合第一个子字体注册 PaperCJK，确保正文和页脚引用同一字体名。  # 数值 0 表示 TTC 的第一个子字体。
            pdfmetrics.registerFont(TTFont("PaperCJK", str(candidate), subfontIndex=0))  # 注册当前 CJK 字体文件。
            # 成功后立即返回统一字体名，避免重复注册。  # 字符串 PaperCJK 是脚本内部固定名称。
            return "PaperCJK"  # 告知渲染器使用已注册字体。
        # 任一字体解析异常都只淘汰当前候选。  # 说明下一候选仍可能正常工作。
        except Exception:  # 捕获 ReportLab 对当前字体的解析或注册错误。
            # 跳过不兼容字体并继续候选循环。  # 说明不会静默回退到不支持中文的 Helvetica。
            continue  # 检查下一字体候选。
    # 当系统字体均不可用时尝试 ReportLab 标准简体中文 CID 字体。  # 说明 STSong-Light 不依赖显式本地路径。
    try:  # 尝试注册标准 CID 后备字体。
        # 注册简体中文 STSong-Light，确保没有本地 CJK 文件时仍能阅读正文。  # 字体名称是 ReportLab 标准标识。
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))  # 将标准 CID 字体加入注册表。
        # 返回后备字体名供所有样式统一使用。  # 说明不会混用西文字体。
        return "STSong-Light"  # 完成后备字体选择。
    # 如果连 CID 字体也无法使用，则阻止生成不可读 PDF。  # 说明异常会保留原始原因。
    except Exception as exception:  # 捕获标准字体注册失败。
        # 抛出清晰错误，避免把方块字或空白论文当作成功工件。  # 中文消息便于本地排障。
        raise RuntimeError("没有可用的中文 PDF 字体，已停止生成不可读论文。") from exception  # 将字体问题升级为渲染失败。


# 定义文本规范化函数，转义 XML 并把特殊连字符替换成 ASCII 连字符。  # 说明遵守 PDF 工件的字符兼容要求。
def _safe_text(value: Any) -> str:  # 接收任意值并返回 Paragraph 可接受的安全字符串。
    # 把 None 规范为空串，其余值转成文本。  # 说明不会把 None 字面量写进论文。
    plain = "" if value is None else str(value)  # 生成待清理的基础字符串。
    # 替换非换行连字符、短横线和长横线，避免嵌入字体缺字。  # 三个 Unicode 字符统一映射为 ASCII "-"。
    normalized = plain.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")  # 完成连字符兼容化。
    # 转义 XML 特殊字符，并把普通换行保留为 ReportLab 段落换行标签。  # 参数 quote=True 同时处理引号。
    return html.escape(normalized, quote=True).replace("\n", "<br/>")  # 返回安全且保留换行的富文本片段。


# 定义紧凑数值格式函数，使跨数量级的残差和修正量能放入表格。  # 说明非数字按普通文本呈现。
def _compact_number(value: Any) -> str:  # 接收数字或其他值并返回短字符串。
    # 对整数和浮点数使用四位有效数字科学/普通混合格式。  # 格式 .4g 在小数和指数之间自动选择。
    if isinstance(value, (int, float)) and not isinstance(value, bool):  # 排除布尔值，因为 bool 是 int 子类。
        # 返回四位有效数字，足够比较量级且避免表格溢出。  # 说明该格式不改变 trace 中原始值。
        return f"{value:.4g}"  # 生成仅用于展示的紧凑数值。
    # 其他类型直接转为文本，None 显示为短横线。  # 说明短横线表示无可用数值。
    return "-" if value is None else str(value)  # 返回缺失标记或原文本。


# 定义页脚绘制函数，写入短标题与页码。  # 说明每页都由 document.build 回调调用。
def _draw_footer(canvas: Any, document: Any, font_name: str) -> None:  # 接收 ReportLab 画布、文档状态和字体名。
    # 保存当前绘图状态，防止页脚设置影响正文。  # 说明函数结束前会恢复状态。
    canvas.saveState()  # 压入当前画布状态。
    # 使用七点五号中文字体，保持页脚清晰且不抢正文。  # 数值 7.5 的单位是 PDF point。
    canvas.setFont(font_name, 7.5)  # 设置页脚字体和字号。
    # 在左下角写入工件短标题。  # y=18 point 与正文底边距保持隔离。
    canvas.drawString(document.leftMargin, 18, "Local CalculiX counterfactual follow-up")  # 绘制固定英文短标题以降低字体度量风险。
    # 在右下角使用固定起点写页码。  # 预留 22 毫米宽度足以容纳三位页码。
    page_x = document.pagesize[0] - document.rightMargin - (22.0 * 72.0 / 25.4)  # 把 22 毫米换算为 PDF point 并计算横坐标。
    # 绘制中文页码，便于打印审阅。  # document.page 从一开始递增。
    canvas.drawString(page_x, 18, f"第 {document.page} 页")  # 写入当前页号。
    # 恢复进入函数前的画布状态。  # 说明正文的颜色和字体不会受页脚污染。
    canvas.restoreState()  # 弹出保存的画布状态。


# 定义表格构造函数，统一标题行、网格、斑马纹和字号。  # 说明列宽由调用方按页面内容指定。
def _make_table(rows: list[list[Any]], widths: list[Any], font_name: str, colors: Any, Table: Any, TableStyle: Any) -> Any:  # 返回已设置样式的 ReportLab 表格。
    # 创建表格并固定第一行为跨页重复表头。  # repeatRows=1 表示仅重复标题行。
    table = Table(rows, colWidths=widths, repeatRows=1)  # 按调用方列宽实例化表格。
    # 应用浅蓝表头、灰色网格和交替行背景，提高密集数值的可读性。  # 所有颜色均使用固定十六进制设计值。
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEFF5")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#183B56")), ("FONTNAME", (0, 0), (-1, -1), font_name), ("FONTSIZE", (0, 0), (-1, -1), 6.4), ("LEADING", (0, 0), (-1, -1), 8.0), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#90A4AE")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]), ("TOPPADDING", (0, 0), (-1, -1), 3.0), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0)]))  # 一次性设置完整表格样式。
    # 返回已完成样式配置的表格对象。  # 说明调用方负责把它加入 story。
    return table  # 提供给页面流式布局。


# 定义主渲染函数，从单一 trace 生成三页本地复核论文。  # 说明不会发起网络、模型或求解器调用。
def _render(trace_path: Path, output_path: Path) -> dict[str, Any]:  # 接收 trace 路径和目标 PDF 路径并返回收据。
    # 导入 ReportLab 颜色模块，用于论文调色。  # 说明第三方依赖只在实际渲染时加载。
    from reportlab.lib import colors  # 提供 HexColor、white 等颜色对象。
    # 导入 A4 页面常量，确保输出适合打印和审阅。  # 说明 A4 尺寸为 210x297 毫米。
    from reportlab.lib.pagesizes import A4  # 提供标准 A4 点坐标尺寸。
    # 导入段落样式类型，用于建立标题、正文和说明文字。  # 说明样式从默认样式继承。
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # 提供样式构造和默认样式表。
    # 导入毫米单位，便于用物理尺寸控制边距和表格列宽。  # 说明一毫米会转换为 PDF point。
    from reportlab.lib.units import mm  # 提供 mm 缩放常量。
    # 导入分页文档与常用 flowable 组件。  # 说明这些对象组成 story 顺序流。
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # 提供段落、表格、间距和分页能力。
    # 以 UTF-8 读取机器 trace，确保中文说明不受系统默认编码影响。  # 说明输入必须是有效 JSON。
    trace = json.loads(trace_path.read_text(encoding="utf-8"))  # 解析完整决策对象。
    # 确保输出父目录存在；parents=True 创建缺失层级，exist_ok=True 允许重复渲染。  # 说明只影响固定 output/pdf 目录。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建 PDF 输出目录。
    # 注册可用的中文字体并获得统一字体名。  # 说明所有文字和表格均使用该字体。
    font_name = _register_pdf_font()  # 完成 CJK 字体选择。
    # 获取 ReportLab 默认样式表作为自定义样式的父级。  # 说明不会直接使用默认西文字体。
    defaults = getSampleStyleSheet()  # 创建基础样式集合。
    # 定义深蓝主标题样式，十八点字号用于论文封面。  # leading=23 point 保证中文行距。
    title_style = ParagraphStyle("LocalTitle", parent=defaults["Title"], fontName=font_name, fontSize=18, leading=23, textColor=colors.HexColor("#183B56"), spaceAfter=9)  # 配置主标题视觉层级。
    # 定义一级标题样式，十三点字号配合蓝绿色。  # spaceBefore 和 spaceAfter 控制章节间距。
    heading_style = ParagraphStyle("LocalHeading", parent=defaults["Heading1"], fontName=font_name, fontSize=12.5, leading=17, textColor=colors.HexColor("#0F5C78"), spaceBefore=8, spaceAfter=5)  # 配置章节标题。
    # 定义二级标题样式，用于结论与证据短节。  # 十点五字号保持紧凑。
    subheading_style = ParagraphStyle("LocalSubheading", parent=defaults["Heading2"], fontName=font_name, fontSize=10.5, leading=14, textColor=colors.HexColor("#2F6F89"), spaceBefore=6, spaceAfter=3)  # 配置小节标题。
    # 定义九点二号正文与十四点行距。  # 颜色使用深灰以降低长文视觉疲劳。
    body_style = ParagraphStyle("LocalBody", parent=defaults["BodyText"], fontName=font_name, fontSize=9.2, leading=14, textColor=colors.HexColor("#263238"), spaceAfter=5)  # 配置论文正文。
    # 定义七点八号小字，用于来源、哈希和边界说明。  # 说明小字仍保持十一点行距。
    small_style = ParagraphStyle("LocalSmall", parent=body_style, fontSize=7.8, leading=11, textColor=colors.HexColor("#546E7A"), spaceAfter=4)  # 配置紧凑审计文本。
    # 创建 A4 文档，左右十六毫米、顶部十六毫米、底部二十二毫米。  # 底部额外空间用于页脚。
    document = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=22 * mm, title="CalculiX local counterfactual follow-up", author="Local engineering decision record")  # 初始化可分页 PDF 文档。
    # 初始化按顺序排版的 flowable 列表。  # 说明所有页面内容均追加到 story。
    story: list[Any] = []  # 保存标题、段落、表格、间距和分页符。
    # 定义局部段落助手，统一安全转义和追加逻辑。  # 默认采用正文样式。
    def add_paragraph(text_value: Any, style: Any = body_style) -> None:  # 接收任意文本和值得使用的 ParagraphStyle。
        # 把规范化后的安全文本创建为段落并加入 story。  # 说明 _safe_text 已处理 XML 和特殊连字符。
        story.append(Paragraph(_safe_text(text_value), style))  # 追加一个可自动换行的文本块。
    # 按 ID 读取三个决策，避免依赖列表顺序。  # 说明 trace 必须包含 D1、D2、D3。
    decisions = {item["id"]: item for item in trace["decisions"]}  # 建立决策 ID 到对象的映射。
    # 获取集成结论对象，用于封面摘要与末页边界。  # 说明该状态不会被渲染器重新计算。
    integrated = trace["integrated_conclusion"]  # 引用 trace 中已冻结的合并判断。
    # 写入论文主标题。  # 文本明确说明这是本地反事实复核。
    add_paragraph("CalculiX 过盈接触：三项本地反事实复核", title_style)  # 生成第一页标题。
    # 写入副标题与运行约束，强调零模型调用。  # 说明该行使用小字降低视觉权重。
    add_paragraph("Local engineer decision record | CalculiX 2.22 / SPOOLES | DeepSeek 调用 0 | 训练 0 | 优化 0", small_style)  # 写入运行模式摘要。
    # 写入摘要标题。  # 说明从此开始论文正文。
    add_paragraph("摘要", heading_style)  # 生成摘要章节标题。
    # 写入一段核心结论，先给结果再给边界。  # 字符串只概括 trace 中已有结论。
    add_paragraph("三项真实本地求解排除了两个简单解释：减小初始增量不能独立修复，penalty 也不是越大越稳定。接触 REMOVE->ADD 本身没有复现停滞；只有在 ADD 同时释放约束时，才出现明显 cutback 与迭代放大。原问题被收窄，但圆柱 surface-to-surface 模型的近 1e-30 不推进循环仍未复现。")  # 写入结果导向摘要。
    # 写入集成状态标题。  # 说明状态直接来自机器 trace。
    add_paragraph("集成判断", subheading_style)  # 生成状态小节。
    # 写入当前状态及允许结论。  # 字符串 narrowed_unresolved 体现未冒充解决。
    add_paragraph(f"状态：{integrated['state']}。代表模型支持有限 penalty 稳定窗口，也支持接触状态与约束路径同步变化是强收敛放大器；它不确认原大型圆柱模型的根因或修复。")  # 写入合并判断正文。
    # 写入 D1 标题。  # 说明第一页继续展示增量反事实。
    add_paragraph("D1 较小初始增量", heading_style)  # 创建第一个实验章节。
    # 初始化 D1 表头，列出控制字段和关键数值。  # 短标签用于适配页面宽度。
    d1_rows: list[list[Any]] = [["mid-z", "inc", "SOLVE-ID", "完成", "no-conv", "尝试", "最大残差", "最大修正"]]  # 建立增量实验表格标题行。
    # 遍历四个 D1 案例并生成显示行。  # 说明顺序来自 trace 的搜索/留出配对。
    for case in decisions["D1"]["cases"]:  # 逐个读取增量对照案例。
        # 把原始值压缩为适合表格的字符串。  # 布尔完成状态转为中文。
        d1_rows.append([_compact_number(case["mid_z"]), _compact_number(case["initial_increment"]), case["solve_id"], "是" if case["completed"] else "否", str(case["no_convergence_count"]), str(case["increment_attempt_count"]), _compact_number(case["max_residual_force"]), _compact_number(case["max_absolute_displacement_correction"])])  # 追加单个案例行。
    # 创建 D1 表格，所有列宽合计不超过 178 毫米正文宽度。  # 各数值是列的物理宽度。
    story.append(_make_table(d1_rows, [12 * mm, 13 * mm, 34 * mm, 10 * mm, 16 * mm, 12 * mm, 24 * mm, 24 * mm], font_name, colors, Table, TableStyle))  # 追加增量对照表。
    # 在表格后加入三毫米垂直间距。  # 说明防止结论紧贴网格。
    story.append(Spacer(1, 3 * mm))  # 插入表后留白。
    # 写入 D1 改判。  # 文本来自 updated_conclusion。
    add_paragraph(f"改判：{decisions['D1']['updated_conclusion']['text_zh']}")  # 解释四个失败算例如何反证独立修复。
    # 强制 D2 从新页开始，避免八行表格在页尾拆分。  # 说明分页是可读性设计。
    story.append(PageBreak())  # 结束第一页。
    # 写入 D2 标题。  # 第二页集中呈现 penalty 稳定窗口。
    add_paragraph("D2 penalty 稳定窗口", heading_style)  # 创建第二个实验章节。
    # 写入单因素和原 deck 边界。  # 说明自动值由 CalculiX 报告为 1.05e7。
    add_paragraph("同一 mid-z 内只改变 pressure-overclosure penalty：自动值（报告 K=1.05e7）、69000、1e5、1e6。原附件已使用 69000，因此代表模型中的成功只能是机制证据或候选 regularization。")  # 描述实验控制与不可外推边界。
    # 初始化 D2 表头，覆盖八个有效求解。  # 说明 RFz 仅在完成算例中存在。
    d2_rows: list[list[Any]] = [["mid-z", "penalty", "SOLVE-ID", "完成", "no-conv", "尝试", "最大修正", "底面 RFz"]]  # 建立 penalty 实验表格标题行。
    # 遍历八个有效案例，排除 trace 中已单列的无效早期调用。  # 说明只使用 decisions.D2.cases。
    for case in decisions["D2"]["cases"]:  # 逐个读取有效 penalty 案例。
        # 将 automatic 和数字 penalty 均转换为短文本。  # 说明展示值不改变原始 trace。
        penalty_label = "auto" if case["penalty"] == "automatic" else _compact_number(case["penalty"])  # 生成 penalty 列标签。
        # 追加当前案例关键数值；缺失反力显示短横线。  # 说明完成状态使用中文是/否。
        d2_rows.append([_compact_number(case["mid_z"]), penalty_label, case["solve_id"], "是" if case["completed"] else "否", str(case["no_convergence_count"]), str(case["increment_attempt_count"]), _compact_number(case["max_absolute_displacement_correction"]), _compact_number(case.get("bottom_reaction_z_sum"))])  # 追加单个 penalty 案例行。
    # 创建并追加八行 D2 表格。  # 列宽总量控制在 A4 正文区域内。
    story.append(_make_table(d2_rows, [12 * mm, 17 * mm, 34 * mm, 10 * mm, 16 * mm, 12 * mm, 24 * mm, 23 * mm], font_name, colors, Table, TableStyle))  # 追加 penalty 对照表。
    # 加入三毫米表后间距。  # 说明下一段不会贴住表格底边。
    story.append(Spacer(1, 3 * mm))  # 插入垂直留白。
    # 写入稳定窗口结论。  # 文本直接来自 trace 的 updated_conclusion。
    add_paragraph(f"改判：{decisions['D2']['updated_conclusion']['text_zh']}")  # 解释非单调稳定性与原 deck 边界。
    # 写入力平衡校核。  # 数值 2.0e-6 来自所有完成 penalty 算例的最大绝对不平衡量。
    add_paragraph("响应校核：完成算例的底面 z 向反力和约为 25，全模型 z 向不平衡量绝对值不超过 2.0e-6。一次无效的带连字符 contact-type 调用已隔离，不计入八个有效样本。", small_style)  # 记录物理一致性和排除尝试。
    # 强制 D3 从新页开始，使状态路径表与最终边界保持完整。  # 说明第三页用于激活实验和证据收据。
    story.append(PageBreak())  # 结束第二页。
    # 写入 D3 标题。  # 第三页展示接触生命周期与约束耦合。
    add_paragraph("D3 接触重新激活与约束路径", heading_style)  # 创建第三个实验章节。
    # 写入 A/B/C 的唯一差异。  # 使用 ASCII 箭头以避免特殊连字符兼容问题。
    add_paragraph("A：接触两步持续激活且上块 U3 固定；B：REMOVE->ADD 且 U3 始终固定；C：REMOVE->ADD，并在 ADD 的同一步释放 U3。A 对 B 只改变接触生命周期，B 对 C 只改变重新激活时的约束路径。")  # 描述顺序单因素设计。
    # 初始化 D3 表头，包含增量推进和近零循环指标。  # 说明三行对应 A/B/C。
    d3_rows: list[list[Any]] = [["案例", "完成", "尝试", "失败", "no-conv", "最大残差", "最大修正", "1e-30 停滞"]]  # 建立激活实验表格标题行。
    # 遍历三条状态路径并形成显示行。  # 说明 unsuccessful_attempts 是 cutback 风险指标。
    for case in decisions["D3"]["cases"]:  # 逐个读取 A、B、C 案例。
        # 追加完成状态、尝试数、失败尝试和数值极值。  # 说明最后一列明确标记是否复现目标循环。
        d3_rows.append([case["case_id"], "是" if case["completed"] else "否", str(case["increment_attempts_total"]), str(case["unsuccessful_attempts"]), str(case["no_convergence_events"]), _compact_number(case["max_residual_force"]), _compact_number(case["max_displacement_correction"]), "是" if case["one_e_minus_30_idle_loop"] else "否"])  # 追加单个状态路径行。
    # 创建并追加 D3 表格。  # 列宽为三行结果留出足够数值空间。
    story.append(_make_table(d3_rows, [12 * mm, 11 * mm, 12 * mm, 12 * mm, 17 * mm, 26 * mm, 25 * mm, 24 * mm], font_name, colors, Table, TableStyle))  # 追加接触激活对照表。
    # 在表格后加入三毫米间距。  # 说明留白用于区分数据和解释。
    story.append(Spacer(1, 3 * mm))  # 插入表后留白。
    # 写入 B 的零状态解释，防止把零量值误判为卡死。  # 说明每个增量仍然推进。
    add_paragraph("B 第一步出现精确零残差和零修正，但每个增量都在第二次迭代收敛并继续前进，因此是无载荷平凡平衡，不是不推进循环。")  # 解释零量值与停滞的区别。
    # 写入 D3 改判。  # 文本直接来自 trace。
    add_paragraph(f"改判：{decisions['D3']['updated_conclusion']['text_zh']}")  # 总结约束路径放大与未复现边界。
    # 写入后续实验标题。  # 说明这是根据三项结果形成的最高价值下一步。
    add_paragraph("下一项最高价值实验", subheading_style)  # 创建后续工作小节。
    # 写入 trace 中冻结的下一项实验，不由渲染器自行扩展。  # 说明仍限定在本地可执行缩减模型。
    add_paragraph(integrated["highest_value_next_experiment_zh"])  # 呈现圆柱 surface-to-surface 缩减实验建议。
    # 写入证据收据标题。  # 说明末尾给出可核对工件。
    add_paragraph("证据收据与限制", subheading_style)  # 创建审计小节。
    # 计算来源 trace 哈希，使 PDF 内容与机器记录绑定。  # 说明哈希在渲染前对只读输入计算。
    trace_sha256 = _sha256_file(trace_path)  # 获取 trace 的大写 SHA-256。
    # 写入求解器、调用数、trace 哈希和原附件哈希。  # 说明这些值均来自 trace 或现场计算。
    add_paragraph(f"CalculiX 2.22 / SPOOLES；ccx.exe SHA-256：{trace['solver']['executable_sha256']}；DeepSeek 调用：{trace['record_scope']['deepseek_calls']}；trace SHA-256：{trace_sha256}；原附件 SHA-256：{trace['record_scope']['original_attachment_sha256']}。", small_style)  # 冻结关键来源标识。
    # 写入不可外推边界。  # 说明当前方块 node-to-surface 与原圆柱 Pardiso deck 不同。
    add_paragraph("限制：本地代表模型不是原圆柱 surface-to-surface 几何，后端不是原 deck 请求的 Pardiso，原约 1160 万自由度模型未被冒充为本地完成。结论只用于机制收窄和下一步实验设计。", small_style)  # 明确论文允许使用范围。
    # 构造统一页脚回调，确保所有页面使用同一字体。  # lambda 把当前 font_name 绑定给绘制函数。
    footer = lambda canvas, doc: _draw_footer(canvas, doc, font_name)  # 创建 ReportLab 页面回调。
    # 执行流式排版并写入最终 PDF。  # onFirstPage 和 onLaterPages 均使用同一页脚。
    document.build(story, onFirstPage=footer, onLaterPages=footer)  # 生成磁盘上的 PDF 工件。
    # 返回包含路径、字体、大小和双哈希的渲染收据。  # 说明此对象将打印到 stdout 供调用方记录。
    return {"path": str(output_path), "font_name": font_name, "size_bytes": output_path.stat().st_size, "pdf_sha256": _sha256_file(output_path), "trace_sha256": trace_sha256, "deepseek_calls": trace["record_scope"]["deepseek_calls"]}  # 汇总渲染结果元数据。


# 定义脚本入口，固定使用仓库内本地复核 trace 和 output/pdf 目录。  # 说明无命令行参数可意外触发其他路径。
def main() -> None:  # 执行路径解析、渲染和收据打印且不返回业务值。
    # 从当前脚本向上两级得到仓库根目录。  # parents[1] 对应 scripts 的父目录。
    repository_root = Path(__file__).resolve().parents[1]  # 解析绝对仓库根路径。
    # 拼接本地复核工件目录。  # 字面路径与 PR 中证据目录保持一致。
    followup_root = repository_root / "experiments" / "results" / "calculix_interference_2747" / "local_followup"  # 定位三项本地复核根目录。
    # 指定唯一机器 trace 输入文件。  # 说明渲染不读取会变化的 console 内容。
    trace_path = followup_root / "local_engineer_trace.json"  # 定位已冻结决策路径。
    # 指定 PDF 技能要求的 output/pdf 输出位置。  # 文件名明确区分此前 DeepSeek 论文。
    output_path = followup_root / "output" / "pdf" / "local_calculix_counterfactual_followup.pdf"  # 定位最终论文 PDF。
    # 调用主渲染函数并获得完整收据。  # 说明该调用只读取 trace 并写 PDF。
    receipt = _render(trace_path, output_path)  # 生成论文与元数据。
    # 以 UTF-8 友好的 JSON 打印收据，便于人工和自动流程核对。  # ensure_ascii=False 保留中文路径。
    print(json.dumps(receipt, ensure_ascii=False, indent=2))  # 把渲染结果输出到标准输出。


# 仅在直接运行脚本时进入 main，导入模块不会生成文件。  # 说明这是标准 Python 入口保护。
if __name__ == "__main__":  # 判断当前模块是否为程序入口。
    # 执行固定本地复核 PDF 渲染流程。  # 说明不会启动求解器或模型客户端。
    main()  # 完成论文生成并打印收据。
