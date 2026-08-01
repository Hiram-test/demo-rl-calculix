# V5 轻量前端

本目录提供一个无依赖、无网络请求的静态输入与证据查看原型。它可以直接由浏览器打开，也可以作为 GitHub Actions artifact 下载后查看。

## 已实现

- 保存用户本人填写的自然语言有限元问题。
- 为当前页面加载生成一个带 frontend-lite 前缀的草案任务标识。
- 允许选择任意格式、任意扩展名的多个 IMP 输入材料。
- 只记录文件名、字节数、浏览器 MIME 类型和最后修改时间等元数据。
- 允许用户显式登记 missing facts，并按现有 ProblemManifest 结构导出 JSON 草案。
- 允许本地导入数组型 agent_trace.json。
- 允许本地导入对象型 validation receipt 候选，但明确标记为结构未验证。
- 只显示证据文件明确提供的 decision、execution observation、计数、错误和用途边界。

## 明确未实现

- 不读取、上传或嵌入 IMP 文件内容。
- 不假设 IMP 的格式、扩展名、字段、大小限制或解析规则。
- 不从自然语言中推断几何、材料、载荷、边界条件、QoI、容差或算法配置。
- 不连接 API、数据库、任务队列或事件流。
- 不调用 DeepSeek、Gmsh 或 CalculiX。
- 不生成网格、求解结果、运行时间或工程结论。
- 不使用 localStorage、sessionStorage 或其他持久化能力。
- 不把 validation receipt 候选当作已通过 schema 验证的数据。

## 导出的 ProblemManifest 字段

manifest_version 表示当前仓库已有合同版本，本原型固定使用 1.0。

task_id 是浏览器生成的草案标识，不代表后端已经创建任务。

user_goal 原样保存用户当前输入的自然语言问题。

input_files 只包含浏览器可见元数据。parse_status 固定为 not_attempted_format_undefined，表示格式未冻结所以未尝试解析。validation_status 固定为 not_attempted_contract_undefined，表示校验合同未冻结所以未执行校验。content_included 固定为 false，sha256 为 null，因为本页面不读取文件内容。

facts 保持为空对象，因为静态页面不解释工程事实。

missing_facts 只来自用户通过页面显式登记的 path、reason、question 和 acceptable_sources。

algorithm_configuration 保持为空对象，因为静态页面不选择优化或求解配置。

observations 逐项记录本页面未连接后端、未解析材料、未调用模型和未运行求解器的事实。

provenance_report 是非阻断摘要，用于说明本草案没有旧 fixture、没有代理派生事实，也没有待确认假设。

## 证据查看规则

数组根节点按现有 TraceRecorder 的 decision 和 execution_observation 事件读取。decision.action 只显示为决策动作，不会被冒充为当前阶段。

对象根节点只作为 receipt 候选。页面只读取明确出现的 stage、status、decision、action、summary_file、scenario_count、scenarios、totals、counters、errors、can_use、cannot_use 和 evidence_refs 等字段；缺失字段统一显示为源文件未提供，页面未推断。

证据 JSON 超过 10 MiB 时不会在浏览器中读取。这个限制只用于保护本地页面内存，不是 IMP 格式或工程文件大小规则。

## 安全与可访问性

所有动态内容通过 textContent 和原生 DOM 节点渲染，不使用 innerHTML。页面没有网络请求、外部脚本、外部字体或图片。全部控件使用原生 label、button、details 和 summary，并提供明显的键盘焦点样式。

## 后续边界

真正的文件上传、格式解析、来源哈希、后端任务创建、求解执行、全阶段计时、预算 ledger 和录屏必须在后续已批准切片中实现。本原型不得作为这些能力已经存在的证据。
