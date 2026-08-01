/* 启用严格模式以减少静默变量和赋值错误。 */
"use strict"; // 使用浏览器与 Node.js 都支持的严格模式。

/* 将证据 JSON 的浏览器内读取上限固定为十兆字节，仅用于内存保护而非 IMP 格式限制。 */
const EVIDENCE_PREVIEW_LIMIT_BYTES = 10 * 1024 * 1024; // 十乘一千零二十四乘一千零二十四字节等于十 MiB。
/* 保存本页面会用到的永久边界文本，避免不同区域出现相互矛盾的描述。 */
const NOT_PROVIDED_TEXT = "源文件未提供，页面未推断。"; // 该字符串明确区分缺失字段与零值。
/* 保存浏览器内的当前输入状态；这些数据不会持久化或发送到网络。 */
const state = { // 创建唯一页面状态对象。
  taskId: "", // 保存本次页面加载生成的任务标识。
  materialFiles: [], // 保存用户当前选择的 File 对象，仅用于读取元数据。
  missingFacts: [], // 保存用户显式登记的缺失事实记录。
  importedPayload: null, // 保存当前成功解析的真实 JSON，清除或失败时恢复为空。
  importedName: "", // 保存当前证据文件名以便向用户说明来源。
}; // 结束页面状态对象定义。

/* 集中保存 DOM 引用，避免重复查询并明确每个控件的用途。 */
const dom = { // 创建页面元素引用对象。
  form: document.querySelector("#task-form"), // 引用任务输入表单。
  taskId: document.querySelector("#task-id"), // 引用只读任务标识输入框。
  userGoal: document.querySelector("#user-goal"), // 引用自然语言问题文本区。
  impFiles: document.querySelector("#imp-files"), // 引用 IMP 多文件选择器。
  clearFiles: document.querySelector("#clear-files"), // 引用清除材料按钮。
  fileList: document.querySelector("#file-list"), // 引用材料元数据列表容器。
  missingPath: document.querySelector("#missing-path"), // 引用缺失事实路径输入框。
  missingReason: document.querySelector("#missing-reason"), // 引用缺失原因输入框。
  missingQuestion: document.querySelector("#missing-question"), // 引用确认问题输入框。
  missingSources: document.querySelector("#missing-sources"), // 引用可接受来源文本区。
  addMissingFact: document.querySelector("#add-missing-fact"), // 引用添加缺失事实按钮。
  missingList: document.querySelector("#missing-list"), // 引用已登记缺失事实列表。
  previewManifest: document.querySelector("#preview-manifest"), // 引用刷新草案预览按钮。
  manifestPreview: document.querySelector("#manifest-preview"), // 引用草案 JSON 预览区。
  formMessage: document.querySelector("#form-message"), // 引用输入区动态消息。
  evidenceFile: document.querySelector("#evidence-file"), // 引用真实证据文件选择器。
  clearEvidence: document.querySelector("#clear-evidence"), // 引用清除证据按钮。
  evidenceMessage: document.querySelector("#evidence-message"), // 引用证据读取消息。
  evidencePreview: document.querySelector("#evidence-preview"), // 引用原始证据 JSON 预览区。
  stageValue: document.querySelector("#stage-value"), // 引用阶段展示字段。
  decisionValue: document.querySelector("#decision-value"), // 引用决策展示字段。
  evidenceValues: document.querySelector("#evidence-values"), // 引用执行与证据列表。
  budgetValues: document.querySelector("#budget-values"), // 引用预算与计数定义列表。
  errorValues: document.querySelector("#error-values"), // 引用错误与问题列表。
  boundaryValues: document.querySelector("#boundary-values"), // 引用用途边界列表。
}; // 结束页面元素引用对象定义。

/**
 * 生成只用于当前输入草案的任务标识。
 * @returns {string} 返回浏览器随机 UUID，旧浏览器则返回带毫秒时间戳的非物理标识。
 */
function createTaskId() { // 定义任务标识生成函数。
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") { // 仅在浏览器提供标准随机 UUID 时使用该能力。
    return "frontend-lite-" + crypto.randomUUID(); // 加入前缀以避免把标识误认成后端任务。
  } // 结束随机 UUID 能力判断。
  return "frontend-lite-" + Date.now().toString(36); // 使用当前毫秒时间的三十六进制文本作为兼容回退。
} // 结束任务标识生成函数。

/**
 * 将字节数量格式化为只用于界面显示的文本。
 * @param {number} bytes 浏览器从 File.size 提供的非负字节数量。
 * @returns {string} 返回带 B、KiB 或 MiB 的可读大小文本。
 */
function formatBytes(bytes) { // 定义文件大小格式化函数。
  if (bytes < 1024) { // 小于一千零二十四字节时保留字节单位。
    return bytes + " B"; // 返回整数原值和 B 单位。
  } // 结束字节单位判断。
  if (bytes < 1024 * 1024) { // 小于一 MiB 时使用 KiB 单位。
    return (bytes / 1024).toFixed(1) + " KiB"; // 除以一千零二十四并保留一位小数。
  } // 结束 KiB 单位判断。
  return (bytes / (1024 * 1024)).toFixed(1) + " MiB"; // 除以一千零二十四的平方并保留一位小数。
} // 结束文件大小格式化函数。

/**
 * 更新动态消息并控制其视觉语义。
 * @param {HTMLElement} element 要写入消息的页面元素。
 * @param {string} text 要显示给用户的消息文本。
 * @param {"neutral"|"error"|"success"} tone 仅表示本地交互结果的颜色类型。
 * @returns {void} 本函数只更新页面，不返回数据。
 */
function setMessage(element, text, tone) { // 定义统一消息更新函数。
  element.textContent = text; // 使用 textContent 防止输入被解释为 HTML。
  element.className = "message"; // 先恢复中性消息基础类。
  if (tone === "error") { // 仅在明确本地输入或解析失败时使用错误样式。
    element.classList.add("error"); // 添加错误颜色类。
  } // 结束错误样式判断。
  if (tone === "success") { // 仅在本地添加、预览或下载完成时使用完成样式。
    element.classList.add("success"); // 添加本地操作完成颜色类。
  } // 结束完成样式判断。
} // 结束统一消息更新函数。

/**
 * 把用户选择的 File 转换为不包含内容的来源记录。
 * @param {File} file 浏览器文件选择器返回的文件对象。
 * @returns {Object} 返回可序列化的文件元数据与明确的未解析状态。
 */
function describeImpFile(file) { // 定义 IMP 文件元数据转换函数。
  const modifiedUtc = file.lastModified > 0 ? new Date(file.lastModified).toISOString() : ""; // 仅在浏览器提供正数毫秒时间时转换为 UTC 文本。
  return { // 返回与当前未知 IMP 格式无关的来源记录。
    path: file.name, // 使用用户所选文件名作为当前浏览器来源路径。
    role: "user_supplied_input_material", // 标记该文件由用户作为当前输入材料选择。
    source: "browser_file_selection", // 明确来源是本地浏览器文件选择。
    size_bytes: file.size, // 保存浏览器报告的原始字节数。
    media_type: file.type, // 原样保存浏览器 MIME 类型，空字符串保持为空。
    last_modified_ms: file.lastModified, // 保存浏览器提供的最后修改毫秒时间戳。
    last_modified_utc: modifiedUtc, // 保存可读 UTC 时间，缺失时保持空字符串。
    parse_status: "not_attempted_format_undefined", // 说明格式未冻结，因此没有尝试解析。
    validation_status: "not_attempted_contract_undefined", // 说明校验合同未冻结，因此没有判定通过或失败。
    content_included: false, // 明确草案不会嵌入文件内容。
    sha256: null, // 明确轻量页面没有读取内容，因此不能计算哈希。
    errors: [], // 未尝试解析时没有解析错误，不能伪造失败。
  }; // 结束文件来源记录。
} // 结束 IMP 文件元数据转换函数。

/**
 * 渲染用户当前选择的 IMP 文件元数据。
 * @returns {void} 本函数只更新文件列表。
 */
function renderMaterialFiles() { // 定义材料文件列表渲染函数。
  dom.fileList.replaceChildren(); // 在重新渲染前移除旧列表避免状态混杂。
  if (state.materialFiles.length === 0) { // 没有文件时显示明确空状态。
    const empty = document.createElement("p"); // 创建空状态段落。
    empty.className = "field-note"; // 使用辅助说明样式。
    empty.textContent = "尚未选择材料；草案仍可导出，但不会生成任何文件事实。"; // 说明无文件不触发默认值或伪造来源。
    dom.fileList.append(empty); // 将空状态加入列表容器。
    return; // 完成空状态渲染后退出。
  } // 结束空文件判断。
  state.materialFiles.forEach(function renderOneFile(file) { // 逐个渲染浏览器真实返回的文件对象。
    const record = document.createElement("div"); // 创建单个文件记录容器。
    record.className = "file-record"; // 应用文件记录样式。
    const name = document.createElement("strong"); // 创建文件名元素。
    name.textContent = file.name; // 安全写入真实文件名。
    const metadata = document.createElement("span"); // 创建文件元数据说明元素。
    const typeText = file.type || "文件未声明类型"; // MIME 为空时仅显示缺失，不补通用类型。
    metadata.textContent = formatBytes(file.size) + " · " + typeText + " · 未解析 / 未校验 / 未读取内容"; // 汇总真实大小、类型和处理状态。
    record.append(name, metadata); // 将文件名和元数据加入记录。
    dom.fileList.append(record); // 将记录加入文件列表。
  }); // 结束逐文件渲染。
} // 结束材料文件列表渲染函数。

/**
 * 根据文件选择器事件更新材料状态。
 * @param {Event} event 文件输入控件触发的 change 事件。
 * @returns {void} 本函数保存 File 引用并更新草案预览。
 */
function handleImpSelection(event) { // 定义 IMP 文件选择处理函数。
  const input = event.currentTarget; // 读取触发事件的文件输入控件。
  state.materialFiles = Array.from(input.files || []); // 将只读 FileList 转为当前页面数组。
  renderMaterialFiles(); // 展示新的文件元数据。
  renderManifestPreview(); // 同步刷新草案 JSON 预览。
  setMessage(dom.formMessage, "已记录 " + state.materialFiles.length + " 个文件的浏览器元数据；未读取或上传内容。", "neutral"); // 报告真实选择数量和处理边界。
} // 结束 IMP 文件选择处理函数。

/**
 * 清除当前材料选择和相关浏览器状态。
 * @returns {void} 本函数不删除用户磁盘上的任何文件。
 */
function clearMaterialFiles() { // 定义材料清除函数。
  state.materialFiles = []; // 清空页面保存的 File 引用数组。
  dom.impFiles.value = ""; // 重置原生文件输入控件。
  renderMaterialFiles(); // 渲染明确空状态。
  renderManifestPreview(); // 刷新草案 JSON 以移除文件记录。
  setMessage(dom.formMessage, "已清除页面中的材料选择；用户文件未被修改。", "neutral"); // 说明操作只影响当前页面。
} // 结束材料清除函数。

/**
 * 将每行一个的来源文本转换为非空字符串数组。
 * @param {string} sourceText 用户输入的多行来源文本。
 * @returns {string[]} 返回去除首尾空白和空行后的来源数组。
 */
function parseAcceptableSources(sourceText) { // 定义来源列表解析函数。
  return sourceText.split(/\r?\n/).map(function trimSource(item) { return item.trim(); }).filter(Boolean); // 按换行拆分并仅保留非空来源文本。
} // 结束来源列表解析函数。

/**
 * 将填写完整的缺失事实加入页面状态。
 * @returns {void} 本函数校验三个核心字段并更新列表。
 */
function addMissingFact() { // 定义添加缺失事实函数。
  const path = dom.missingPath.value.trim(); // 读取并清理逻辑字段路径。
  const reason = dom.missingReason.value.trim(); // 读取并清理缺失原因。
  const question = dom.missingQuestion.value.trim(); // 读取并清理确认问题。
  const acceptableSources = parseAcceptableSources(dom.missingSources.value); // 解析用户明确提供的可接受来源。
  if (!path || !reason || !question) { // 任一核心字段为空时不创建不完整记录。
    setMessage(dom.formMessage, "添加 missing fact 前，请填写字段路径、缺失原因和确认问题。", "error"); // 明确提示缺失的输入类别。
    if (!path) { dom.missingPath.focus(); } else if (!reason) { dom.missingReason.focus(); } else { dom.missingQuestion.focus(); } // 将焦点移到第一个缺失核心字段。
    return; // 阻止不完整记录进入草案。
  } // 结束核心字段完整性判断。
  const duplicateIndex = state.missingFacts.findIndex(function findDuplicate(item) { return item.path === path; }); // 查找相同逻辑路径以避免重复提问。
  const row = { path: path, reason: reason, question: question, acceptable_sources: acceptableSources }; // 创建符合现有 MissingFact 结构的记录。
  if (duplicateIndex >= 0) { // 相同路径已存在时采用当前用户输入替换旧记录。
    state.missingFacts[duplicateIndex] = row; // 更新相同路径的缺失事实。
  } else { // 相同路径不存在时追加新记录。
    state.missingFacts.push(row); // 将缺失事实加入页面状态。
  } // 结束重复路径处理。
  dom.missingPath.value = ""; // 清空路径输入便于登记下一项。
  dom.missingReason.value = ""; // 清空原因输入。
  dom.missingQuestion.value = ""; // 清空确认问题输入。
  dom.missingSources.value = ""; // 清空来源文本区。
  renderMissingFacts(); // 重新渲染缺失事实列表。
  renderManifestPreview(); // 同步刷新草案预览。
  setMessage(dom.formMessage, duplicateIndex >= 0 ? "已更新同路径 missing fact。" : "已添加 missing fact。", "success"); // 报告本地添加或更新结果。
} // 结束添加缺失事实函数。

/**
 * 删除指定下标的缺失事实。
 * @param {number} index 要从当前页面数组删除的零基下标。
 * @returns {void} 本函数更新列表和草案预览。
 */
function removeMissingFact(index) { // 定义删除缺失事实函数。
  state.missingFacts.splice(index, 1); // 仅从当前页面状态删除指定记录。
  renderMissingFacts(); // 更新缺失事实列表。
  renderManifestPreview(); // 同步更新草案预览。
  setMessage(dom.formMessage, "已从当前草案移除该 missing fact。", "neutral"); // 说明删除仅影响当前草案。
} // 结束删除缺失事实函数。

/**
 * 渲染用户显式登记的缺失事实。
 * @returns {void} 本函数使用原生 DOM 节点避免执行输入文本。
 */
function renderMissingFacts() { // 定义缺失事实列表渲染函数。
  dom.missingList.replaceChildren(); // 清空旧列表避免重复记录。
  if (state.missingFacts.length === 0) { // 尚未登记缺失事实时显示中性空状态。
    const empty = document.createElement("li"); // 创建空状态列表项。
    empty.textContent = "尚未显式登记 missing facts；页面不会自动从问题文本推断。"; // 说明空列表不等同于事实完整。
    dom.missingList.append(empty); // 将空状态加入列表。
    return; // 完成空状态后退出。
  } // 结束空缺失事实判断。
  state.missingFacts.forEach(function renderMissingFact(item, index) { // 逐个渲染用户登记记录。
    const row = document.createElement("li"); // 创建单条记录容器。
    const description = document.createElement("span"); // 创建记录文字元素。
    const sourceText = item.acceptable_sources.length > 0 ? item.acceptable_sources.join("、") : "未登记可接受来源"; // 仅根据用户输入显示来源或明确缺失。
    description.textContent = item.path + "｜" + item.reason + "｜问题：" + item.question + "｜来源：" + sourceText; // 安全写入记录内容。
    const removeButton = document.createElement("button"); // 创建删除按钮。
    removeButton.type = "button"; // 防止删除按钮提交表单。
    removeButton.className = "remove-button"; // 应用删除按钮样式。
    removeButton.textContent = "删除"; // 提供清晰操作名称。
    removeButton.setAttribute("aria-label", "删除 missing fact：" + item.path); // 为辅助技术提供具体删除目标。
    removeButton.addEventListener("click", function handleRemove() { removeMissingFact(index); }); // 将按钮绑定到当前记录下标。
    row.append(description, removeButton); // 将文字和按钮加入记录。
    dom.missingList.append(row); // 将记录加入列表。
  }); // 结束逐条缺失事实渲染。
} // 结束缺失事实列表渲染函数。

/**
 * 根据当前页面状态生成 ProblemManifest 输入草案。
 * @returns {Object} 返回现有 schema 八个必填字段及来源报告。
 */
function buildManifest() { // 定义草案构建函数。
  const inputFiles = state.materialFiles.map(describeImpFile); // 将当前 File 对象转换为不含内容的来源记录。
  return { // 返回仅由当前浏览器输入形成的草案对象。
    manifest_version: "1.0", // 使用仓库现有 ProblemManifest 合同版本。
    task_id: state.taskId, // 使用当前页面加载生成的任务标识。
    user_goal: dom.userGoal.value.trim(), // 原样保存用户当前自然语言目标并去除首尾空白。
    input_files: inputFiles, // 保存用户选择文件的浏览器元数据。
    facts: {}, // 静态页面不解析工程事实，因此保持空对象。
    missing_facts: state.missingFacts.map(function copyMissingFact(item) { return { path: item.path, reason: item.reason, question: item.question, acceptable_sources: item.acceptable_sources.slice() }; }), // 深复制用户显式登记的缺失事实。
    algorithm_configuration: {}, // 静态页面不选择算法配置，因此保持空对象。
    observations: [ // 记录草案的真实生成边界。
      "这是由 frontend-lite 本地静态页面生成的输入草案。", // 说明生成来源。
      "页面未连接后端，未创建远端任务。", // 说明没有服务端状态。
      "IMP 文件未解析、未校验且未嵌入内容。", // 说明材料处理状态。
      "页面未调用模型、未生成网格、未运行有限元求解器。", // 说明没有计算发生。
      "工程事实与算法配置仍需用户或后端根据当前证据确认。", // 说明后续确认责任。
    ], // 结束草案观察列表。
    provenance_report: { // 提供与当前草案一致的非阻断来源摘要。
      task_id: state.taskId, // 关联当前草案任务标识。
      fact_count: 0, // 静态页面没有生成工程事实。
      algorithm_setting_count: 0, // 静态页面没有生成算法配置。
      missing_fact_count: state.missingFacts.length, // 记录用户显式登记的缺失事实数量。
      issues: [], // 页面未执行后端来源校验，因此不伪造问题列表。
      has_legacy_fixture_values: false, // 页面从未读取旧案例值。
      has_pending_assumptions: false, // 页面不创建待确认假设。
    }, // 结束来源摘要对象。
  }; // 结束 ProblemManifest 草案对象。
} // 结束草案构建函数。

/**
 * 生成并展示当前 ProblemManifest 草案 JSON。
 * @returns {Object} 返回与预览一致的草案对象。
 */
function renderManifestPreview() { // 定义草案预览函数。
  const manifest = buildManifest(); // 根据当前页面状态构建草案。
  dom.manifestPreview.textContent = JSON.stringify(manifest, null, 2); // 使用两空格缩进安全展示 JSON。
  return manifest; // 返回草案供下载流程复用。
} // 结束草案预览函数。

/**
 * 下载一个浏览器内生成的 JSON 文件。
 * @param {Object} payload 要序列化的数据对象。
 * @param {string} filename 下载文件名。
 * @returns {void} 本函数在创建下载后立即释放临时 URL。
 */
function downloadJson(payload, filename) { // 定义 JSON 下载函数。
  const jsonText = JSON.stringify(payload, null, 2) + "\n"; // 使用两空格缩进并添加末尾换行。
  const blob = new Blob([jsonText], { type: "application/json;charset=utf-8" }); // 创建只包含草案文本的 UTF-8 JSON Blob。
  const objectUrl = URL.createObjectURL(blob); // 为当前 Blob 创建临时浏览器 URL。
  const link = document.createElement("a"); // 创建临时下载链接。
  link.href = objectUrl; // 将临时 URL 绑定到下载链接。
  link.download = filename; // 指定用户可识别的下载文件名。
  document.body.append(link); // 临时加入文档以兼容浏览器点击行为。
  link.click(); // 触发用户已请求的下载。
  link.remove(); // 下载触发后移除临时链接元素。
  window.setTimeout(function revokeObjectUrl() { URL.revokeObjectURL(objectUrl); }, 0); // 在当前调用栈结束后释放临时 URL，零毫秒仅表示排入下一轮事件循环。
} // 结束 JSON 下载函数。

/**
 * 校验并导出 ProblemManifest 草案。
 * @param {SubmitEvent} event 表单提交事件。
 * @returns {void} 本函数阻止浏览器网络提交并改为本地下载。
 */
function handleManifestExport(event) { // 定义草案导出处理函数。
  event.preventDefault(); // 阻止表单向任何地址发送网络请求。
  const goal = dom.userGoal.value.trim(); // 读取去除首尾空白后的自然语言目标。
  if (!goal) { // 自然语言目标为空时阻止导出。
    setMessage(dom.formMessage, "请先填写当前自然语言有限元问题。", "error"); // 告知用户唯一必填字段。
    dom.userGoal.focus(); // 将焦点移到问题文本区。
    return; // 结束本次导出尝试。
  } // 结束自然语言目标校验。
  const manifest = renderManifestPreview(); // 生成与页面预览一致的草案。
  downloadJson(manifest, state.taskId + "-problem-manifest-draft.json"); // 下载带任务标识的草案文件。
  setMessage(dom.formMessage, "已导出输入草案；这不代表后端任务、模型调用或求解已经发生。", "success"); // 明确本地下载完成与工程运行的区别。
} // 结束草案导出处理函数。

/**
 * 使用字符串安全创建列表内容。
 * @param {HTMLElement} listElement 要更新的 ul 或 ol 元素。
 * @param {string[]} items 要显示的非执行文本数组。
 * @param {string} emptyText 数组为空时显示的边界文本。
 * @returns {void} 本函数替换整个列表。
 */
function renderStringList(listElement, items, emptyText) { // 定义安全列表渲染函数。
  listElement.replaceChildren(); // 移除旧列表避免证据混杂。
  const values = items.length > 0 ? items : [emptyText]; // 空数组时使用明确的未提供文本。
  values.forEach(function appendItem(item) { // 逐条创建安全文本节点。
    const row = document.createElement("li"); // 创建列表项。
    row.textContent = String(item); // 将任意值转为文本并避免执行 HTML。
    listElement.append(row); // 将列表项加入目标列表。
  }); // 结束逐条列表渲染。
} // 结束安全列表渲染函数。

/**
 * 渲染明确提供的预算或计数字段。
 * @param {Object} values 仅包含来源 JSON 中显式数值或文本的对象。
 * @returns {void} 本函数不推导或汇总未知字段。
 */
function renderMetricList(values) { // 定义预算指标渲染函数。
  dom.budgetValues.replaceChildren(); // 清空旧指标避免来源混杂。
  const entries = Object.entries(values); // 读取对象自身可枚举字段。
  if (entries.length === 0) { // 没有明确预算字段时显示未提供状态。
    const row = document.createElement("div"); // 创建单条定义列表容器。
    const term = document.createElement("dt"); // 创建指标名称。
    const description = document.createElement("dd"); // 创建指标值。
    term.textContent = "状态"; // 使用中性指标名称。
    description.textContent = "源文件未提供"; // 明确没有预算来源。
    row.append(term, description); // 组装定义列表项。
    dom.budgetValues.append(row); // 将空状态加入列表。
    return; // 完成空状态后退出。
  } // 结束空预算判断。
  entries.forEach(function appendMetric(entry) { // 逐个渲染明确字段。
    const row = document.createElement("div"); // 创建指标行容器。
    const term = document.createElement("dt"); // 创建指标名称元素。
    const description = document.createElement("dd"); // 创建指标值元素。
    term.textContent = entry[0]; // 原样显示来源字段名。
    description.textContent = typeof entry[1] === "object" ? JSON.stringify(entry[1]) : String(entry[1]); // 对对象安全序列化，其余值转为文本。
    row.append(term, description); // 组装指标行。
    dom.budgetValues.append(row); // 将指标行加入定义列表。
  }); // 结束逐个预算指标渲染。
} // 结束预算指标渲染函数。

/**
 * 从标准数组型 agent trace 中提取明确字段。
 * @param {Array} events 现有 TraceRecorder 写出的事件数组。
 * @returns {Object} 返回仅基于标准事件字段的展示摘要。
 */
function normalizeAgentTrace(events) { // 定义标准 trace 规范化函数。
  const decisions = events.filter(function isDecision(event) { return event && event.type === "decision" && event.decision && typeof event.decision === "object"; }); // 选择明确的 decision 事件。
  const observations = events.filter(function isObservation(event) { return event && event.type === "execution_observation" && event.observation && typeof event.observation === "object"; }); // 选择明确的 execution_observation 事件。
  const lastDecision = decisions.length > 0 ? decisions[decisions.length - 1].decision : null; // 读取最后一个明确决策对象。
  const evidenceItems = observations.map(function describeObservation(event, index) { return "execution_observation[" + index + "]：" + JSON.stringify(event.observation); }); // 原样摘要执行观察，不推断工程语义。
  const decisionIssues = decisions.flatMap(function collectIssues(event) { return Array.isArray(event.decision.issues) ? event.decision.issues.map(String) : []; }); // 收集 decision 明确记录的问题。
  const observationErrors = observations.filter(function hasError(event) { return event.observation.status === "error" || event.observation.error; }).map(function describeError(event) { return JSON.stringify(event.observation); }); // 收集显式错误观察。
  const selectedSkills = lastDecision && Array.isArray(lastDecision.selected_skill_ids) ? lastDecision.selected_skill_ids.map(String) : []; // 读取最后决策明确选择的 Skill ID。
  const questions = lastDecision && Array.isArray(lastDecision.questions_for_user) ? lastDecision.questions_for_user.map(String) : []; // 读取最后决策明确提出的问题。
  const decisionText = lastDecision ? "action=" + String(lastDecision.action || "未提供") + "；rationale=" + String(lastDecision.rationale || "未提供") : "trace 中没有标准 decision 事件。"; // 组合明确动作和理由。
  const evidenceText = evidenceItems.concat(selectedSkills.map(function describeSkill(skillId) { return "selected_skill_id：" + skillId; }), questions.map(function describeQuestion(question) { return "question_for_user：" + question; })); // 将明确观察、Skill 和问题合并为展示列表。
  return { // 返回不包含工程推断的标准 trace 摘要。
    kind: "agent_trace_array", // 标记该数据符合数组型 trace 入口形态。
    stage: NOT_PROVIDED_TEXT, // 当前 trace 合同没有标准阶段字段，因此不使用 action 冒充。
    decision: decisionText, // 保存最后一个明确决策的文本。
    evidence: evidenceText, // 保存明确执行观察、Skill 和问题。
    budget: {}, // 当前标准 trace 没有统一预算字段。
    errors: decisionIssues.concat(observationErrors), // 保存明确 decision issues 和错误观察。
    boundaries: [], // 当前标准 trace 没有统一用途边界字段。
  }; // 结束标准 trace 摘要。
} // 结束标准 trace 规范化函数。

/**
 * 从对象型 validation receipt 候选中读取少量明确字段。
 * @param {Object} payload 解析后的对象型 JSON。
 * @returns {Object} 返回标记为结构未验证的只读摘要。
 */
function normalizeReceiptCandidate(payload) { // 定义对象型 receipt 候选规范化函数。
  const scenarios = Array.isArray(payload.scenarios) ? payload.scenarios : []; // 仅在 scenarios 是数组时读取场景。
  const explicitStage = typeof payload.stage === "string" ? payload.stage : (typeof payload.status === "string" ? payload.status : NOT_PROVIDED_TEXT); // 只从明确 stage 或 status 字段读取阶段文本。
  const explicitDecision = payload.decision && typeof payload.decision === "object" ? JSON.stringify(payload.decision) : (typeof payload.action === "string" ? "action=" + payload.action : "对象型 receipt 候选未提供标准 decision 字段。"); // 只读取明确 decision 或 action 字段。
  const evidenceItems = []; // 创建显式证据展示数组。
  if (typeof payload.summary_file === "string") { evidenceItems.push("summary_file：" + payload.summary_file); } // 记录明确 summary_file 字段。
  if (typeof payload.scenario_count === "number") { evidenceItems.push("scenario_count：" + payload.scenario_count); } // 记录明确 scenario_count 字段而不解释通过含义。
  scenarios.forEach(function collectScenarioEvidence(scenario, index) { // 逐个读取场景中的明确字段。
    if (!scenario || typeof scenario !== "object") { return; } // 跳过非对象场景项。
    if (typeof scenario.case_id === "string") { evidenceItems.push("scenarios[" + index + "].case_id：" + scenario.case_id); } // 记录明确案例标识。
    if (Array.isArray(scenario.evidence_refs)) { scenario.evidence_refs.forEach(function collectRef(reference) { evidenceItems.push("evidence_ref：" + String(reference)); }); } // 记录明确证据引用。
  }); // 结束场景证据读取。
  const budget = payload.totals && typeof payload.totals === "object" ? payload.totals : (payload.counters && typeof payload.counters === "object" ? payload.counters : {}); // 仅使用明确 totals 或 counters 对象。
  const errors = Array.isArray(payload.errors) ? payload.errors.map(String) : []; // 仅使用明确 errors 数组。
  const boundaries = []; // 创建显式用途边界数组。
  scenarios.forEach(function collectScenarioBoundaries(scenario) { // 逐个读取场景的明确用途字段。
    if (!scenario || typeof scenario !== "object") { return; } // 跳过非对象场景项。
    if (Array.isArray(scenario.can_use)) { scenario.can_use.forEach(function collectCanUse(item) { boundaries.push("can_use：" + String(item)); }); } // 记录明确 can_use 项。
    if (Array.isArray(scenario.cannot_use)) { scenario.cannot_use.forEach(function collectCannotUse(item) { boundaries.push("cannot_use：" + String(item)); }); } // 记录明确 cannot_use 项。
  }); // 结束用途边界读取。
  return { // 返回结构未验证的 receipt 候选摘要。
    kind: "receipt_candidate_unvalidated", // 标记对象形态没有冻结 schema。
    stage: explicitStage, // 保存明确 stage 或 status 文本。
    decision: explicitDecision, // 保存明确 decision 或 action 文本。
    evidence: evidenceItems, // 保存明确证据引用和场景字段。
    budget: budget, // 保存明确 totals 或 counters 对象。
    errors: errors, // 保存明确 errors 数组。
    boundaries: boundaries, // 保存明确 can_use 和 cannot_use 项。
  }; // 结束 receipt 候选摘要。
} // 结束对象型 receipt 候选规范化函数。

/**
 * 将规范化证据摘要渲染到六个状态区域。
 * @param {Object} summary normalizeAgentTrace 或 normalizeReceiptCandidate 的输出。
 * @returns {void} 本函数只展示明确字段。
 */
function renderEvidenceSummary(summary) { // 定义证据摘要渲染函数。
  dom.stageValue.textContent = summary.stage; // 安全显示明确阶段或未提供文本。
  dom.decisionValue.textContent = summary.decision; // 安全显示明确决策文本。
  renderStringList(dom.evidenceValues, summary.evidence, NOT_PROVIDED_TEXT); // 渲染执行与证据列表。
  renderMetricList(summary.budget); // 渲染明确预算或计数对象。
  renderStringList(dom.errorValues, summary.errors, "源文件没有提供明确 errors 或 issues。"); // 渲染明确错误列表。
  renderStringList(dom.boundaryValues, summary.boundaries, NOT_PROVIDED_TEXT); // 渲染明确用途边界列表。
} // 结束证据摘要渲染函数。

/**
 * 重置证据状态，防止旧数据与新文件失败状态混杂。
 * @returns {void} 本函数清空内存和全部证据展示。
 */
function resetEvidenceDisplay() { // 定义证据清除函数。
  state.importedPayload = null; // 清空当前解析结果。
  state.importedName = ""; // 清空当前证据文件名。
  dom.evidenceFile.value = ""; // 重置原生证据文件选择器。
  dom.evidencePreview.textContent = "尚未导入证据。"; // 恢复原始 JSON 空状态。
  renderEvidenceSummary({ stage: NOT_PROVIDED_TEXT, decision: "尚未导入真实 trace。", evidence: [], budget: {}, errors: ["尚未导入证据。"], boundaries: [] }); // 恢复六个状态区域的中性空状态。
  setMessage(dom.evidenceMessage, "", "neutral"); // 清空证据动态消息。
} // 结束证据清除函数。

/**
 * 读取并展示用户选择的真实 JSON 证据文件。
 * @param {Event} event 证据文件输入控件触发的 change 事件。
 * @returns {Promise<void>} 在文件读取和 JSON 解析完成后结束。
 */
async function handleEvidenceSelection(event) { // 定义异步证据读取函数。
  resetEvidenceDisplay(); // 在处理新文件前清除旧证据避免混杂。
  const input = event.currentTarget; // 读取触发事件的文件输入控件。
  const file = input.files && input.files[0] ? input.files[0] : null; // 只处理用户选择的第一个 JSON 文件。
  if (!file) { // 用户取消选择时保持空状态。
    return; // 不显示错误，因为取消不是失败。
  } // 结束无文件判断。
  if (file.size > EVIDENCE_PREVIEW_LIMIT_BYTES) { // 文件超过十 MiB 浏览器内存保护上限时拒绝读取。
    setMessage(dom.evidenceMessage, "证据文件超过 10 MiB 本地预览上限；文件未读取。", "error"); // 说明该限制只影响页面预览。
    return; // 阻止大文件进入浏览器内存。
  } // 结束证据文件大小判断。
  try { // 捕获文件读取或 JSON 解析失败。
    const text = await file.text(); // 在用户明确选择后读取证据文本。
    const payload = JSON.parse(text); // 使用标准 JSON 解析器解析内容。
    const summary = Array.isArray(payload) ? normalizeAgentTrace(payload) : (payload && typeof payload === "object" ? normalizeReceiptCandidate(payload) : null); // 根据根节点形态选择严格 trace 或对象候选路径。
    if (!summary) { // 根节点既不是数组也不是对象时拒绝展示。
      throw new Error("JSON 根节点必须是 agent trace 数组或 receipt 候选对象。"); // 创建可读格式错误。
    } // 结束根节点形态判断。
    state.importedPayload = payload; // 保存当前成功解析的数据供页面查看。
    state.importedName = file.name; // 保存真实来源文件名。
    dom.evidencePreview.textContent = JSON.stringify(payload, null, 2); // 安全展示未经改写的格式化 JSON。
    renderEvidenceSummary(summary); // 展示仅来自明确字段的摘要。
    const kindText = summary.kind === "agent_trace_array" ? "标准数组型 agent trace" : "对象型 receipt 候选（结构未验证）"; // 说明当前采用的解析边界。
    setMessage(dom.evidenceMessage, "已本地导入 " + file.name + "；识别为" + kindText + "。", "success"); // 报告真实文件名和识别形态。
  } catch (error) { // 处理文件读取或 JSON 解析异常。
    resetEvidenceDisplay(); // 清空可能残留的部分状态。
    setMessage(dom.evidenceMessage, "无法读取证据 JSON：" + String(error && error.message ? error.message : error), "error"); // 显示明确解析错误而不保留旧结果。
  } // 结束证据读取异常处理。
} // 结束异步证据读取函数。

/* 生成当前页面的任务标识并写入只读控件。 */
state.taskId = createTaskId(); // 在每次页面加载时创建一个非后端任务标识。
/* 将任务标识显示给用户。 */
dom.taskId.value = state.taskId; // 保证下载文件和预览使用同一标识。
/* 绑定 IMP 文件选择事件。 */
dom.impFiles.addEventListener("change", handleImpSelection); // 文件变化时只读取元数据并刷新草案。
/* 绑定材料清除按钮。 */
dom.clearFiles.addEventListener("click", clearMaterialFiles); // 点击时清除页面 File 引用。
/* 绑定缺失事实添加按钮。 */
dom.addMissingFact.addEventListener("click", addMissingFact); // 点击时校验并登记 missing fact。
/* 绑定草案预览按钮。 */
dom.previewManifest.addEventListener("click", function handlePreviewClick() { renderManifestPreview(); setMessage(dom.formMessage, "已根据当前浏览器输入刷新草案预览。", "success"); }); // 点击时只更新本地 JSON。
/* 绑定表单提交事件。 */
dom.form.addEventListener("submit", handleManifestExport); // 提交时阻止网络请求并下载草案。
/* 绑定真实证据文件选择事件。 */
dom.evidenceFile.addEventListener("change", handleEvidenceSelection); // 选择文件后进行本地 JSON 解析。
/* 绑定证据清除按钮。 */
dom.clearEvidence.addEventListener("click", resetEvidenceDisplay); // 点击时清除旧证据和全部摘要。
/* 渲染初始材料空状态。 */
renderMaterialFiles(); // 明确页面尚未选择材料。
/* 渲染初始缺失事实空状态。 */
renderMissingFacts(); // 明确页面尚未登记缺参。
/* 生成初始 ProblemManifest 草案预览。 */
renderManifestPreview(); // 让用户立即看到无隐藏默认值的八字段结构。
/* 初始化证据区域为空状态。 */
resetEvidenceDisplay(); // 确保页面启动时不展示任何虚构证据。