# 数据字段

`bridge_fea_mesh_cases/` 下的五个 CSV 使用相同字段，共包含 100 条来源记录：

- `t1_official_01.csv`、`t1_official_02.csv`：37 条官方标准、指南、政府报告和官方软件例；
- `t2_research_01.csv`、`t2_research_02.csv`：48 条同行评审研究；
- `t3_t4_signals.csv`：5 条行业/新闻信号和 10 条社区问题信号。

| 字段 | 含义 |
|---|---|
| `id` | 稳定目录编号；O=官方，P=论文，N=新闻，C=社区 |
| `evidence_tier` | T1–T4 证据层级 |
| `source_type` | 官方指南、软件例、论文、新闻或社区问题 |
| `publisher` | 发布机构或期刊 |
| `year` | 发布年份；官方在线文档以当前可访问版本为准 |
| `component_family` | 构件或桥梁系统 |
| `analysis_context` | 分析类型和场景 |
| `engineering_qoi` | 工程关注结果量 |
| `mesh_need_class` | 归一化网格需求类别 |
| `mesh_question` | 从来源抽取的核心网格问题 |
| `mesh_attention` | 应关注的建模、网格或验证事项 |
| `ai_intent_signal` | AI 在意图编译中应识别的信号 |
| `source_title` | 来源题名 |
| `url` | 官方永久链接或 DOI |

注意：`mesh_attention` 是对来源内容的结构化摘要，用于研究分类，不替代原文、规范或负责工程师判断。
