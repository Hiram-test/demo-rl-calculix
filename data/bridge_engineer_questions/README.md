# 桥梁有限元工程师问题数据

## 文件

- `bridge_engineer_questions.csv.gz`：174 个原子问题的 UTF-8 CSV 压缩文件；
- `summary.json`：渠道、工作阶段、工程师工作和问题族汇总。

数据来自 81 个独立帖子或官方 FAQ：

- 57 个 SIMULIA Community 原子问题；
- 57 个 Ansys Innovation Space 原子问题；
- 25 个 ResearchGate 原子问题；
- 35 个 Eng-Tips、Dlubal、SOFiSTiK、Autodesk 和 Physics Forums 原子问题。

解压示例：

```bash
gzip -dc data/bridge_engineer_questions/bridge_engineer_questions.csv.gz   > bridge_engineer_questions.csv
```

## 边界

- 这是公开问题样本，不是统计抽样调查；
- 一个来源可以拆出多个原子问题；
- 论坛回答不作为权威结论；
- 问题文本为人工转述，不复制来源正文；
- 频次只能解释为“在本样本中出现”，不能外推为工程师总体比例。

详细方法见 `docs/bridge_fea_mesh/QUESTION_CODING_METHOD.md`。
