# 重写切片 02：工程经验 Skill 与新决策循环

本切片第一次加入新的 DeepSeek 决策路径。

## 新决策链

```text
当前用户输入/文件
→ ProblemManifest（事实、来源、缺参、假设）
→ 读取通用工程经验 Skill
→ DeepSeek 只选择一个下一步动作
→ 保存完整 request/response/decision trace
→ 等待用户、生成当前任务代码、执行/修改代码，或记录当前结论
```

没有 `required_artifacts`、`finish_requirements`、solver 次数、PSO 次数或论文门禁。

## Skill 的含义

Skill 只记录可复用的工程问题、证据需求、推理步骤、工具使用建议、输出记录与适用边界。Skill 不允许包含：

- 当前案例参数；
- 固定几何、材料、荷载或边界条件；
- 案例专用 builder/function；
- 必须生成的 artifact 清单；
- 用于通关的 finish requirements。

## 版本与模型配置

`DEEPSEEK_MODEL` 必须显式配置。V5 不会静默选择模型版本，以避免不同版本的提示、能力和结果继续混在一起。

## 无门禁

API 配置错误、模型响应错误、代码错误和求解错误都写入 trace。它们会影响下一步决策，但不会删除报告和中间记录。
