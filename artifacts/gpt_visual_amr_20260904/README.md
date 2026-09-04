# GPT 视觉 AMR 实验凭据

本目录保存 2026-09-04 这轮真实实验的完整压缩包分卷与校验清单。实验代码、可直接阅读的 JSON 结果和汇总图也保存在仓库原路径。

- 完整 ZIP：20,907,631 字节，14 个按顺序拼接的分卷。
- 完整 ZIP SHA-256：`a07903dc28022b92982c116417de3dbe622ae7acdc0ccdfebeb1afeac05fc949`。
- 765 个文件，覆盖全部九个实验目录、两份报告、模型、五张 GPT 实际看过的图及其封存决策、实际求解输入与日志、运行时来源和独立验证。
- 实验求解共 152 次，另外保存了 1 次原生求解器验证；计数和每例具体结果可查 `runs/gpt_visual_results.json`。

在本目录执行：

```bash
cat gpt_visual_evidence.zip.part[0-9][0-9][0-9] > gpt_visual_evidence.zip # 按原字节顺序重建完整压缩包。
sha256sum -c SHA256SUMS.txt # 核对所有分卷、完整压缩包和旁附凭据。
unzip -t gpt_visual_evidence.zip # 核对 ZIP 内部完整性。
unzip gpt_visual_evidence.zip -d extracted_evidence # 在独立目录展开，避免覆盖当前代码和稍后增加的报告。
(cd extracted_evidence && sha256sum -c CONTENT_SHA256SUMS.txt) # 核对每一份归档文件。
```

`initial.npz`、`observation.npz`、图像和封存 JSON 均保留原始字节。为避免重复保存可计算的场，后继解和参考解 NPZ 只保留原始 `nodes.npy`、`cells.npy`、`u.npy` 字节；可用 `compute_post` 与 `zz_indicator` 重建派生场。清单明确区分完整原文件 `original_sha256` 与归档文件 `archived_sha256`；压缩包不能单独复核已经省略的完整原 NPZ 字节哈希。FRD、重复中间文件及原生运行库不包含在包中。

证据包创建后的固定候选图像审计与汇总图另存于仓库：`scripts/audit_visual_causality.py`、`runs/visual_wm_probe/visual_causality_audit.json`、`runs/gpt_visual_figures/`。它们没有增加真实求解或改变五例封存选择。最新说明以 `docs/gpt_visual_amr_20260904.md` 与独立审查为准。

本轮已经跑通 GPT 直接看图的视觉动作闭环；完整结果包含失败实例，并未证明整体战胜最强已测经典方法。
