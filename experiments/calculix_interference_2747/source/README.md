# CalculiX 论坛 2747 原始大型输入

本目录保存可进入普通 Git 历史的小型来源清单、校验值、再分发说明与发布决策。作者公开的原始归档和两个大型 `.inp` 不写入 Git blob，而作为独立 GitHub prerelease 的资产提供：

- [CalculiX Interference Contact 2747 - Original Models and Evidence v1](https://github.com/Hiram-test/demo-rl-calculix/releases/tag/calculix-interference-2747-evidence-v1)
- 原始论坛问题：[Interference Contact and mesh refinement](https://calculix.discourse.group/t/interference-contact-and-mesh-refinement/2747)
- 作者公开附件：[Proton Drive share](https://drive.proton.me/urls/2YKDRXYGJG#fwmEJFkaCbui)

## 发布资产

| 资产 | 字节数 | 用途 |
|---|---:|---|
| `Shear_setups.zip` | 116,272,555 | 作者发布的原始压缩包，字节保持不变，内含下面两个 `.inp`。 |
| `Shear_setup-INTER01-COARSE_PIN.inp` | 267,223,015 | 粗 PIN2 网格的完整原始输入，便于不解压直接下载。 |
| `Shear_setup-INTER01-deactivate1thenreactivate.inp` | 277,391,649 | 细 PIN2 网格并带 REMOVE→ADD 接触历史的完整原始输入。 |
| `local_calculix_counterfactual_followup.pdf` | 143,011 | 基于缩减模型的本地三项反事实复核，不是原始大型模型求解结果。 |

下载后必须用 [`SHA256SUMS.txt`](SHA256SUMS.txt) 校验。归档内恰好包含上述两个 `.inp`，两者都不依赖外部 `*INCLUDE` 文件。

## 为什么使用 Release

三个原始文件都超过 GitHub 普通 Git 的 100 MiB 单文件限制。它们是一次性的第三方研究附件，不值得为整个仓库引入 Git LFS 的配额与 clone 行为变化。因此：

- Git 保存清单、哈希、来源、边界和决策记录；
- GitHub Release 保存原始二进制归档和大型文本输入；
- Release 标为 prerelease，因为当前根因状态仍为 `narrowed_unresolved`。

## 求解边界

这两个原始输入约有 383 万至 386 万节点、约 1150 万至 1160 万自由度，并请求 Pardiso。本仓库没有声称在当前工作站或标准 GitHub runner 上完成它们的求解。现有论文和反事实结果来自真实 CalculiX 的缩减代表模型；它们用于缩小机制范围，不能替代原始大型模型的执行证据。

## `SOURCE_MANIFEST.json` 字段

- `manifest_schema`：清单数据契约的版本。
- `source`：论坛标题、帖子地址、附件地址及冻结时是否存在作者确认解。
- `release`：承载大文件的 tag、固定 URL、发布类型和根因状态。
- `assets`：每个主要发布资产的名称、角色、大小、摘要与字节保持方式。
- `execution_boundary`：是否执行过原始大型模型，以及不能把缩减模型结果升级为原模型结论的原因。

第三方文件的权利边界见 [`SOURCE_AND_REDISTRIBUTION_NOTICE.md`](SOURCE_AND_REDISTRIBUTION_NOTICE.md)，本轮“为什么这样发布”的可审计路径见 [`RELEASE_DECISION.md`](RELEASE_DECISION.md)，实际发布说明见 [`RELEASE_NOTES.md`](RELEASE_NOTES.md)。
