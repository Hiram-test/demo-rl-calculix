# Live LLM V3 接口修复结果

旧版稀疏接口连续出现空输出的根因已经定位为 reasoning token 截断，而不是模型主动返回空决策。V2 两次调用的 `finish_reason` 均为 `length`，700 个 completion token 全部计入 `reasoning_tokens`，最终公开 `content_length=0`；因此旧结果不得解释为 LLM 的机制判断失败。

V3 将 completion 预算改成 reasoning-aware 两级策略：先 4096 token，若仍然满足 `finish_reason=length` 且 reasoning token 占预算超过 80%，自动提升到 8192 token；系统只读取最终公开 answer，不读取或保存隐藏推理正文。第一次 V3 调用仍然在 4096/4096 reasoning token 后被长度截断，第二次 8192 上限调用正常以 `finish_reason=stop` 结束，其中 completion 3208、reasoning 3180，最终公开答案长度 53，因此证明接口问题已经修复。

真实 `deepseek-v4-pro` 最终委派为 region 12 depth 2 confidence 0.8 与 region 10 depth 1 confidence 0.7。该结果经过 deterministic compiler 后执行真实 CalculiX；动态 Dörfler 基线为 `H=3`、objective `1.256228669594032`，LLM 委派加物理纠错最终仍为 `H=3`、objective `2.6783955804257324`，因此没有节省反馈轮次且未达到共同终态质量门。真实 future-hit 高持续区域主要为 13、14、15，而本次 LLM 选择 12 与 10，depth MAE 为 0.9375，persistent-hotspot F1 为 0。

这个结果的解释边界非常明确：接口故障已经修复，当前剩余问题已经转化为真正的科学问题，即现有理论结构上下文和一次粗网格 Dörfler 证据不足以让该模型可靠识别未来持续热点。下一步需要改进机制 evidence skill，而不是继续修改输出格式；应优先向模型提供由理论推导或低成本数值探针得到的区域响应/持久性证据，再测试能否接近已被真实 CalculiX 证明存在的 `H=3 → H=1` trajectory-compression upper bound。
