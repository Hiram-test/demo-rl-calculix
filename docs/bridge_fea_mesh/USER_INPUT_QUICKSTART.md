# 用户问题输入入口

## 浏览器入口

```bash
python -m pip install -r requirements-mesh-need.txt
PYTHONPATH=src python scripts/mesh_need.py serve --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765`。页面可输入工程问题、预定用途、当前结论、模型背景、CalculiX `.inp` 路径、固定 QoI、网格结果序列、平衡/能量证据以及提供商无关的 AI 候选诊断。

每次运行至少生成：

```text
case.json
diagnosis.json
ai_prompt.json
skill_result.json
evidence_ledger.json
pipeline-summary.json
```

## 终端入口

```bash
PYTHONPATH=src python scripts/mesh_need.py ask \
  --question "裂纹尖端应力随网格细化越来越大，还应继续加密吗？" \
  --intended-use "比较固定路径结构应力" \
  --current-claim "当前网格足以支持设计比较" \
  --qoi-name "fixed-path stress" \
  --qoi-location "same physical path" \
  --qoi-method "fixed path average" \
  --mesh-series '[{"h":4,"peak":100,"qoi":82},{"h":2,"peak":150,"qoi":83},{"h":1,"peak":230,"qoi":83.4}]' \
  --output-dir mesh-need-runs/crack-tip
```

也可直接运行仓库示例：

```bash
PYTHONPATH=src python scripts/mesh_need.py pipeline \
  --case examples/mesh_need/crack_tip_case.json \
  --ai-proposal examples/mesh_need/unsafe_ai_hotspot_proposal.json \
  --output-dir /tmp/crack-tip-demo
```

AI JSON 留空时使用可重复规则初诊；提供 AI JSON 时，数值、连接、QoI 和重网格保护层仍可覆盖它。系统不内置云模型密钥，也不会自行上传工程数据。
