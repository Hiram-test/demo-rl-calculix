#!/usr/bin/env python3
"""Send one finite-element evidence packet to DeepSeek and validate its JSON reply.

The API key is read only from the DEEPSEEK_API_KEY environment variable. The
script never prints or writes the key and never stores model reasoning traces.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mesh_need.diagnosis import validate_ai_analysis  # noqa: E402


SYSTEM_PROMPT = """你是有限元建模与网格证据分析助手。

你必须直接分析用户问题、模型描述和求解器证据，不得套用预写的物理分类规则，也不得把数值相关性冒充因果结论。请形成多个竞争性的物理或数值假设，明确区分观测、解释与未知，并提出能够最大程度区分这些假设的最小下一步计算。

只输出一个合法 JSON 对象，不要使用 Markdown，不要输出推理过程。JSON 必须包含：
{
  "problem_restatement": "需要作出的工程决策",
  "competing_hypotheses": [
    {
      "hypothesis": "假设",
      "supporting_evidence": ["引用输入 JSON 的字段路径和值"],
      "challenging_evidence": ["引用输入 JSON 的字段路径和值"],
      "currently_unresolved": ["尚不能区分之处"]
    }
  ],
  "evidence_assessment": [
    {
      "observation": "直接观测",
      "interpretation": "可能解释",
      "confidence": "高/中/低",
      "evidence_paths": ["输入 JSON 字段路径"]
    }
  ],
  "recommended_next_action": {
    "action": "一个最小的下一步计算或模型改动",
    "why_discriminating": "它如何区分竞争假设",
    "controlled_variables": ["保持不变的量"],
    "comparison_outputs": ["需要比较的固定输出量"],
    "decision_rule": "观察到何种差异分别支持哪些假设"
  },
  "uncertainties": ["仍未解决的不确定性"],
  "optional_skill": {
    "name": "可选工具或工作流；不需要时写 null",
    "justification": "为什么需要或不需要"
  }
}

不要认证模型正确，不要声称证据不足时存在唯一答案。至少给出两个竞争假设。"""


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        name: getattr(usage, name)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if getattr(usage, name, None) is not None
    }


def request_analysis(packet: dict[str, Any], model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not available in this workflow")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    packet_text = json.dumps(packet, ensure_ascii=False, indent=2)
    last_error: Exception | None = None

    for attempt in range(1, 3):
        user_prompt = (
            "请分析以下有限元证据包，并严格按 system 消息给出的 JSON 结构输出。"
            "所有证据引用必须指向本证据包中的实际字段，不要补造未提供的模型事实。\n\n"
            + packet_text
        )
        if attempt == 2:
            user_prompt = "上一次输出为空或结构无效。请重新输出完整、合法且非空的 JSON。\n\n" + user_prompt

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=12000,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
            stream=False,
        )

        content = response.choices[0].message.content
        if not content or not content.strip():
            last_error = RuntimeError("DeepSeek returned empty content")
            continue

        try:
            analysis = json.loads(content)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

        errors = validate_ai_analysis(analysis)
        if errors:
            last_error = RuntimeError("Invalid AI analysis: " + "; ".join(errors))
            continue

        metadata = {
            "provider": "deepseek",
            "requested_model": model,
            "response_model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "usage": _usage_dict(response.usage),
        }
        return analysis, metadata

    raise RuntimeError(f"DeepSeek did not return a valid analysis after two attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="AI analysis packet JSON")
    parser.add_argument("output", type=Path, help="validated DeepSeek analysis JSON")
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
    )
    args = parser.parse_args()

    packet = json.loads(args.input.read_text(encoding="utf-8"))
    analysis, metadata = request_analysis(packet, args.model)
    analysis["_metadata"] = metadata

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "deepseek_analysis_completed",
                "model": metadata["response_model"],
                "output": str(args.output),
                "hypothesis_count": len(analysis["competing_hypotheses"]),
                "usage": metadata["usage"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
