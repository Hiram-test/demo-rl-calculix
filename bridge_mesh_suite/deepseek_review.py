from __future__ import annotations

import json
import os
import time
from typing import Any


SYSTEM_PROMPT = """你是桥梁有限元网格报告复核员，面向普通工程人员。
你只根据给定证据回答，不要求用户理解专业理论，不罗列大量假设，不虚构模型信息。
每个场景只做四件事：
1. 直接回答用户的原始问题；
2. 用通俗语言说明为什么；
3. 明确系统已经实际修改或落实了什么；
4. 说明哪些结果能用、哪些不能用。
不得把尖点或裂纹尖端最大应力说成收敛设计值。不得把线弹性结果冒充弹塑性结论。
返回一个JSON对象，格式：
{
  "overall_review": "总体复核",
  "scenario_reviews": [
    {
      "scenario_id": "与输入一致",
      "user_answer": "直接面向用户的回答",
      "evidence_used": ["证据摘要"],
      "implemented_change": ["已落实修改"],
      "can_use": ["可用范围"],
      "cannot_use": ["不可用范围"]
    }
  ]
}
不要输出Markdown。"""


def review_with_deepseek(summary: dict[str, Any], *, model: str = "deepseek-v4-pro") -> dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for the full bridge-component run")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(summary, ensure_ascii=False)},
    ]
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=5000,
                extra_body={"thinking": {"type": "enabled"}},
            )
            text = response.choices[0].message.content or ""
            data = json.loads(text)
            if not isinstance(data.get("scenario_reviews"), list):
                raise ValueError("scenario_reviews missing")
            expected = {x["scenario_id"] for x in summary["scenarios"]}
            got = {str(x.get("scenario_id")) for x in data["scenario_reviews"]}
            if expected != got:
                raise ValueError(f"scenario ids mismatch: expected={expected}, got={got}")
            data["provider_metadata"] = {
                "provider": "deepseek",
                "model": model,
                "raw_response": text,
                "usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                },
            }
            return data
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"DeepSeek review failed: {last_error}")
