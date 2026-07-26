from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from .core import run_pipeline


def _json_field(form: dict[str, list[str]], name: str, default: Any) -> Any:
    raw = form.get(name, [""])[0].strip()
    if not raw:
        return default
    return json.loads(raw)


def _page(message: str = "", result: dict[str, Any] | None = None) -> bytes:
    result_html = ""
    if result is not None:
        result_html = f"<h2>本次输出</h2><pre>{html.escape(json.dumps(result, ensure_ascii=False, indent=2))}</pre>"
    message_html = f"<p><strong>{html.escape(message)}</strong></p>" if message else ""
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Model-aware FEA Mesh-Need MVP</title>
<style>body{{max-width:980px;margin:2rem auto;font-family:system-ui;line-height:1.5}}label{{display:block;font-weight:600;margin-top:1rem}}textarea,input{{width:100%;box-sizing:border-box;padding:.55rem}}textarea{{min-height:5rem}}button{{margin-top:1.2rem;padding:.7rem 1.2rem}}pre{{background:#f3f3f3;padding:1rem;overflow:auto}}</style></head>
<body><h1>Model-aware FEA Mesh-Need & Evidence MVP</h1>
<p>这里输出的是可检验诊断与证据台账，不是最终正确性裁决。</p>{message_html}
<form method="post">
<label>你的工程问题</label><textarea name="question" required></textarea>
<label>预定用途</label><input name="intended_use">
<label>当前希望支持的结论</label><input name="current_claim">
<label>模型、载荷与约束背景</label><textarea name="context"></textarea>
<label>CalculiX .inp 本地路径（可选）</label><input name="calculix_inp">
<label>固定 QoI 名称</label><input name="qoi_name">
<label>固定物理位置</label><input name="qoi_location">
<label>提取协议</label><input name="qoi_method">
<label>容差</label><input name="qoi_tolerance" value="0.02">
<label>网格结果 JSON（数组，每项含 h、peak、qoi）</label><textarea name="mesh_series">[]</textarea>
<label>力/力矩证据 JSON（可含 external_force、reaction_force、external_moment、reaction_moment）</label><textarea name="equilibrium">{{}}</textarea>
<label>能量历史 JSON（数组）</label><textarea name="energy_history">[]</textarea>
<label>QoI 指标点 JSON（可选）</label><textarea name="qoi_indicator_points">[]</textarea>
<label>热点区域 JSON（可选）</label><textarea name="hotspots">[]</textarea>
<label>提供商无关 AI 候选诊断 JSON（可选）</label><textarea name="ai_proposal">{{}}</textarea>
<label>输出目录</label><input name="output_dir" value="mesh-need-runs/latest">
<button type="submit">生成诊断与证据台账</button></form>{result_html}</body></html>""".encode("utf-8")


def application(environ: dict[str, Any], start_response: Any) -> list[bytes]:
    if environ.get("REQUEST_METHOD") == "GET":
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [_page()]
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        form = parse_qs(environ["wsgi.input"].read(length).decode("utf-8"), keep_blank_values=True)
        equilibrium = _json_field(form, "equilibrium", {})
        qoi_tolerance = float(form.get("qoi_tolerance", ["0.02"])[0] or 0.02)
        case: dict[str, Any] = {
            "question": form.get("question", [""])[0],
            "intended_use": form.get("intended_use", [""])[0],
            "current_claim": form.get("current_claim", [""])[0],
            "context": form.get("context", [""])[0],
            "calculix_inp": form.get("calculix_inp", [""])[0],
            "qoi": {"name": form.get("qoi_name", [""])[0], "location": form.get("qoi_location", [""])[0], "extraction_method": form.get("qoi_method", [""])[0], "tolerance": qoi_tolerance},
            "mesh_series": _json_field(form, "mesh_series", []),
            "energy_history": _json_field(form, "energy_history", []),
            "qoi_indicator_points": _json_field(form, "qoi_indicator_points", []),
            "hotspots": _json_field(form, "hotspots", []),
        }
        if isinstance(equilibrium, dict):
            case.update(equilibrium)
        proposal = _json_field(form, "ai_proposal", {})
        result = run_pipeline(case, Path(form.get("output_dir", ["mesh-need-runs/latest"])[0]), proposal if proposal else None)
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [_page("已生成。所有诊断仍需独立证据复核。", result)]
    except Exception as exc:  # local engineering UI: return actionable error
        start_response("400 Bad Request", [("Content-Type", "text/html; charset=utf-8")])
        return [_page(f"输入或执行失败：{exc}")]


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    with make_server(host, port, application) as server:
        print(f"Mesh-Need MVP running at http://{host}:{port}")
        server.serve_forever()
