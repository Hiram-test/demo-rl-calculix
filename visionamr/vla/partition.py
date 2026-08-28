"""Vision heads: propose named partition seeds from a rendered view.

The heads never emit geometric primitives; they point at structural
locations ("wheel patch edge", "inner strip line", "re-entrant corner",
"calm mid-field") the way a human marks up a plot, and the geodesic
partition in ``regions.Partition`` grows organic region shapes from
those seeds.

* ``ScriptedVisionPartitioner`` -- drawing-only stand-in: structural
  anchors on the CAD (loads, supports, corners, holes).  No solve.
* ``LLMVisionPartitioner``      -- multimodal LLM on geometry views
  (orthographic drawings, not a solved stress field).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..fem_post import PostState
from ..geometry import Problem
from ..indicators import zz_indicator  # noqa: F401  (kept for head extensions)
from .drawing import (
    REMAINDER_NAMES,
    VIEW_AXES,
    DrawnRegion,
    drawing_centroid_xyz,
    halo_drawing,
    irregular_from_points,
    markup_from_spec,
    poly_tuple,
    view_for_feature,
)
from .regions import Seed


def _farthest_point_seeds(points: np.ndarray, k: int, min_sep: float) -> list[int]:
    """Greedy farthest-point sampling indices."""

    if len(points) == 0 or k <= 0:
        return []
    picked = [int(np.argmin(points[:, 0]))]
    for _ in range(k - 1):
        d = np.min(
            np.linalg.norm(points[:, None, :] - points[picked][None, :, :], axis=2),
            axis=1,
        )
        cand = int(np.argmax(d))
        if d[cand] < min_sep:
            break
        picked.append(cand)
    return picked


@dataclass
class ScriptedVisionPartitioner:
    """Draw irregular regions on the probe views, then assign a size each.

    Mimics an engineer marking the drawing before the first analysis:
    a non-box stroke around each load, support, corner, or hole, then a
    remainder size for unpainted volume.  Does not read a solved field.
    """

    peak_quantile: float = 0.60
    max_hot: int = 5
    min_sep_frac: float = 0.10
    anchor_kinds: tuple[str, ...] = ("load", "support", "clamp", "corner", "hole")
    anchor_sep_frac: float = 0.06
    n_field: int = 3
    field_sep_frac: float = 0.22
    h_remainder: float = 0.78
    kind_frac: tuple = (
        ("load", 0.18, 0.32),
        ("hole", 0.20, 0.34),
        ("corner", 0.26, 0.40),
        ("support", 0.42, 0.58),
        ("clamp", 0.46, 0.62),
    )

    def propose(self, problem: Problem, post: PostState | None = None, eta2=None) -> list[Seed]:
        del post, eta2  # drawing step is solve-free; ignore any residual field
        diam = problem.diameter
        drawings = []
        used: set[str] = set()
        seed_pts: list[np.ndarray] = []

        for f in problem.features:
            if f.kind not in self.anchor_kinds:
                continue
            if any(
                np.linalg.norm(f.xyz - p) < self.anchor_sep_frac * diam for p in seed_pts
            ):
                continue
            name = f"{f.name}_zone"
            if name in used:
                name = f"{name}_a"
            used.add(name)
            view = view_for_feature(f, problem)
            rad = max(float(f.r), 0.05 * diam) * 1.4
            grade = self._kind_grade(f)
            from .grades import prior_h

            drawings.append(
                halo_drawing(
                    name, f.xyz, prior_h(grade, problem.h0), problem,
                    view=view, radius=rad, grade=grade,
                )
            )
            seed_pts.append(f.xyz)

        from .grades import prior_h

        field_h = prior_h(5, problem.h0)
        field_pt = _bbox_center(problem)
        seeds = [
            Seed(d.name, drawing_centroid_xyz(d, problem), d.h, d.origin) for d in drawings
        ]
        seeds.append(Seed("field", field_pt, field_h, origin="coarse"))
        if not drawings:
            drawings = [
                halo_drawing(
                    "field", np.array(field_pt), field_h, problem,
                    origin="coarse", grade=5,
                )
            ]
            seeds = [Seed("field", field_pt, field_h, origin="coarse")]
        self.last_drawings = drawings
        self.last_grades = {
            d.name: (d.grade if d.grade is not None else 4) for d in drawings
        }
        self.last_grades["field"] = 5
        return seeds

    def _kind_grade(self, feature) -> int:
        """Judge coarseness from the drawing, not a size."""

        tag = feature.name.lower()
        if feature.kind == "load" or any(w in tag for w in ("edge", "rim", "corner")):
            return 1
        if feature.kind == "hole":
            return 2
        if feature.kind in ("support", "clamp"):
            return 3
        return 4

    @staticmethod
    def _name(problem: Problem, pt: np.ndarray, rank: int, used: set[str]) -> str:
        best, dist = None, np.inf
        for f in problem.features:
            d = float(np.linalg.norm(pt - f.xyz))
            if d < dist:
                best, dist = f, d
        name = (
            f"{best.name}_zone"
            if best is not None and dist < 0.25 * problem.diameter
            else f"hotspot_{rank}"
        )
        if name in used:
            name = f"{name}_{rank}"
        return name


VLM_SYSTEM_PROMPT = """你是资深有限元网格工程师。你看到的是结构图纸的正交视图
（俯视 top = x-y，正视 front = x-z，侧视 side = y-z），
以及荷载与支承标注。这不是求解后的应力云图。不要假设已经算过场。

像用笔画一样圈出不规则区域。不要轴对齐方框，不要固定分块模板。

允许拆结构剖面。三向外视图看不清内部或接触面时，
可以自己定一个剖切面，在剖面上画区。
剖切面用 plane（xy / xz / yz）和 cut（沿未用轴的位置，模型单位）说明。
剖面不是盒子，也不是整层模板；只圈你真正要加密或放粗的那一块。

每一块只给一个粗细等级，1 最密、5 最疏。
图上看着不一样，等级就必须不一样。禁止所有区填同一个数。
不要给连续单元尺寸。尺寸是工具的，不是你的。

没画到的体积也要判别等级，不能默认当粗场。
看着剩余区域，给出 remainder_grade。
不要给连续尺寸，不要调参，不要委派参数。
不许省略。

只回复 JSON：
{"regions": [{"name": "<结构名>",
"view": "top"|"front"|"side"|"section",
"plane": "xy"|"xz"|"yz",
"cut": <剖面位置，仅 section 需要>,
"polygon": [[u,v], ...],
"grade": <1 最密 … 5 最疏>}, ...],
"remainder_grade": <1 到 5>}
多边形至少 3 个顶点，坐标用该视图或剖面的模型单位。
图上需要几块就画几块；不需要剖面就不要硬拆。
不要给连续单元尺寸。等级就够了。"""


def resolve_vlm_endpoint() -> tuple[str, str | None, str]:
    """Return (api_base, api_key, model).  Prefer SpaceXAI / xAI."""

    key = (
        os.environ.get("XAI_API_KEY")
        or os.environ.get("VLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if os.environ.get("XAI_API_KEY") and not os.environ.get("VLM_API_BASE"):
        base = "https://api.x.ai/v1"
        model = os.environ.get("VLM_MODEL", "grok-4")
    else:
        base = os.environ.get("VLM_API_BASE") or (
            "https://api.x.ai/v1" if os.environ.get("XAI_API_KEY")
            else "https://api.openai.com/v1"
        )
        default_model = "grok-4" if "x.ai" in base else "gpt-4o"
        model = os.environ.get("VLM_MODEL", default_model)
    return base, key, model


def _bbox_center(problem: Problem) -> tuple[float, float, float]:
    b = problem.bbox
    z = 0.0 if problem.dim == 2 else 0.5 * (b[2] + b[5])
    return (0.5 * (b[0] + b[3]), 0.5 * (b[1] + b[4]), z)


def ensure_remainder_seed(seeds: list[Seed], spec: dict, problem: Problem) -> list[Seed]:
    """Leftover volume must carry an eye-assigned grade, not an implicit h0."""

    rem = spec.get("remainder_fineness_fraction") if isinstance(spec, dict) else None
    rem_g = spec.get("remainder_grade") if isinstance(spec, dict) else None
    if rem is None and rem_g is not None:
        from .grades import GRADE_PRIOR, parse_grade

        rem = GRADE_PRIOR[parse_grade(rem_g, "remainder_grade")]
    out: list[Seed] = []
    has = False
    for s in seeds:
        if s.origin == "coarse" or s.name.lower() in REMAINDER_NAMES:
            has = True
            h = (
                float(np.clip(float(rem), 0.1, 1.0)) * problem.h0
                if rem is not None
                else s.h
            )
            out.append(Seed(s.name, s.xyz, h, origin="coarse"))
        else:
            out.append(s)
    if has:
        return out
    if rem is None:
        raise ValueError("JSON missing remainder_grade or remainder_fineness_fraction")
    h = float(np.clip(float(rem), 0.1, 1.0)) * problem.h0
    out.append(Seed("field", _bbox_center(problem), h, origin="coarse"))
    return out


def seeds_from_spec(spec: dict, problem: Problem, max_seeds: int = 12) -> list[Seed]:
    """Validate VLM JSON into seeds.  Drawn regions preferred; old seed lists still parse."""

    drawings = markup_from_spec(spec, problem, max_regions=max_seeds)
    seeds = [Seed(d.name, drawing_centroid_xyz(d, problem), d.h, d.origin) for d in drawings]
    return ensure_remainder_seed(seeds, spec, problem)


def drawings_from_spec(spec: dict, problem: Problem, max_regions: int = 12) -> list[DrawnRegion]:
    return markup_from_spec(spec, problem, max_regions=max_regions)


def parse_seed_json(content: str, problem: Problem, max_seeds: int = 12) -> list[Seed]:
    """Parse model content (raw JSON or fenced) into seeds."""

    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    spec = json.loads(text)
    return seeds_from_spec(spec, problem, max_seeds=max_seeds)


def load_seed_cache(path, problem: Problem) -> list[Seed]:
    spec = json.loads(Path_read(path))
    return seeds_from_spec(spec, problem, max_seeds=24)


def load_drawings_cache(path, problem: Problem) -> list[DrawnRegion]:
    return drawings_from_spec(json.loads(Path_read(path)), problem, max_regions=24)


def Path_read(path) -> str:
    from pathlib import Path

    return Path(path).read_text()


def dump_seed_cache(path, seeds: list[Seed], drawings: list | None = None, problem=None) -> None:
    from pathlib import Path

    if drawings:
        spec = {
            "regions": [
                {
                    "name": d.name,
                    "view": d.view,
                    "fineness_fraction": (
                        d.h / problem.h0 if problem is not None else d.h
                    ),
                    "polygon": [list(p) for p in d.polygon],
                    **({} if d.plane is None else {"plane": d.plane}),
                    **({} if d.cut is None else {"cut": d.cut}),
                    **({} if d.slab is None else {"slab": d.slab}),
                }
                for d in drawings
            ]
        }
        field = next((s for s in seeds if s.origin == "coarse"), None)
        if field is not None and problem is not None:
            spec["remainder_fineness_fraction"] = field.h / problem.h0
    else:
        spec = {
            "seeds": [
                {
                    "name": s.name,
                    "x": s.xyz[0],
                    "y": s.xyz[1],
                    "z": s.xyz[2],
                    "fineness_fraction": s.h,
                }
                for s in seeds
            ]
        }
    Path(path).write_text(json.dumps(spec, indent=1))


@dataclass
class RandomSeedPartitioner:
    """AB1 lower bound: same cardinality, uniform-random locations/sizes."""

    n_seeds: int = 10
    rng_seed: int = 0

    def propose(self, problem: Problem, post: PostState | None = None, eta2=None) -> list[Seed]:
        del post, eta2
        rng = np.random.default_rng(self.rng_seed)
        b = problem.bbox
        lo = np.array(b[:3], dtype=float)
        hi = np.array(b[3:], dtype=float)
        pts = rng.uniform(lo, hi, size=(self.n_seeds, 3))
        if problem.dim == 2:
            pts[:, 2] = 0.0
        hs = rng.uniform(0.2, 1.0, size=self.n_seeds) * problem.h0
        drawings = [
            halo_drawing(
                f"rand_{i}", pts[i], float(hs[i]), problem,
                radius=0.08 * problem.diameter, phase=0.5 * i,
            )
            for i in range(self.n_seeds)
        ]
        self.last_drawings = drawings
        from .grades import grade_from_frac

        self.last_grades = {
            f"rand_{i}": grade_from_frac(float(hs[i]) / problem.h0)
            for i in range(self.n_seeds)
        }
        return [
            Seed(f"rand_{i}", tuple(pts[i]), h=float(hs[i]), origin="vision")
            for i in range(self.n_seeds)
        ]


def _grades_from_spec(spec: dict, drawings, seeds) -> dict:
    from .grades import grade_from_frac, parse_grade

    grades = {}
    for d in drawings:
        if d.grade is not None:
            grades[d.name] = int(d.grade)
    rem_g = spec.get("remainder_grade") if isinstance(spec, dict) else None
    if rem_g is not None:
        g = parse_grade(rem_g, "remainder_grade")
        grades["field"] = g
        for s in seeds:
            if s.origin == "coarse" or s.name.lower() in REMAINDER_NAMES:
                grades[s.name] = g
    elif isinstance(spec, dict) and spec.get("remainder_fineness_fraction") is not None:
        g = grade_from_frac(spec["remainder_fineness_fraction"])
        grades["field"] = g
        for s in seeds:
            if s.origin == "coarse" or s.name.lower() in REMAINDER_NAMES:
                grades[s.name] = g
    return grades


@dataclass
class CachedDrawingPartitioner:
    """Replay a human / agent drawing JSON.  Propose ignores any solved field."""

    path: str
    revisions: list[str] = field(default_factory=list)
    max_regions: int = 16
    _rev_i: int = 0

    def propose(self, problem: Problem, post: PostState | None = None, eta2=None) -> list[Seed]:
        del post, eta2
        spec = json.loads(Path_read(self.path))
        self.last_drawings = drawings_from_spec(spec, problem, max_regions=self.max_regions)
        seeds = seeds_from_spec(spec, problem, max_seeds=self.max_regions)
        self.last_grades = _grades_from_spec(spec, self.last_drawings, seeds)
        self.last_info = {
            "source": "cached_drawing",
            "path": self.path,
            "n": len(seeds),
            "grades": dict(self.last_grades),
        }
        return seeds

    def revise(self, problem: Problem, observation: dict | None = None):
        """Next vision decision: grades only.  The tool owns the numbers."""

        del problem
        if self._rev_i >= len(self.revisions):
            return None
        path = self.revisions[self._rev_i]
        self._rev_i += 1
        spec = json.loads(Path_read(path))
        grades = dict(getattr(self, "last_grades", {}) or {})
        if spec.get("grades"):
            grades.update({str(k): int(v) for k, v in spec["grades"].items()})
        if spec.get("remainder_grade") is not None:
            from .grades import parse_grade

            g = parse_grade(spec["remainder_grade"], "remainder_grade")
            for name in list(grades):
                if name.lower() in REMAINDER_NAMES:
                    grades[name] = g
            grades.setdefault("field", g)
        self.last_grades = grades
        info = {
            "thought": spec.get("thought") or spec.get("note") or "",
            "grades": dict(grades),
            "stop": bool(spec.get("stop", False)),
            "source": path,
            "remaining": None if not observation else observation.get("remaining"),
        }
        self.last_info = {**getattr(self, "last_info", {}), "revise": info}
        return info


@dataclass
class LLMVisionPartitioner:
    """Multimodal-LLM vision head (OpenAI-compatible chat completions).

    Contract: 3 retries, strict JSON seeds, low temperature; on persistent
    failure fall back to ScriptedVisionPartitioner and record the fallback
    in ``last_info`` (campaign reports the fallback rate; never silently
    invents a six-zone template).
    """

    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    max_seeds: int = 12
    n_retries: int = 3
    timeout_s: float = 180.0
    cache_path: str | None = None
    dump_dir: str | None = None
    fallback: ScriptedVisionPartitioner | None = None

    def __post_init__(self) -> None:
        if self.fallback is None:
            self.fallback = ScriptedVisionPartitioner()
        self.last_info: dict = {}

    def propose(self, problem: Problem, post: PostState | None = None, eta2=None) -> list[Seed]:
        from pathlib import Path

        errors: list[str] = []
        if self.cache_path and Path(self.cache_path).exists():
            spec = json.loads(Path(self.cache_path).read_text())
            seeds = seeds_from_spec(spec, problem, max_seeds=self.max_seeds)
            self.last_drawings = drawings_from_spec(spec, problem, max_regions=self.max_seeds)
            self.last_grades = _grades_from_spec(spec, self.last_drawings, seeds)
            self.last_info = {
                "source": "llm_cache",
                "attempts": 0,
                "n_seeds": len(seeds),
                "cache": self.cache_path,
                "grades": dict(self.last_grades),
            }
            return seeds

        api_base = self.api_base
        api_key = self.api_key
        model = self.model
        if api_base is None or api_key is None or model is None:
            b, k, m = resolve_vlm_endpoint()
            api_base = api_base or b
            api_key = api_key or k
            model = model or m

        if not api_key:
            return self._fallback(problem, post, eta2, ["no_api_key"])

        from ..viz import render_drawing_png

        png = render_drawing_png(problem)
        if self.dump_dir:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.dump_dir) / "drawing.png").write_bytes(png)

        for attempt in range(1, self.n_retries + 1):
            try:
                content = self._chat(png, problem, api_base, api_key, model)
                text = content.strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    if text.lower().startswith("json"):
                        text = text[4:]
                    text = text.strip()
                spec = json.loads(text)
                drawings = drawings_from_spec(spec, problem, max_regions=self.max_seeds)
                seeds = seeds_from_spec(spec, problem, max_seeds=self.max_seeds)
                self.last_drawings = drawings
                self.last_grades = _grades_from_spec(spec, drawings, seeds)
                self.last_info = {
                    "source": "llm",
                    "attempts": attempt,
                    "n_seeds": len(seeds),
                    "model": model,
                    "grades": dict(self.last_grades),
                }
                if self.dump_dir:
                    dump_seed_cache(
                        Path(self.dump_dir) / "llm_seeds.json",
                        seeds, drawings=drawings, problem=problem,
                    )
                    (Path(self.dump_dir) / "llm_raw.txt").write_text(content)
                return seeds
            except Exception as exc:  # noqa: BLE001  -- must not kill the campaign
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        return self._fallback(problem, post, eta2, errors)

    def _fallback(self, problem, post, eta2, errors: list[str]) -> list[Seed]:
        seeds = self.fallback.propose(problem, post, eta2)
        self.last_drawings = list(getattr(self.fallback, "last_drawings", []) or [])
        self.last_grades = dict(getattr(self.fallback, "last_grades", {}) or {})
        self.last_info = {
            "source": "scripted_fallback",
            "attempts": self.n_retries,
            "errors": errors,
            "n_seeds": len(seeds),
            "grades": dict(self.last_grades),
        }
        return seeds

    def _chat(
        self,
        png: bytes,
        problem: Problem,
        api_base: str,
        api_key: str,
        model: str,
    ) -> str:
        payload = {
            "model": model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"结构 '{problem.name}'，包围盒 {problem.bbox}，"
                                f"背景尺寸 h0={problem.h0:.3g}。"
                                f"这是图纸，还没有求解。按图画出不规则区域并给等级 1–5；"
                                f"没画到的体积也要给 remainder_grade。"
                                f"看不清的地方允许拆剖面再画。不要给连续尺寸。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(png).decode(),
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
        }
        req = urllib.request.Request(
            f"{api_base.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"]
