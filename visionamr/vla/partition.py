"""Vision heads: propose named partition seeds from a rendered view.

The heads never emit geometric primitives; they point at structural
locations ("wheel patch edge", "inner strip line", "re-entrant corner",
"calm mid-field") the way a human marks up a plot, and the geodesic
partition in ``regions.Partition`` grows organic region shapes from
those seeds.

* ``ScriptedVisionPartitioner`` -- deterministic stand-in: von Mises
  peaks (non-max suppression) + structural anchors + coarse field seeds.
* ``LLMVisionPartitioner``      -- multimodal LLM on rendered views
  (orthographic views in 3-D), strict JSON seed schema.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..fem_post import PostState
from ..geometry import Problem
from ..indicators import zz_indicator  # noqa: F401  (kept for head extensions)
from .drawing import (
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

    Mimics how an engineer marks a plot: a non-box stroke around each
    hotspot and each drawing feature, plus a coarse field for leftover
    volume.  Sizes are the eye's fineness, not an equidistribution paint.
    """

    peak_quantile: float = 0.60
    max_hot: int = 5
    min_sep_frac: float = 0.10
    anchor_kinds: tuple[str, ...] = ("load", "support", "corner", "hole")
    anchor_sep_frac: float = 0.06
    n_field: int = 3
    field_sep_frac: float = 0.22
    h_hot: float = 0.30       # initial fineness of the hottest seed (x h0)
    h_mild: float = 0.60
    h_anchor: float = 0.40

    def propose(self, problem: Problem, post: PostState, eta2: np.ndarray) -> list[Seed]:
        mesh = post.mesh
        vm = post.vm_node
        diam = problem.diameter
        thr = np.quantile(vm, self.peak_quantile)

        e = mesh.edges
        is_peak = np.ones(mesh.n_nodes, dtype=bool)
        a, b = e[:, 0], e[:, 1]
        lower_a = vm[a] < vm[b]
        np.logical_and.at(is_peak, a[lower_a], False)
        lower_b = vm[b] < vm[a]
        np.logical_and.at(is_peak, b[lower_b], False)
        cand = np.nonzero(is_peak & (vm >= thr))[0]
        cand = cand[np.argsort(vm[cand])[::-1]]

        picked: list[int] = []
        for n in cand:
            if len(picked) >= self.max_hot:
                break
            if all(
                np.linalg.norm(mesh.nodes[n] - mesh.nodes[p]) >= self.min_sep_frac * diam
                for p in picked
            ):
                picked.append(int(n))

        vm_gmax = max(float(vm.max()), 1e-30)
        drawings = []
        used: set[str] = set()
        seed_pts: list[np.ndarray] = []

        for rank, n in enumerate(picked):
            pt = mesh.nodes[n]
            name = self._name(problem, pt, rank, used)
            used.add(name)
            intensity = float(vm[n]) / vm_gmax
            frac = self.h_mild - (self.h_mild - self.h_hot) * intensity
            view = "top"
            near = np.linalg.norm(mesh.nodes - pt, axis=1) < 0.08 * diam
            pts = mesh.nodes[near] if near.any() else pt.reshape(1, 3)
            ax0, ax1 = VIEW_AXES[view]
            poly = irregular_from_points(pts[:, [ax0, ax1]], 0.05 * diam)
            drawings.append(
                DrawnRegion(name, float(frac * problem.h0), view, poly_tuple(poly))
            )
            seed_pts.append(pt)

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
            near = np.linalg.norm(mesh.nodes - f.xyz, axis=1) < rad
            pts = mesh.nodes[near] if near.sum() >= 3 else np.vstack(
                [f.xyz, mesh.nodes[np.argmin(np.linalg.norm(mesh.nodes - f.xyz, axis=1))]]
            )
            ax0, ax1 = VIEW_AXES[view]
            poly = irregular_from_points(pts[:, [ax0, ax1]], rad)
            drawings.append(
                DrawnRegion(
                    name, float(self.h_anchor * problem.h0), view, poly_tuple(poly)
                )
            )
            seed_pts.append(f.xyz)

        seeds = [
            Seed(d.name, drawing_centroid_xyz(d, problem), d.h, d.origin) for d in drawings
        ]
        field_pt = np.mean(mesh.nodes, axis=0)
        if problem.dim == 2:
            field_pt[2] = 0.0
        seeds.append(Seed("field", tuple(field_pt), float(problem.h0), origin="coarse"))
        if not drawings:
            drawings = [
                halo_drawing("field", field_pt, problem.h0, problem, origin="coarse")
            ]
            seeds = [Seed("field", tuple(field_pt), float(problem.h0), origin="coarse")]
        self.last_drawings = drawings
        return seeds

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


VLM_SYSTEM_PROMPT = """You are a senior finite-element meshing engineer.
You see rendered views of a structure with its von Mises stress field
(orthographic projections for 3-D parts).  Draw irregular regions the
way you would with a pen -- never axis-aligned boxes, never a fixed
slab template.  First outline each region as a polygon on one view
(top = x-y, front = x-z, side = y-z).  Then assign that region a
single target element size as a fraction of the background size.
Reply with strict JSON and nothing else:
{"regions": [{"name": "<structural name>", "view": "top"|"front"|"side",
"polygon": [[u,v], ...],
"fineness_fraction": <hot 0.15-0.5, calm 0.8-1.0>}, ...]}
Polygons need at least 3 vertices in model units on that view.
Draw both hot regions and leave the unpainted bulk coarse.
Use as many or as few regions as the picture demands."""


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


def seeds_from_spec(spec: dict, problem: Problem, max_seeds: int = 12) -> list[Seed]:
    """Validate VLM JSON into seeds.  Drawn regions preferred; old seed lists still parse."""

    drawings = markup_from_spec(spec, problem, max_regions=max_seeds)
    return [Seed(d.name, drawing_centroid_xyz(d, problem), d.h, d.origin) for d in drawings]


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
                }
                for d in drawings
            ]
        }
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

    def propose(self, problem: Problem, post: PostState, eta2: np.ndarray) -> list[Seed]:
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
        return [
            Seed(f"rand_{i}", tuple(pts[i]), h=float(hs[i]), origin="vision")
            for i in range(self.n_seeds)
        ]


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

    def propose(self, problem: Problem, post: PostState, eta2: np.ndarray) -> list[Seed]:
        from pathlib import Path

        errors: list[str] = []
        if self.cache_path and Path(self.cache_path).exists():
            seeds = load_seed_cache(self.cache_path, problem)
            self.last_drawings = load_drawings_cache(self.cache_path, problem)
            self.last_info = {
                "source": "llm_cache",
                "attempts": 0,
                "n_seeds": len(seeds),
                "cache": self.cache_path,
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

        from ..viz import render_field_png

        png = render_field_png(problem, post)
        if self.dump_dir:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.dump_dir) / "probe_field.png").write_bytes(png)

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
                seeds = [
                    Seed(d.name, drawing_centroid_xyz(d, problem), d.h, d.origin)
                    for d in drawings
                ]
                self.last_drawings = drawings
                self.last_info = {
                    "source": "llm",
                    "attempts": attempt,
                    "n_seeds": len(seeds),
                    "model": model,
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
        self.last_info = {
            "source": "scripted_fallback",
            "attempts": self.n_retries,
            "errors": errors,
            "n_seeds": len(seeds),
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
                                f"Structure '{problem.name}', bbox {problem.bbox}, "
                                f"background size h0={problem.h0:.3g}. "
                                f"Draw irregular regions, then assign each a size."
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
