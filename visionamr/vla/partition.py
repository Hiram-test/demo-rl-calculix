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
    """Peak + structural-anchor + field seeds from the probe response.

    Mimics how an engineer marks a plot: peaks of the response field,
    the structurally declared detail locations (load patch edges,
    support strips, corners, holes -- these exist on the drawing even
    when the coarse probe underresolves them), and a few calm-bulk
    points that should stay coarse.
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

        # nodal local maxima over the mesh edge graph
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
        seeds: list[Seed] = []
        used: set[str] = set()
        for rank, n in enumerate(picked):
            pt = mesh.nodes[n]
            name = self._name(problem, pt, rank, used)
            used.add(name)
            intensity = float(vm[n]) / vm_gmax
            frac = self.h_mild - (self.h_mild - self.h_hot) * intensity
            seeds.append(Seed(name, tuple(pt), h=float(frac * problem.h0)))

        # structural anchors from the drawing (deduplicated against peaks)
        for f in problem.features:
            if f.kind not in self.anchor_kinds:
                continue
            if any(
                np.linalg.norm(f.xyz - s.point()) < self.anchor_sep_frac * diam
                for s in seeds
            ):
                continue
            name = f"{f.name}_zone"
            if name in used:
                name = f"{name}_a"
            used.add(name)
            seeds.append(
                Seed(name, tuple(f.xyz), h=float(self.h_anchor * problem.h0))
            )

        # coarse "field" seeds so the calm bulk forms its own coarse regions
        low = post.vm_elem <= np.quantile(post.vm_elem, 0.5)
        pts = mesh.centroids[low]
        if len(seeds) > 0 and len(pts) > 0:
            seed_pts = np.array([s.point() for s in seeds])
            d_to_hot = np.min(
                np.linalg.norm(pts[:, None, :] - seed_pts[None, :, :], axis=2), axis=1
            )
            pts = pts[d_to_hot > self.field_sep_frac * diam]
        for j, idx in enumerate(
            _farthest_point_seeds(pts, self.n_field, self.field_sep_frac * diam)
        ):
            seeds.append(
                Seed(f"field_{j}", tuple(pts[idx]), h=float(problem.h0), origin="coarse")
            )
        if not seeds:
            seeds = [Seed("domain", tuple(np.mean(mesh.nodes, axis=0)), h=problem.h0)]
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
(orthographic projections for 3-D parts).  Mark the structural locations
that control the discretization error the way you would with a pen:
stress concentrations, load-patch edges, support/reaction lines,
re-entrant corners -- and also a few points in the calm bulk that should
stay coarse.  Each mark becomes the seed of one mesh region grown
geodesically around it (regions are organic shapes, never boxes).
Reply with strict JSON:
{"seeds": [{"name": "<structural name>", "x": .., "y": .., "z": ..,
"fineness_fraction": <target element size as a fraction of the background
size, hot 0.15-0.5, calm bulk 0.8-1.0>}, ...]}
Use as many or as few seeds as the picture demands; never a fixed
template.  Coordinates are in the model units printed on the axes."""


@dataclass
class LLMVisionPartitioner:
    """Multimodal-LLM vision head (OpenAI-compatible chat completions)."""

    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    max_seeds: int = 12

    def propose(self, problem: Problem, post: PostState, eta2: np.ndarray) -> list[Seed]:
        from ..viz import render_field_png

        png = render_field_png(problem, post)
        payload = {
            "model": self.model or os.environ.get("VLM_MODEL", "gpt-4o"),
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
                                f"background size h0={problem.h0:.3g}. Mark the seeds."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(png).decode()
                            },
                        },
                    ],
                },
            ],
        }
        api_base = self.api_base or os.environ.get(
            "VLM_API_BASE", "https://api.openai.com/v1"
        )
        api_key = self.api_key or os.environ.get("VLM_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        if not api_key:
            raise RuntimeError(
                "LLMVisionPartitioner needs VLM_API_KEY (or OPENAI_API_KEY); "
                "use ScriptedVisionPartitioner for offline runs"
            )
        req = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read())
        content = body["choices"][0]["message"]["content"]
        spec = json.loads(content)
        lo = np.array(problem.bbox[:3])
        hi = np.array(problem.bbox[3:])
        seeds: list[Seed] = []
        for i, s in enumerate(spec.get("seeds", [])[: self.max_seeds]):
            frac = float(np.clip(s.get("fineness_fraction", 0.4), 0.1, 1.0))
            pt = np.clip(
                np.array([float(s["x"]), float(s.get("y", 0.0)), float(s.get("z", 0.0))]),
                lo,
                hi,
            )
            seeds.append(
                Seed(
                    str(s.get("name", f"llm_seed_{i}"))[:48],
                    tuple(pt),
                    h=float(frac * problem.h0),
                )
            )
        if not seeds:
            raise RuntimeError("VLM returned no seeds")
        return seeds
