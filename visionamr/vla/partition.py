"""Vision partitioners: turn a rendered view of the structure + response
field into named regions with delegate sizes.

Two implementations:

* ``ScriptedVisionPartitioner`` -- deterministic hotspot clustering on the
  von Mises field of the probe solve.  Serves as the reproducible stand-in
  and the "vision quality" ablation lower bound.
* ``LLMVisionPartitioner`` -- renders a PNG and asks a multimodal LLM for
  regions in strict JSON (the paper's actual vision head).  Requires
  VLM_API_KEY / VLM_MODEL (OpenAI-compatible chat completions endpoint).

Both return regions whose count follows the view, never a fixed template.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..fem_post import PostState
from ..geometry import Problem
from ..sizefield import Region


@dataclass
class ScriptedVisionPartitioner:
    """Hotspot clustering: quantile threshold + connected components.

    For the ``nested_top`` hottest clusters an inner box is emitted around
    the peak (a vision-style nested frame); each box still carries exactly
    one size, so nesting realizes region-level grading without breaking
    the one-size-per-region contract.
    """

    quantile: float = 0.90
    min_nodes: int = 3
    max_regions: int = 6
    pad_factor: float = 1.6
    h_hot: float = 0.22       # hottest region delegate size, fraction of h0
    h_mild: float = 0.55      # mildest region delegate size, fraction of h0
    nested_top: int = 1
    nested_shrink: float = 0.38
    nested_h_factor: float = 0.45

    def partition(
        self, problem: Problem, post: PostState, eta2: np.ndarray
    ) -> list[Region]:
        mesh = post.mesh
        vm = post.vm_node
        thr = np.quantile(vm, self.quantile)
        hot = vm >= thr
        if hot.sum() < self.min_nodes:
            hot = vm >= np.quantile(vm, 0.8)

        # connected components of hot nodes over the mesh edge graph
        e = mesh.edges
        keep = hot[e[:, 0]] & hot[e[:, 1]]
        ek = e[keep]
        n = mesh.n_nodes
        g = coo_matrix(
            (np.ones(len(ek)), (ek[:, 0], ek[:, 1])), shape=(n, n)
        )
        n_comp, labels = connected_components(g, directed=False)
        regions: list[Region] = []
        comp_stats = []
        for c in range(n_comp):
            idx = np.nonzero((labels == c) & hot)[0]
            if len(idx) < self.min_nodes:
                continue
            comp_stats.append((float(vm[idx].max()), idx))
        comp_stats.sort(key=lambda s: -s[0])
        vm_gmax = max(float(vm.max()), 1e-30)

        bx0, by0, bx1, by1 = problem.bbox
        for rank, (vmax, idx) in enumerate(comp_stats[: self.max_regions]):
            pts = mesh.nodes[idx]
            pad = self.pad_factor * float(np.mean(mesh.node_sizes[idx]))
            xmin, ymin = pts.min(axis=0) - pad
            xmax, ymax = pts.max(axis=0) + pad
            xmin, xmax = max(xmin, bx0), min(xmax, bx1)
            ymin, ymax = max(ymin, by0), min(ymax, by1)
            intensity = vmax / vm_gmax
            frac = self.h_mild - (self.h_mild - self.h_hot) * intensity
            name = self._name_region(problem, 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), rank)
            outer = Region(name, xmin, ymin, xmax, ymax, h=float(frac * problem.h0))
            regions.append(outer)
            if rank < self.nested_top:
                peak = mesh.nodes[idx[np.argmax(vm[idx])]]
                half = 0.5 * self.nested_shrink * max(xmax - xmin, ymax - ymin)
                regions.append(
                    Region(
                        f"{name}_core",
                        max(peak[0] - half, bx0),
                        max(peak[1] - half, by0),
                        min(peak[0] + half, bx1),
                        min(peak[1] + half, by1),
                        h=float(self.nested_h_factor * outer.h),
                    )
                )
        return regions

    @staticmethod
    def _name_region(problem: Problem, cx: float, cy: float, rank: int) -> str:
        best, dist = None, np.inf
        for f in problem.features:
            d = float(np.hypot(cx - f.x, cy - f.y))
            if d < dist:
                best, dist = f, d
        xmin, ymin, xmax, ymax = problem.bbox
        near = 0.25 * float(np.hypot(xmax - xmin, ymax - ymin))
        if best is not None and dist < near:
            return f"{best.name}_zone"
        return f"hotspot_{rank}"


VLM_SYSTEM_PROMPT = """You are a finite-element meshing engineer. You see a
plane-stress structure with its von Mises stress field. Identify the
structural regions that need local mesh refinement (stress concentrations,
re-entrant corners, load introductions, support reactions). Reply with a
strict JSON object:
{"regions": [{"name": "<structural name>", "xmin": .., "ymin": .., "xmax": ..,
"ymax": .., "size_fraction": <target element size as a fraction of the
background size, in (0.1, 0.8)>}, ...]}
Use as many or as few regions as the picture demands; never emit a fixed
template. Coordinates are in the model units printed on the axes."""


@dataclass
class LLMVisionPartitioner:
    """Multimodal-LLM vision head (OpenAI-compatible chat completions)."""

    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    max_regions: int = 8

    def partition(
        self, problem: Problem, post: PostState, eta2: np.ndarray
    ) -> list[Region]:
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
                                f"Structure '{problem.name}', bounding box {problem.bbox}. "
                                f"Background mesh size h0={problem.h0:.3g}. Partition it."
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
        regions: list[Region] = []
        bx0, by0, bx1, by1 = problem.bbox
        for i, r in enumerate(spec.get("regions", [])[: self.max_regions]):
            frac = float(np.clip(r.get("size_fraction", 0.4), 0.1, 0.8))
            regions.append(
                Region(
                    str(r.get("name", f"llm_region_{i}"))[:48],
                    max(float(r["xmin"]), bx0),
                    max(float(r["ymin"]), by0),
                    min(float(r["xmax"]), bx1),
                    min(float(r["ymax"]), by1),
                    h=frac * problem.h0,
                )
            )
        if not regions:
            raise RuntimeError("VLM returned no regions")
        return regions
