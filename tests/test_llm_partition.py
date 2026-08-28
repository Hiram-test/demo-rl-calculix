import json

import numpy as np
import pytest

from visionamr.geometry import make_bearing_block
from visionamr.vla.partition import (
    LLMVisionPartitioner,
    RandomSeedPartitioner,
    parse_seed_json,
    seeds_from_spec,
)
from visionamr.vla.regions import Seed


def test_seeds_from_spec_clips_and_names():
    problem = make_bearing_block()
    spec = {
        "seeds": [
            {"name": "patch_edge", "x": 200.0, "y": 200.0, "z": 120.0,
             "fineness_fraction": 0.3},
            {"name": "field", "x": -50.0, "y": 10.0, "z": 0.0,
             "fineness_fraction": 0.9},
        ]
    }
    seeds = seeds_from_spec(spec, problem)
    assert len(seeds) == 2
    assert seeds[0].name == "patch_edge"
    assert seeds[1].xyz[0] == pytest.approx(0.0, abs=1e-9)  # clipped to bbox
    assert np.isclose(seeds[0].h, 0.3 * problem.h0)


def test_regions_from_spec_are_drawn_polygons():
    from visionamr.vla.partition import drawings_from_spec

    problem = make_bearing_block()
    spec = {
        "regions": [
            {
                "name": "patch",
                "view": "top",
                "fineness_fraction": 0.25,
                "polygon": [[100, 100], [250, 80], [260, 220], [90, 210]],
            }
        ]
    }
    drawings = drawings_from_spec(spec, problem)
    assert drawings[0].name == "patch"
    assert drawings[0].view == "top"
    assert len(drawings[0].polygon) == 4
    assert np.isclose(drawings[0].h, 0.25 * problem.h0)


def test_parse_fenced_json():
    problem = make_bearing_block()
    content = """```json
{"seeds": [{"name": "a", "x": 1, "y": 2, "z": 3, "fineness_fraction": 0.4}],
 "remainder_fineness_fraction": 0.8}
```"""
    seeds = parse_seed_json(content, problem)
    assert seeds[0].name == "a"
    assert any(s.origin == "coarse" for s in seeds)


def test_regions_spec_assigns_remainder():
    problem = make_bearing_block()
    spec = {
        "regions": [
            {
                "name": "patch",
                "view": "top",
                "fineness_fraction": 0.25,
                "polygon": [[100, 100], [250, 80], [260, 220], [90, 210]],
            }
        ],
        "remainder_fineness_fraction": 0.72,
    }
    seeds = seeds_from_spec(spec, problem)
    field = next(s for s in seeds if s.origin == "coarse")
    assert np.isclose(field.h, 0.72 * problem.h0)


def test_regions_spec_requires_remainder():
    problem = make_bearing_block()
    spec = {
        "regions": [
            {
                "name": "patch",
                "view": "top",
                "fineness_fraction": 0.25,
                "polygon": [[100, 100], [250, 80], [260, 220], [90, 210]],
            }
        ]
    }
    with pytest.raises(ValueError, match="remainder_fineness_fraction"):
        seeds_from_spec(spec, problem)


def test_section_region_needs_cut():
    from visionamr.vla.partition import drawings_from_spec

    problem = make_bearing_block()
    spec = {
        "regions": [
            {
                "name": "column",
                "view": "section",
                "plane": "xz",
                "fineness_fraction": 0.3,
                "polygon": [[150, 40], [300, 50], [280, 120], [160, 110]],
            }
        ]
    }
    with pytest.raises(ValueError, match="cut"):
        drawings_from_spec(spec, problem)


def test_eye_bearing_markup_assigns_varied_remainder():
    from pathlib import Path

    from visionamr.vla.partition import drawings_from_spec

    problem = make_bearing_block()
    spec = json.loads(
        (Path(__file__).resolve().parent / "fixtures" / "bearing_block_eye.json").read_text()
    )
    drawings = drawings_from_spec(spec, problem)
    seeds = seeds_from_spec(spec, problem)
    hs = [round(s.h / problem.h0, 2) for s in seeds]
    assert len(set(hs)) == len(hs)
    assert any(s.origin == "coarse" for s in seeds)
    assert any(d.view == "section" for d in drawings)
    assert np.isclose(next(s for s in seeds if s.origin == "coarse").h, 0.78 * problem.h0)


def test_prompt_requires_remainder_and_allows_section():
    from visionamr.vla.partition import VLM_SYSTEM_PROMPT

    assert "remainder_fineness_fraction" in VLM_SYSTEM_PROMPT
    assert "拆结构剖面" in VLM_SYSTEM_PROMPT
    assert "leave the unpainted bulk coarse" not in VLM_SYSTEM_PROMPT


def test_rejects_empty_seeds():
    problem = make_bearing_block()
    with pytest.raises(ValueError):
        seeds_from_spec({"seeds": []}, problem)


def test_llm_fallback_without_api_key(monkeypatch):
    problem = make_bearing_block()
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    part = LLMVisionPartitioner()

    from visionamr.mesher import Mesh

    # Synthetic tet — this path must not require Gmsh or a field render.
    mesh = Mesh(
        nodes=np.array(
            [[0, 0, 0], [400, 0, 0], [0, 400, 0], [0, 0, 120]], dtype=float
        ),
        cells=np.array([[0, 1, 2, 3]]),
        dim=3,
    )

    class DummyPost:
        def __init__(self):
            self.mesh = mesh
            self.vm_node = np.array([1.0, 4.0, 2.0, 8.0])
            self.vm_elem = np.array([3.75])

    seeds = part.propose(problem, DummyPost(), np.ones(mesh.n_cells))
    assert seeds
    assert part.last_info["source"] == "scripted_fallback"
    assert "no_api_key" in part.last_info["errors"][0]


def test_random_partitioner_reproducible():
    problem = make_bearing_block()
    a = RandomSeedPartitioner(n_seeds=7, rng_seed=3).propose(problem, None, None)
    b = RandomSeedPartitioner(n_seeds=7, rng_seed=3).propose(problem, None, None)
    assert [s.xyz for s in a] == [s.xyz for s in b]
    assert len(a) == 7
