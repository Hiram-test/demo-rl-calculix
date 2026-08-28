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
    assert seeds[1].xyz[0] == 0.0  # clipped to bbox
    assert np.isclose(seeds[0].h, 0.3 * problem.h0)


def test_parse_fenced_json():
    problem = make_bearing_block()
    content = """```json
{"seeds": [{"name": "a", "x": 1, "y": 2, "z": 3, "fineness_fraction": 0.4}]}
```"""
    seeds = parse_seed_json(content, problem)
    assert seeds[0].name == "a"


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

    class DummyPost:
        mesh = None
        vm_node = None
        vm_elem = None

    # Scripted fallback needs a real mesh; just check _fallback path with no key
    # via propose would need a mesh.  Direct _fallback:
    from visionamr.mesher import generate_uniform
    from visionamr.fem_post import compute_post

    mesh = generate_uniform(problem, problem.h0)
    u = np.zeros((mesh.n_nodes, 3))
    post = compute_post(mesh, problem, u)
    eta2 = np.ones(mesh.n_cells)
    seeds = part.propose(problem, post, eta2)
    assert seeds
    assert part.last_info["source"] == "scripted_fallback"
    assert "no_api_key" in part.last_info["errors"][0]


def test_random_partitioner_reproducible():
    problem = make_bearing_block()
    a = RandomSeedPartitioner(n_seeds=7, rng_seed=3).propose(problem, None, None)
    b = RandomSeedPartitioner(n_seeds=7, rng_seed=3).propose(problem, None, None)
    assert [s.xyz for s in a] == [s.xyz for s in b]
    assert len(a) == 7
