#!/usr/bin/env python3
"""Run EXPERIMENT_PLAN.md steps S0–S9.

Examples:
  python scripts/run_campaign.py --steps S0
  python scripts/run_campaign.py --steps S0,S1,S2,S3 --instances canonical
  python scripts/run_campaign.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")

from visionamr.campaign import FAMILIES_2D, FAMILIES_3D, run_steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=None, help="comma-separated, e.g. S0,S1,S3")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--families", default="bearing_block,deck_panel")
    ap.add_argument("--instances", default=None,
                    help="comma-separated keys (default: canonical+test_*)")
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--include-train", action="store_true")
    ap.add_argument("--learn-families", default="lbracket,plate_holes")
    ap.add_argument("--n-experts", type=int, default=None)
    ap.add_argument("--rl-episodes", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--rl-seeds", default=None,
                    help="comma-separated seed indices (overrides --n-seeds range)")
    args = ap.parse_args()

    if args.all:
        steps = [f"S{i}" for i in range(10)]
    elif args.steps:
        steps = [s.strip().upper() for s in args.steps.split(",") if s.strip()]
    else:
        ap.error("give --steps or --all")

    families = tuple(s.strip() for s in args.families.split(",") if s.strip())
    instances = (
        [s.strip() for s in args.instances.split(",") if s.strip()]
        if args.instances else None
    )
    learn = tuple(s.strip() for s in args.learn_families.split(",") if s.strip())
    run_steps(
        steps,
        families=families,
        instance_keys=instances,
        include_train=args.include_train,
        pilot_only=args.pilot_only,
        no_llm=args.no_llm,
        learn_families=learn,
        n_experts=args.n_experts,
        rl_episodes=args.rl_episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
