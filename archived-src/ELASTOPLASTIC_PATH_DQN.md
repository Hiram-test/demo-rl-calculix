# Elastoplastic load-path mesh DQN

This backend is a research redesign of the original static linear-elastic mesh demo.
It does not claim that adding `*PLASTIC` alone makes reinforcement learning useful.
The sequential decision problem is the **entire monotonic loading path** under a
fixed mesh-resource budget.

## Why the static elastic design was insufficient

The old problem could usually be reduced to ranking the current stress/error
hotspots.  It also used fixed geometric cells as graph nodes, so changing the
finite-element density did not change graph topology.  Earlier state schemas
omitted the actual mesh-density state and, in the first CalculiX prototype,
exposed fine-reference local errors to the policy.  That is oracle leakage.

## Physical benchmark

- thin plate with an eccentric circular hole;
- `CPS3` plane-stress elements;
- small-strain J2 plasticity with isotropic hardening;
- left edge constrained in `x`, one left node anchored in `y`;
- right edge under prescribed monotonic `x` displacement;
- 20 physical load levels;
- solver stress, reaction and `PEEQ` are read from CalculiX output.

The eccentric hole avoids a trivially symmetric twin-hotspot problem.  It does
not by itself prove an RL advantage; hotspot and plastic-front heuristic
baselines remain mandatory.

## MDP

One episode spans the load path.  At each load level the agent selects one
global `(control patch, action)` candidate:

- `refine`;
- `coarsen`;
- `keep` (advance load without forcing a mesh mutation).

`keep` is represented on one canonical active patch so the Q surface does not
contain dozens of physically duplicate no-op actions.

The state contains current stress/yield ratios, area-weighted `PEEQ`, plastic
area fraction, plastic work, displacement, patch geometry, mesh size, mesh
gradation, load fraction, tangent stiffness, resource budget and action history.
Fine-reference errors are used only for reward/evaluation and never enter the
policy state.

The reward combines path quantities:

- reaction-force error;
- plastic-work error;
- patch `PEEQ` profile error;
- plastic-front overlap error;
- element budget;
- mesh gradation;
- ineffective mesh mutation penalty.

Three-step returns are available through `--n-step 3` so mesh choices can receive
credit from later plastic evolution rather than only the next load point.

## Remeshing and history

After a mesh action, the monotonic history from zero to the new load level is
recomputed on the new mesh.  This is deliberately expensive but avoids an
unverified projection of plastic internal variables.  A production method
should implement conservative transfer of plastic strain/hardening variables
and verify dissipation consistency.

## Run locally

```bash
python rl_main_local.py \
  --backend calculix-plastic \
  --mode train \
  --gmsh-cmd gmsh \
  --ccx-cmd ccx \
  --plastic-plate-config examples/calculix_plastic_plate_eccentric.json \
  --goal-file examples/goal_plastic.json \
  --max-episodes 100 \
  --max-steps 25 \
  --baseline-mesh-size 0.06 \
  --n-step 3 \
  --eval-frequency 10
```

## Scientific limitations

This fixed benchmark is a prototype, not sufficient evidence for a paper.  A
publication-quality study should train across hole position/radius, yield
stress, hardening and load level, then test unseen parameter combinations.  It
must compare against residual/goal-oriented AMR, stress hotspot, `PEEQ` hotspot,
plastic-front tracking, uniform meshes and wall-clock/resource budgets.
