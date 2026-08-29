# Multi-step world-model VLA for a three-dimensional bridge component

## Scientific boundary

This implementation keeps three methods separate.

- `local_prediction` remains an independent baseline and is never imported by the world-model controller.
- Exact element-level Dörfler marking is recomputed after every real CalculiX solve.
- WM-VLA may only add discrete future-hit depth around regions that contain current Dörfler-marked elements.

The controller therefore does not learn or copy the local-prediction size field, its exponent, its allowed-error equation, or any of its labels. The campaign result records `local_prediction_used: false`, and the unit suite rejects imports of `predicted_sizes()`.

## Closed loop

The executable loop is

\[
\text{common probe}
\rightarrow
\text{CalculiX solve}
\rightarrow
\text{ZZ and exact Dörfler marking}
\rightarrow
\text{regional state graph}
\rightarrow
\text{finite-horizon world-model rollout}
\rightarrow
\text{discrete action certification}
\rightarrow
\text{Gmsh remesh}
\rightarrow
\text{next real solve}.
\]

The loop is not limited to two or three solves. `max_solves` controls the real feedback horizon, while `planning_horizon` controls the number of hypothetical future transitions evaluated before each action.

## World state and transition model

Each region stores measured estimator mass, element count, realized size, maximum von Mises stress, material volume, graph adjacency, current Dörfler capture fractions, and the history of prior Dörfler hits. The action is

\[
a_i=d_i,\qquad d_i\in\{0,1,\ldots,d_{\max}\},
\]

where `d_i` means that region `i` receives additional future refinement depth beyond the mandatory current Dörfler action. It is not a continuous size proposed by an LLM.

The transition model predicts regional log changes in estimator mass and element count. A finite-element scaling prior supplies the initial transition, and a deterministic bootstrap ridge ensemble learns the residual between that prior and real Gmsh-plus-CalculiX transitions. Ensemble spread is propagated as a risk upper bound during planning.

The planner uses bounded beam search. Pure Dörfler is explicitly rolled out over the same horizon. A non-zero world action is admitted only when its terminal upper error prediction improves on that Dörfler rollout, its conservative resource prediction remains under the equation cap, and its uncertainty passes the configured gate.

## Dörfler floor

The implementation uses two distinct gates.

At the action level, the deterministic compiler first builds the exact Dörfler nodal target map. Any world action is then applied by taking smaller targets on a local graph support around already marked elements. Certification checks

\[
h^{WM}_j\le h^{D}_j
\]

for every mandatory Dörfler node and forbids all local coarsening. If finiteness, bounds, dominance, uncertainty, or resource certification fails, the action is replaced by exact Dörfler.

At the experiment level, the held-out campaign compares the best Dörfler point reachable with no more sequential solves and no more equations than each WM-VLA point. Any point below that Pareto floor fails the campaign by default.

Target-size domination is a deterministic method invariant. It is not presented as a theorem that an independently regenerated unstructured mesh must have monotonically smaller exact error. The held-out physical gate is retained because Gmsh topology changes and estimator non-monotonicity remain empirical.

## Bridge component

`visionamr.bridge_diaphragm` builds a solid three-dimensional steel-box-girder segment containing:

- top and bottom plates and two webs;
- two longitudinal top-plate ribs;
- a transverse diaphragm with a circular access opening;
- imprinted pin and roller bearing patches;
- an imprinted offset wheel-load patch.

The case produces interacting load-edge, rib-junction, diaphragm-junction, access-rim, and bearing hotspots. Geometry, wheel footprint, wheel offset, hole radius, and pressure are varied between independent training and held-out test cases without changing the component topology.

## Deterministic parameter contract

The world model outputs only integral regional depths. Numerical target sizes are produced by repository tools from the current mesh, the exact Dörfler set, the fixed refinement factor, mesh-size bounds, and graph-halo depth. Every action creates an `ActionReceipt` containing action and target hashes, support size, equation estimate, and all certification results. This boundary is directly suitable for exposure as an MCP tool; MCP or another caller cannot inject arbitrary continuous sizes into the solver path.

## Commands

Focused unit gates:

```bash
python -m pytest -q tests/test_world_controller.py
```

Coarse real Gmsh-CalculiX smoke gate:

```bash
python scripts/run_world_model_vla_bridge.py --smoke --no-reference --train-cases 1 --test-cases 1 --max-solves 3 --horizon 3 --n-eq-cap 60000 --output artifacts/world_model_vla_bridge_smoke
```

Full held-out bridge campaign:

```bash
python scripts/run_world_model_vla_bridge.py --train-cases 4 --test-cases 3 --max-solves 8 --horizon 4 --n-eq-cap 180000 --output artifacts/world_model_vla_bridge
```

The full campaign exits with status `2` if any held-out instance violates the Dörfler performance floor. `--allow-weaker` is available only for exploratory diagnostics and is not used by the verification workflow.

## Interpretation of a successful result

A valid world-model advantage requires all of the following:

1. the Dörfler floor passes on every held-out case;
2. at least one non-zero delegated action is actually executed;
3. the final best error improves by more than two percent or the same final Dörfler quality is reached with at least one fewer sequential physical solve;
4. all actions retain parameter receipts and the campaign records that local prediction was not used.

If only the first condition passes, the method is a safe Dörfler-equivalent fallback, not evidence that the world model has added value.
