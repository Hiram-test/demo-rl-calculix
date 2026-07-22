# State-aware graph DQN V2

This directory keeps the archived V1 implementation unchanged and adds a V2
training path for the repeated-cell/action problem.

## Root causes fixed

1. **Policy/target mismatch.** V1 selects one global `(cell, action)` candidate,
   but its Bellman target takes `max(action)` only on the same cell. V2 uses
   Double DQN with `max_(all valid cells, all valid actions)` in the next state.
2. **Missing dynamic mesh state.** V1 never passes the actual per-cell mesh size
   to the Q network. V2 appends mesh size, bound position, previous action,
   action validity, action recency and local reference-energy discrepancy.
3. **Empty global state.** V1 always passes a `[1, 0]` global tensor. V2 includes
   resource use, remaining budget, step progress, ALLSE error/progress, mesh
   statistics, failure state and a structured engineering objective.
4. **Local-only Q head.** V2 combines each GCN node embedding with graph mean/max
   pooling and explicit global/task context in a dueling Q architecture.
5. **Invalid repeated actions.** Refine/coarsen bounds are hard-masked. A pair
   that just rolled back or produced no mesh change is blocked for the next
   decision so deterministic argmax cannot loop on it.
6. **Mutable/obsolete replay semantics.** V2 stores detached CPU snapshots and
   permits different graph sizes between `state` and `next_state`.
7. **Dynamic graph safety.** PyG `GCNConv` is explicitly created with
   `cached=False`; a dependency-free native PyTorch GCN is provided as fallback.

## Files

- `state_aware_dqn_agent.py`: V2 graph state, dueling Double DQN and replay.
- `state_aware_env.py`: state/action-mask wrapper for `AbaqusEnv`.
- `rl_main_state_aware.py`: independent V2 training entry point.
- `goal_example.json`: structured objective contract; this is the future LLM
  planner boundary.
- `tests/`: fast tests that do not launch Abaqus.

## Run tests

```bash
python -m pip install -r requirements-state-aware.txt
python -m unittest discover -s tests -p "test_state_aware_*.py" -v
```

## Train

Run from the archived source directory so the existing Abaqus helper scripts are
found by relative path:

```bash
python rl_main_state_aware.py \
  --template-cae-file DEMO.cae \
  --goal-file goal_example.json \
  --max-episodes 100 \
  --max-steps 100 \
  --debug
```

Use `--sample-goals` to train one goal-conditioned policy over varying accuracy,
resource and localization priorities. V2 writes to `simulations_v2/` and
`checkpoints_v2/`, so V1 experiments remain intact.

## Checkpoint compatibility

V1 checkpoints and replay buffers are intentionally not loaded. Their target
semantics are inconsistent with the global action policy, and the V2 input and
network schemas are different. Keep V1 as an experimental baseline and begin a
clean V2 run.

## Important modeling note

In the archived model, graph nodes are Abaqus **geometric cells**, not generated
finite elements. Refining the mesh inside a geometric cell normally changes the
mesh density and element count but does not change the cell-adjacency topology.
Therefore topology alone cannot signal a state change. V2 explicitly encodes the
mesh-density/resource state. A future element-level or hierarchical-patch graph
would be a separate modeling change, not a bug fix.
