# Local Abaqus and CalculiX backends

The V2 graph DQN now has two local finite-element execution paths.  Both use the
same `StateAwareDQNAgent`, the same global `(cell, action)` Bellman target, the
same goal JSON contract, and the same checkpoint format.  Solver-specific work
is isolated behind an environment implementation.

## Files

- `rl_main_local.py`: unified `preflight`, `solve`, and `train` entry point.
- `state_aware_env.py`: existing Abaqus environment.
- `calculix_backend.py`: Gmsh mesh generator, CalculiX deck writer, FRD parser,
  local feature reconstruction, reward, and state/action-mask implementation.
- `mesh_goal.py`: backend-independent structured objective schema.
- `examples/calculix_plate.json`: small plate-with-hole benchmark.
- `examples/goal_local.json`: goal-conditioning example.
- `scripts/run_*_local.{sh,ps1}`: Linux/macOS and Windows launchers.

The repository does **not** redistribute Abaqus, Gmsh, or CalculiX binaries.
Point the launchers at software installed on the local machine.

## Python environment

Use a normal Python environment, not Abaqus Python, because the DQN requires a
recent PyTorch.  The Python process launches Abaqus or CalculiX as external
commands.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-state-aware.txt
```

`torch-geometric` remains optional; the project contains a native PyTorch GCN
fallback.

## CalculiX backend

Install:

1. Gmsh with a callable `gmsh` executable.
2. CalculiX CrunchiX with a callable `ccx` executable.  Some distributions name
   the executable `ccx_2.22`, `ccx_2.23`, or similarly.

Set explicit paths when the commands are not on `PATH`:

```bash
export GMSH_CMD=/opt/gmsh/bin/gmsh
export CCX_CMD=/opt/CalculiX/ccx_2.23
```

```powershell
$env:GMSH_CMD = "C:\Tools\gmsh\gmsh.exe"
$env:CCX_CMD = "C:\Tools\CalculiX\ccx.exe"
```

Run the checks in increasing order:

```bash
./scripts/run_calculix_local.sh preflight
./scripts/run_calculix_local.sh solve
./scripts/run_calculix_local.sh train --max-episodes 3 --max-steps 10 --debug
```

PowerShell:

```powershell
.\scripts\run_calculix_local.ps1 -Mode preflight
.\scripts\run_calculix_local.ps1 -Mode solve
.\scripts\run_calculix_local.ps1 -Mode train --max-episodes 3 --max-steps 10 --debug
```

The CalculiX path performs the following operations for every mesh decision:

1. write `model.geo` with one background mesh-size box per virtual cell;
2. invoke Gmsh and parse its ASCII MSH 2.2 triangle mesh;
3. write a CalculiX `CPE3` plane-strain input deck;
4. invoke `ccx -i model`;
5. parse the last ASCII `DISP` dataset in `model.frd`;
6. reconstruct triangle strain, stress, von Mises stress, and strain energy;
7. aggregate those values to the fixed virtual-cell graph used by the DQN.

Each step keeps its generated `.geo`, `.msh`, `.inp`, `.frd`, and command logs
under `simulations_local/calculix/` for inspection.

### Change the CalculiX example

Edit `examples/calculix_plate.json`.  Geometry, material, load, hole size, and
virtual-cell grid are configurable.  The first implementation deliberately uses
a small 2-D problem so solver and RL defects are easy to distinguish.

## Abaqus backend

Abaqus continues to use `DEMO.cae` and the archived extraction scripts.  Set the
launcher command explicitly when necessary:

```bash
export ABAQUS_CMD=/opt/SIMULIA/Commands/abaqus
./scripts/run_abaqus_local.sh preflight
./scripts/run_abaqus_local.sh solve
./scripts/run_abaqus_local.sh train --max-episodes 3 --max-steps 10 --debug
```

PowerShell:

```powershell
$env:ABAQUS_CMD = "C:\SIMULIA\Commands\abaqus.bat"
.\scripts\run_abaqus_local.ps1 -Mode preflight
.\scripts\run_abaqus_local.ps1 -Mode solve
.\scripts\run_abaqus_local.ps1 -Mode train --max-episodes 3 --max-steps 10 --debug
```

Abaqus is launched externally from the normal Python process.  The CAE helper
scripts still run through Abaqus where required, while PyTorch remains in the
normal Python environment.

## Direct CLI examples

```bash
python rl_main_local.py --backend calculix --mode solve \
  --gmsh-cmd gmsh --ccx-cmd ccx \
  --plate-config examples/calculix_plate.json

python rl_main_local.py --backend abaqus --mode solve \
  --abaqus-cmd abaqus --template-cae-file DEMO.cae
```

Useful backend-independent modes:

- `preflight`: check executable paths and required input files without solving.
- `solve`: calculate/cache the fine reference, then run one initial coarse solve
  and print the state dimensions and relative error.
- `train`: run the full V2 DQN loop.

## Tests

Tests do not require commercial software or a local CalculiX installation:

```bash
python -m unittest discover -s tests -p "test_state_aware_*.py" -v
python -m unittest discover -s tests -p "test_calculix_*.py" -v
```

The CalculiX tests include a fake-command end-to-end path that creates a Gmsh
mesh and FRD displacement fixture, then exercises deck generation and numerical
postprocessing.  CI also runs a real Gmsh/CalculiX smoke solve when the Ubuntu
packages are available.

## Checkpoint separation

Defaults are backend-specific:

- `checkpoints_local/abaqus/`
- `checkpoints_local/calculix/`

A checkpoint records its backend and cannot be resumed under the other solver.
The two backends have different node-feature dimensions, so sharing raw model
weights would be invalid even though the DQN architecture is common.
