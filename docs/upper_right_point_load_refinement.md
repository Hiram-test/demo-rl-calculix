# Upper-right point-load mesh-refinement study

This report records a solver-generated, focused refinement study for the
upper-right corner of the controlled plane-stress cantilever benchmark.

## Provenance

- GitHub Actions workflow: `Mesh Need Demo`
- Successful workflow run: `30214005278`
- Head commit used by the runner: `bbea30a3d11a7918ed1da464b499546332ac7c92`
- Solver: CalculiX `ccx` from the Ubuntu `calculix-ccx` package
- Element: CPS4
- Geometry: 100 mm by 20 mm, thickness 1 mm
- Boundary condition: left edge fixed in x and y
- Load: -1000 N vertical nodal load at the upper-right corner

All values below were parsed from CalculiX `.dat` files generated on the
GitHub-hosted runner. No response values were pre-filled.

## Results

| Elements | Nominal size (mm) | Global peak stress (MPa) | Upper-right corner-element peak (MPa) | Nearest Gauss-point distance (mm) | Corner peak × distance | Fixed 5–10 mm ring mean stress (MPa) | Fixed 10×10 mm patch mean stress (MPa) | Remote probe U2 (mm) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 5.000 | 1265.047 | 411.331 | 1.494292 | 614.648 | 148.320 | 180.414 | -0.747223 |
| 320 | 2.500 | 1449.331 | 817.165 | 0.747146 | 610.542 | 153.202 | 190.902 | -0.767267 |
| 1280 | 1.250 | 1636.591 | 1621.642 | 0.373573 | 605.802 | 154.900 | 195.769 | -0.772870 |
| 5120 | 0.625 | 3182.778 | 3182.778 | 0.186787 | 594.500 | 156.631 | 198.022 | -0.774437 |

## Observations

1. The upper-right nearest-element peak increases by factors of approximately
   `1.987`, `1.984`, and `1.963` when the mesh size is halved.
2. The product of corner peak stress and nearest Gauss-point distance has a
   coefficient of variation of only `1.24%` across all four meshes. This is
   consistent with an approximately inverse-distance near field.
3. The fixed 5–10 mm ring mean changes by `1.12%` between the two finest
   meshes.
4. The fixed 10×10 mm patch mean changes by `1.15%` between the two finest
   meshes.
5. The remote fixed-point displacement changes by only `0.20%` between the two
   finest meshes.

## Interpretation

The raw upper-right point-load peak is not a convergent local-stress quantity.
As the mesh is refined, the nearest integration point approaches the idealized
point load and the local peak grows approximately as `1/r`. In contrast,
fixed-region averages and the remote displacement are much less sensitive and
approach stable values.

This result does not mean that the entire finite-element solution is invalid.
It means that the raw point-load peak must not be used as the convergence target
or as a hotspot-PSO objective. If local stress at the load introduction is the
engineering quantity of interest, the load must be represented through a
finite physical transfer region, such as a bearing/contact area, bolt-hole
bearing region, plate, or distributed traction.

The global peak column is retained only to show why a global maximum is an
unsafe diagnostic: on coarse meshes it is governed by the fixed/free boundary
corner, while the upper-right point-load peak becomes dominant on the finest
mesh. Region-specific extraction is therefore required.
