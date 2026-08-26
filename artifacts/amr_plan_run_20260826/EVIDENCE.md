# Evidence for this campaign

Root: `/workspace/demo-rl-calculix-tbeam/artifacts/amr_plan_run/`

Solver: `/workspace/agentic-work/ccx` → `ccx_2.23`, SHA-256 `31be21fc2f0902bd9a05acc2651dbac6dc2a2573dabbf235e39a38cb6f458862`.

Not used as production methods: X0–X5, `split_x_stations` as Dörfler, `run_element_doerfler*`, `run_hw_slin`, `TYPE=C3D27`, `get_taper('T01')` for T01–T10.

## Steps

| Step | Path | Note |
|---|---|---|
| 0 | `step0/locked-objects.json` | SHA, 55 region vectors |
| 1 | `step1/volumes.json`, `figures/fig01_section_laws.pdf` | volumes, T01 ≠ G01 |
| 2 | `step2/summary.json` | h=20 mm C3D8, 11 beams S-LIN |
| 2b | `step2b/summary.json` | G01 h=28 mm, N=10368 |
| 2c | `step2c/summary.json` | G01 T0 C3D20, N=9573 |
| 3 | `step3/t0.json` | 11 × 540/912/2592 |
| 4 | `step4/summary.json` | uniform ladder, not Dörfler |
| 5 | `step5/{id}_{case}.json` | visual partition three-point |
| 6 | `step6/delivery.json` | in-process Q2 + hanging T_k^N |
| 7 | `step7/r8.json` | four conformal seeds + hanging seed |
| 8 | `step8/G01_zz.json` etc. | hanging 8-split Dörfler |
| paper | `paper/manuscript.pdf` | this run only |

## Figure scripts

`src/amr_plan_campaign/figures.py` reads only `amr_plan_run/`.
