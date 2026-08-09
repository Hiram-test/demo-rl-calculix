# 3D action-support benchmark result

Reference mesh: 11238 nodes, 55387 C3D4 elements.

## Tip-displacement QoI: ten lowest errors

| rank | id | source | dof proxy | relative error | Pareto |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | LLM_H2_hole_root_coupled | llm_semantic | 4359 | 2.491239e-01 | True |
| 2 | LLM_T1_root_hole_transfer | llm_semantic | 4089 | 2.511455e-01 | True |
| 3 | LLM_H3_positive_y_transfer | llm_semantic | 3480 | 2.750829e-01 | True |
| 4 | LLM_T2_root_band | llm_semantic | 3120 | 2.795308e-01 | True |
| 5 | A311 | oracle_atom | 1740 | 2.796924e-01 | True |
| 6 | A310 | oracle_atom | 1698 | 2.837295e-01 | True |
| 7 | LLM_H1_hole_local | llm_semantic | 2562 | 2.889665e-01 | False |
| 8 | A010 | oracle_atom | 1710 | 3.151910e-01 | False |
| 9 | LLM_T3_eccentric_load_path | llm_semantic | 2130 | 3.190147e-01 | False |
| 10 | A001 | oracle_atom | 1785 | 3.192812e-01 | False |

## Hole-opening QoI: ten lowest errors

| rank | id | source | dof proxy | relative error | Pareto |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | A200 | oracle_atom | 1701 | 2.222913e+00 | True |
| 2 | LLM_T1_root_hole_transfer | llm_semantic | 4089 | 3.206780e+00 | False |
| 3 | A210 | oracle_atom | 1701 | 3.255687e+00 | False |
| 4 | LLM_H2_hole_root_coupled | llm_semantic | 4359 | 4.674270e+00 | False |
| 5 | A310 | oracle_atom | 1698 | 6.229183e+00 | True |
| 6 | A301 | oracle_atom | 1743 | 7.775177e+00 | False |
| 7 | LLM_T2_root_band | llm_semantic | 3120 | 8.137410e+00 | False |
| 8 | A211 | oracle_atom | 1683 | 9.425615e+00 | True |
| 9 | A001 | oracle_atom | 1785 | 9.502403e+00 | False |
| 10 | A011 | oracle_atom | 1755 | 9.795892e+00 | False |
