# 3D action-support benchmark result

Reference mesh: 11238 nodes, 55387 C3D4 elements.

## Tip-displacement QoI: ten lowest errors

| rank | id | source | dof proxy | relative error | Pareto |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | LLM_H2_hole_root_coupled | llm_semantic | 4359 | 1.882922e-02 | True |
| 2 | LLM_T1_root_hole_transfer | llm_semantic | 4089 | 2.261633e-02 | True |
| 3 | LLM_H3_positive_y_transfer | llm_semantic | 3480 | 6.052112e-02 | True |
| 4 | LLM_T2_root_band | llm_semantic | 3120 | 6.127276e-02 | True |
| 5 | LLM_H1_hole_local | llm_semantic | 2562 | 7.535897e-02 | True |
| 6 | A010 | oracle_atom | 1710 | 1.103268e-01 | True |
| 7 | A000 | oracle_atom | 1737 | 1.160815e-01 | False |
| 8 | A001 | oracle_atom | 1785 | 1.166151e-01 | False |
| 9 | A011 | oracle_atom | 1755 | 1.175273e-01 | False |
| 10 | A100 | oracle_atom | 1668 | 1.204433e-01 | True |

## Hole-opening QoI: ten lowest errors

| rank | id | source | dof proxy | relative error | Pareto |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | A011 | oracle_atom | 1755 | 1.680360e-01 | True |
| 2 | LLM_H1_hole_local | llm_semantic | 2562 | 2.930593e-01 | False |
| 3 | LLM_H3_positive_y_transfer | llm_semantic | 3480 | 3.261257e-01 | False |
| 4 | LLM_T1_root_hole_transfer | llm_semantic | 4089 | 4.041643e-01 | False |
| 5 | LLM_H2_hole_root_coupled | llm_semantic | 4359 | 5.132919e-01 | False |
| 6 | LLM_T2_root_band | llm_semantic | 3120 | 1.297870e+00 | False |
| 7 | A010 | oracle_atom | 1710 | 1.359871e+00 | True |
| 8 | A310 | oracle_atom | 1698 | 1.435657e+00 | True |
| 9 | A210 | oracle_atom | 1701 | 1.444802e+00 | False |
| 10 | LLM_T3_eccentric_load_path | llm_semantic | 2130 | 1.456547e+00 | False |
