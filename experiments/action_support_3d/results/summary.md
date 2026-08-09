# 3D action-support benchmark result

Reference mesh: 11238 nodes, 55387 C3D4 elements.

## Tip-displacement QoI: ten lowest errors

| rank | id | source | dof proxy | relative error | Pareto |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | LLM_H2_hole_root_coupled | llm_semantic | 4359 | 4.092768e-02 | True |
| 2 | LLM_T1_root_hole_transfer | llm_semantic | 4089 | 4.366164e-02 | True |
| 3 | LLM_T2_root_band | llm_semantic | 3120 | 7.872464e-02 | True |
| 4 | LLM_H3_positive_y_transfer | llm_semantic | 3480 | 8.413937e-02 | False |
| 5 | LLM_H1_hole_local | llm_semantic | 2562 | 9.716632e-02 | True |
| 6 | A010 | oracle_atom | 1710 | 1.290976e-01 | True |
| 7 | A000 | oracle_atom | 1737 | 1.319427e-01 | False |
| 8 | A001 | oracle_atom | 1785 | 1.337062e-01 | False |
| 9 | A011 | oracle_atom | 1755 | 1.378313e-01 | False |
| 10 | A100 | oracle_atom | 1668 | 1.425206e-01 | True |

## Hole-opening QoI: ten lowest errors

| rank | id | source | dof proxy | relative error | Pareto |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | LLM_H2_hole_root_coupled | llm_semantic | 4359 | 1.215726e-02 | True |
| 2 | LLM_T1_root_hole_transfer | llm_semantic | 4089 | 1.252046e-02 | True |
| 3 | LLM_T2_root_band | llm_semantic | 3120 | 2.872047e-02 | True |
| 4 | A001 | oracle_atom | 1785 | 7.823563e-02 | True |
| 5 | A011 | oracle_atom | 1755 | 8.003464e-02 | True |
| 6 | A000 | oracle_atom | 1737 | 8.035410e-02 | True |
| 7 | A010 | oracle_atom | 1710 | 8.451872e-02 | True |
| 8 | LLM_H3_positive_y_transfer | llm_semantic | 3480 | 9.300435e-02 | False |
| 9 | LLM_H1_hole_local | llm_semantic | 2562 | 9.727957e-02 | False |
| 10 | A211 | oracle_atom | 1683 | 1.162954e-01 | True |
