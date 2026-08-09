# 3D action-support benchmark result

Reference mesh: 11238 nodes, 55387 C3D4 elements.

## Tip-displacement QoI: ten lowest errors

| rank | id | source | dof proxy | relative error | Pareto |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | LLM_H2_hole_root_coupled | llm_semantic | 2994 | 5.798650e-02 | True |
| 2 | LLM_T1_root_hole_transfer | llm_semantic | 3009 | 5.833478e-02 | False |
| 3 | LLM_T2_root_band | llm_semantic | 2982 | 8.133371e-02 | True |
| 4 | LLM_H3_positive_y_transfer | llm_semantic | 2919 | 9.274682e-02 | True |
| 5 | LLM_H1_hole_local | llm_semantic | 2940 | 9.493663e-02 | False |
| 6 | A011 | oracle_atom | 2982 | 1.299115e-01 | False |
| 7 | A010 | oracle_atom | 3027 | 1.316431e-01 | False |
| 8 | A001 | oracle_atom | 3039 | 1.318174e-01 | False |
| 9 | A111 | oracle_atom | 3051 | 1.337291e-01 | False |
| 10 | A000 | oracle_atom | 3018 | 1.393483e-01 | False |

## Hole-opening QoI: ten lowest errors

| rank | id | source | dof proxy | relative error | Pareto |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | LLM_T2_root_band | llm_semantic | 2982 | 2.487613e-02 | True |
| 2 | LLM_H2_hole_root_coupled | llm_semantic | 2994 | 2.608174e-02 | False |
| 3 | LLM_T1_root_hole_transfer | llm_semantic | 3009 | 2.700813e-02 | False |
| 4 | A001 | oracle_atom | 3039 | 7.396801e-02 | False |
| 5 | A011 | oracle_atom | 2982 | 7.576000e-02 | False |
| 6 | A000 | oracle_atom | 3018 | 8.872090e-02 | False |
| 7 | A010 | oracle_atom | 3027 | 9.620931e-02 | False |
| 8 | LLM_H3_positive_y_transfer | llm_semantic | 2919 | 9.901498e-02 | True |
| 9 | LLM_H1_hole_local | llm_semantic | 2940 | 9.945494e-02 | False |
| 10 | A111 | oracle_atom | 3051 | 1.194632e-01 | False |
