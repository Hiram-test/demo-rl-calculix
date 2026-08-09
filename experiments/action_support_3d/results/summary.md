# 3D action-support benchmark result

Reference mesh: 11238 nodes, 55387 C3D4 elements.

## Tip-displacement QoI: ten lowest errors

| rank | id | source | dof proxy | relative error | Pareto |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | LLM_H2_hole_root_coupled | llm_semantic | 2994 | 5.798650e-02 | True |
| 2 | LLM_T1_root_hole_transfer | llm_semantic | 3009 | 5.833478e-02 | False |
| 3 | LLM_T2_root_band | llm_semantic | 2982 | 8.133371e-02 | True |
| 4 | P_A000_A011 | atom_pair_exhaustive | 2961 | 9.097655e-02 | True |
| 5 | LLM_H3_positive_y_transfer | llm_semantic | 2919 | 9.274682e-02 | True |
| 6 | LLM_H1_hole_local | llm_semantic | 2940 | 9.493663e-02 | False |
| 7 | P_A001_A010 | atom_pair_exhaustive | 2991 | 9.551967e-02 | False |
| 8 | P_A000_A001 | atom_pair_exhaustive | 3009 | 9.815955e-02 | False |
| 9 | P_A010_A011 | atom_pair_exhaustive | 3027 | 1.007526e-01 | False |
| 10 | P_A010_A111 | atom_pair_exhaustive | 3057 | 1.057022e-01 | False |

## Hole-opening QoI: ten lowest errors

| rank | id | source | dof proxy | relative error | Pareto |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | LLM_T2_root_band | llm_semantic | 2982 | 2.487613e-02 | True |
| 2 | LLM_H2_hole_root_coupled | llm_semantic | 2994 | 2.608174e-02 | False |
| 3 | LLM_T1_root_hole_transfer | llm_semantic | 3009 | 2.700813e-02 | False |
| 4 | P_A010_A011 | atom_pair_exhaustive | 3027 | 2.857619e-02 | False |
| 5 | P_A001_A010 | atom_pair_exhaustive | 2991 | 3.507026e-02 | False |
| 6 | P_A000_A011 | atom_pair_exhaustive | 2961 | 3.615804e-02 | True |
| 7 | P_A000_A001 | atom_pair_exhaustive | 3009 | 4.612283e-02 | False |
| 8 | P_A011_A210 | atom_pair_exhaustive | 3009 | 5.214021e-02 | False |
| 9 | P_A001_A011 | atom_pair_exhaustive | 2949 | 5.381072e-02 | True |
| 10 | P_A001_A110 | atom_pair_exhaustive | 2973 | 6.207683e-02 | False |
