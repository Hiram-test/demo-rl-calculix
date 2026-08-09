# Exhaustive fixed-pair comparison

All 120 unordered two-atom supports and all 16 single atoms were resource-normalized with the same target of 3000 DOF before solving.

## Global interior vertical displacement

Best single atom: `A011`, relative error 1.299115e-01, DOF 2982.

Best exhaustive atom pair: `P_A000_A011`, relative error 9.097655e-02, DOF 2961; 3 of 120 pairs lie within 10% of this pair optimum.

Best frozen LLM support: `LLM_H2_hole_root_coupled`, relative error 5.798650e-02, DOF 2994; rank 1 among 120 exhaustive pairs plus six semantic candidates.

| rank | candidate | source | relative error | DOF |
| ---: | --- | --- | ---: | ---: |
| 1 | LLM_H2_hole_root_coupled | llm_semantic | 5.798650e-02 | 2994 |
| 2 | LLM_T1_root_hole_transfer | llm_semantic | 5.833478e-02 | 3009 |
| 3 | LLM_T2_root_band | llm_semantic | 8.133371e-02 | 2982 |
| 4 | P_A000_A011 | atom_pair_exhaustive | 9.097655e-02 | 2961 |
| 5 | LLM_H3_positive_y_transfer | llm_semantic | 9.274682e-02 | 2919 |
| 6 | LLM_H1_hole_local | llm_semantic | 9.493663e-02 | 2940 |
| 7 | P_A001_A010 | atom_pair_exhaustive | 9.551967e-02 | 2991 |
| 8 | P_A000_A001 | atom_pair_exhaustive | 9.815955e-02 | 3009 |
| 9 | P_A010_A011 | atom_pair_exhaustive | 1.007526e-01 | 3027 |
| 10 | P_A010_A111 | atom_pair_exhaustive | 1.057022e-01 | 3057 |

## Local axial difference across the hole ligaments

Best single atom: `A001`, relative error 7.396801e-02, DOF 3039.

Best exhaustive atom pair: `P_A010_A011`, relative error 2.857619e-02, DOF 3027; 1 of 120 pairs lie within 10% of this pair optimum.

Best frozen LLM support: `LLM_T2_root_band`, relative error 2.487613e-02, DOF 2982; rank 1 among 120 exhaustive pairs plus six semantic candidates.

| rank | candidate | source | relative error | DOF |
| ---: | --- | --- | ---: | ---: |
| 1 | LLM_T2_root_band | llm_semantic | 2.487613e-02 | 2982 |
| 2 | LLM_H2_hole_root_coupled | llm_semantic | 2.608174e-02 | 2994 |
| 3 | LLM_T1_root_hole_transfer | llm_semantic | 2.700813e-02 | 3009 |
| 4 | P_A010_A011 | atom_pair_exhaustive | 2.857619e-02 | 3027 |
| 5 | P_A001_A010 | atom_pair_exhaustive | 3.507026e-02 | 2991 |
| 6 | P_A000_A011 | atom_pair_exhaustive | 3.615804e-02 | 2961 |
| 7 | P_A000_A001 | atom_pair_exhaustive | 4.612283e-02 | 3009 |
| 8 | P_A011_A210 | atom_pair_exhaustive | 5.214021e-02 | 3009 |
| 9 | P_A001_A011 | atom_pair_exhaustive | 5.381072e-02 | 2949 |
| 10 | P_A001_A110 | atom_pair_exhaustive | 6.207683e-02 | 2973 |

