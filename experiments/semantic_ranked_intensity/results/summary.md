# Region-level Dörfler + LLM ranked refinement intensity

Reference QoI: `-1.242689837311e-02` mm at `138519` DOF proxy.

Both methods use the same `theta = 0.50` and the same unit size ratio `q = 0.80`. The baseline applies Dörfler to elements and refines marked elements by one level. The semantic method applies Dörfler directly to the three LLM-defined region action units, then refines each selected region wholesale with frozen ranks root=3 (`0.512h`), hole=2 (`0.640h`), background=1 (`0.800h`).

There is no second element-level Dörfler inside a selected semantic region, so the semantic action-support representation is not collapsed back to the global element ranking.

| target relative error | global DOF | semantic region-level DOF |
| ---: | ---: | ---: |
| < 0.100 | 2913 | 4077 |
| < 0.050 | 5352 | 9993 |
| < 0.030 | 10407 | 28212 |
| < 0.020 |  | 28212 |
| < 0.015 |  | 52362 |
| < 0.010 |  | 52362 |
