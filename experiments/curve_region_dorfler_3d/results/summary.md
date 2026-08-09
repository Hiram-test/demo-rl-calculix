# Freeform LLM semantic regions + Dörfler versus global Dörfler

Reference QoI: `-1.227036966756e-02` mm from a global `2.5` mm mesh with `33714` DOF proxy.

Frozen LLM support: a full-cross-section root band from x=0 to 18 mm plus a circular envelope of radius 18 mm centred on the radius-8 mm through-hole at x=35 mm.

Both methods use the same stress-jump indicator, theta=0.50, descending-prefix Dörfler marking, local remeshing operator, reference solution, QoI and number of AMR rounds. The only intervention is whether Dörfler sees the whole mesh or only elements inside the frozen drawn semantic regions.

| round | global DOF | global error | curve-region DOF | curve-region error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1131 | 1.658902e-01 | 1131 | 1.658902e-01 |
| 1 | 2337 | 6.869756e-02 | 1965 | 8.400369e-02 |
| 2 | 6372 | 2.538311e-02 | 4044 | 4.536199e-02 |
| 3 | 12723 | 1.244822e-02 | 7338 | 2.943841e-02 |
| 4 | 14925 | 1.126240e-02 | 7968 | 2.580843e-02 |

## Same-accuracy resource comparison

| target relative error | global DOF | curve-region DOF |
| ---: | ---: | ---: |
| < 10% | 2337 | 1965 |
| < 5% | 6372 | 4044 |
| < 3% | 6372 | 7338 |

Interpretation: success means the drawn semantic support shifts the QoI-error-versus-DOF trajectory left/down; failure or late saturation means the drawing omitted regions that later become important.
