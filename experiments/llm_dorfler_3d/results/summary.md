# LLM semantic partition + Dörfler versus global Dörfler

Reference QoI: `-1.227036966756e-02` mm from a global `2.5` mm mesh with `33714` DOF proxy.

Frozen LLM support: `A000, A001, A010, A011, A110, A111` (6 of 16 atoms).

Both methods use the same stress-jump element indicator, the same `theta = 0.50`, the same descending-prefix Dörfler rule, the same local remeshing operator, and the same stopping round. The only difference is the candidate pool presented to Dörfler.

| round | global DOF | global rel. error | LLM-gated DOF | LLM-gated rel. error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1131 | 1.658902e-01 | 1131 | 1.658902e-01 |
| 1 | 2337 | 6.869756e-02 | 1929 | 8.773049e-02 |
| 2 | 6372 | 2.538311e-02 | 4284 | 4.282434e-02 |
| 3 | 12723 | 1.244822e-02 | 7476 | 2.730610e-02 |
| 4 | 14925 | 1.126240e-02 | 7917 | 2.591448e-02 |

Final-round global Dörfler: error `1.126240e-02` at `14925` DOF proxy.
Final-round LLM-gated Dörfler: error `2.591448e-02` at `7917` DOF proxy.

Interpretation rule: a useful semantic partition should move the QoI-error-versus-DOF trajectory down and/or left; the LLM itself never ranks finite elements and never sees the estimator values.
