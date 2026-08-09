# Independent semantic normalization domains + Dörfler

Reference QoI: `-1.227036966756e-02` mm at `33714` DOF proxy.

The full domain is partitioned into three exhaustive semantic normalization domains: root, hole, and residual background. Every nonempty domain independently applies the same `theta = 0.50` Dörfler bulk rule to its own estimator mass, and the three marked prefixes are united before remeshing.

Because the regions form a disjoint exhaustive partition, the union automatically captures at least the same global theta fraction of estimator mass, while preventing one high-magnitude physical mechanism from completely suppressing another normalization domain.

| round | global DOF | global relative error | semantic DOF | semantic relative error | semantic marked elements |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1131 | 1.658902e-01 | 1131 | 1.658902e-01 | 162 |
| 1 | 2337 | 6.869756e-02 | 2589 | 7.348526e-02 | 641 |
| 2 | 6372 | 2.538311e-02 | 7239 | 2.414774e-02 | 2259 |
| 3 | 12723 | 1.244822e-02 | 14493 | 1.154936e-02 | 3838 |
| 4 | 14925 | 1.126240e-02 | 16170 | 1.038959e-02 | 3960 |
| 5 | 15126 | 1.137642e-02 | 16602 | 1.107241e-02 | 4041 |
| 6 | 15309 | 1.131557e-02 | 16755 | 1.065685e-02 | 0 |

Interpretation must use the error–DOF trajectory or minimum DOF at common error targets; equal-round states are not resource matched.
