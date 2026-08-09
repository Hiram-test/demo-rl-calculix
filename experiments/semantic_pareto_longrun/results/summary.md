# Full-domain semantic partition + Dörfler versus global Dörfler

Reference QoI: `-1.242689837311e-02` mm at `138519` DOF proxy.

The LLM drawing defines an exhaustive three-region partition: root band, hole circle, and the residual background. No finite element is permanently excluded. Every round recomputes aggregate estimator mass per region, selects semantic regions by a region-level bulk criterion, then applies element-level Dörfler inside those regions until the same absolute global estimator fraction is captured as the conventional baseline.

Resource fairness: only minimum DOF needed to reach the same QoI error target is interpreted. Unequal final-round states are not called a win or an overtake.

| target relative error | global DOF | semantic-partition DOF |
| ---: | ---: | ---: |
| < 10.0% | 2337 | 2679 |
| < 5.0% | 6579 | 8727 |
| < 3.0% | 19827 | 22728 |
| < 2.0% | 19827 | 22728 |
| < 1.5% | 19827 | 50385 |

See `region_decisions.json` for whether the background region becomes active in later rounds.
