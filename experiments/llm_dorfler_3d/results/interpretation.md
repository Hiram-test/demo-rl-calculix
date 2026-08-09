# Resource-aligned interpretation

The same-round rows use different DOF counts, so the cleaner comparison is the minimum DOF required to cross fixed QoI-error thresholds.

| QoI relative-error target | Global Dörfler | LLM-gated Dörfler | LLM-gated resource change |
| ---: | ---: | ---: | ---: |
| ≤ 10% | 2337 DOF | 1929 DOF | -17.5% |
| ≤ 5% | 6372 DOF | 4284 DOF | -32.8% |
| ≤ 3% | 6372 DOF | 7476 DOF | +17.3% |

This first frozen six-atom semantic support therefore shows a finite-budget advantage rather than universal dominance. It reaches moderate accuracy with fewer DOF, then saturates because the action support excludes regions that become necessary for stricter accuracy. Global Dörfler eventually overtakes it when more computational resource is available.

This is the behavior the experiment was designed to expose: semantic action-support gating can concentrate early refinement, while an incomplete semantic support introduces a representation ceiling. The LLM selection was frozen before any estimator values or adaptive results were observed and is not changed after seeing this outcome.
