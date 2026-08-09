# Region-detected variable-theta semantic Dörfler

Reference QoI: `-1.227036966756e-02` mm at `33714` DOF proxy.

Each round computes estimator density E_r/V_r in the root, hole, and residual background regions. Hotter regions receive larger theta_r and cooler regions receive smaller theta_r. A single multiplier is solved so sum(theta_r E_r) equals theta_global sum(E_r) with theta_global=0.50 before discrete Dörfler prefix overshoot.

| round | global DOF | global error | semantic DOF | semantic error | regional theta values |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1131 | 1.658902e-01 | 1131 | 1.658902e-01 | background_region:0.2101;hole_region:0.4993;root_region:0.8239 |
| 1 | 2337 | 6.869756e-02 | 2253 | 7.510434e-02 | background_region:0.2185;hole_region:0.5309;root_region:0.8083 |
| 2 | 6372 | 2.538311e-02 | 5874 | 2.791090e-02 | background_region:0.3172;hole_region:0.6253;root_region:0.7198 |
| 3 | 12723 | 1.244822e-02 | 10335 | 1.425480e-02 | background_region:0.2446;hole_region:0.5716;root_region:0.7841 |
| 4 | 14925 | 1.126240e-02 | 13269 | 1.194390e-02 | background_region:0.1824;hole_region:0.5658;root_region:0.7700 |
| 5 | 15126 | 1.137642e-02 | 14340 | 1.146907e-02 | background_region:0.1691;hole_region:0.5780;root_region:0.7525 |
| 6 | 15309 | 1.131557e-02 | 14733 | 1.159950e-02 |  |

Use `theta_history.csv` to verify that hotspot regions actually receive higher theta and more marked elements than the residual background when their estimator density is higher.
