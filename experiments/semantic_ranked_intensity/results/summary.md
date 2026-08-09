# Fixed-theta Dörfler + LLM ranked refinement intensity

Reference QoI: `-1.242689837311e-02` mm at `138519` DOF proxy.

All Dörfler detection uses the same `theta = 0.50`. The global baseline refines every marked element by one size-ratio level `q = 0.80`. The semantic method keeps the same theta but maps frozen LLM ranks to physical size ratios: root `0.512`, hole `0.640`, background `0.800`.

No theta modulation and no estimator-derived semantic ranking are used. The only semantic intervention is region-dependent refinement depth through a fixed ordinal size-ratio mapping.

| target relative error | global DOF | semantic ranked-intensity DOF |
| ---: | ---: | ---: |
| < 0.100 | 2913 | 2538 |
| < 0.050 | 5352 | 9009 |
| < 0.030 | 10407 | 20241 |
| < 0.020 | 18507 | 20241 |
| < 0.015 |  | 29097 |
| < 0.010 |  | 29097 |
