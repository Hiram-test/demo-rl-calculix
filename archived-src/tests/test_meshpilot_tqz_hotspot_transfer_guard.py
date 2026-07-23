import unittest

from meshpilot_pso import ObjectiveValue
from meshpilot_tqz_hotspot_batch_v5 import strict_hotspot_transfer_guard


class StrictHotspotTransferGuardTests(unittest.TestCase):
    def test_requires_at_least_twenty_five_percent_coarse_improvement(self):
        coarse = ObjectiveValue((0,) * 6, 0.10, 0.08, 1000, True)
        strong = ObjectiveValue((1,) * 6, 0.074, 0.05, 1200, True)
        marginal = ObjectiveValue((2,) * 6, 0.080, 0.04, 1200, True)
        self.assertTrue(strict_hotspot_transfer_guard(strong, coarse, 1.2))
        self.assertFalse(strict_hotspot_transfer_guard(marginal, coarse, 1.2))

    def test_feasible_always_beats_infeasible(self):
        feasible = ObjectiveValue((1,) * 6, 0.50, 0.20, 1000, True)
        infeasible = ObjectiveValue((0,) * 6, 0.01, 0.01, 20000, False, 0.2)
        self.assertTrue(strict_hotspot_transfer_guard(feasible, infeasible, 1.2))
        self.assertFalse(strict_hotspot_transfer_guard(infeasible, feasible, 1.2))


if __name__ == "__main__":
    unittest.main()
