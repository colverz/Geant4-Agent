from __future__ import annotations

import unittest

from tools.evaluate_simulation_scenarios import evaluate_simulation_scenarios


class SimulationScenarioCasebankTest(unittest.TestCase):
    def test_simulation_scenario_casebank_reaches_runtime_contract(self) -> None:
        report = evaluate_simulation_scenarios()

        self.assertEqual(report["failed"], 0)
        self.assertGreaterEqual(report["total"], 3)


if __name__ == "__main__":
    unittest.main()
