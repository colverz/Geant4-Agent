from __future__ import annotations

import unittest

from tools.local_geant4_smoke import runtime_env_help


class LocalGeant4SmokeToolTest(unittest.TestCase):
    def test_runtime_env_help_names_explicit_opt_in_variables(self) -> None:
        message = runtime_env_help()

        self.assertIn("GEANT4_RUNTIME_COMMAND_JSON", message)
        self.assertIn("GEANT4_RUNTIME_COMMAND", message)
        self.assertIn("GEANT4_LIVE_SMOKE=1", message)


if __name__ == "__main__":
    unittest.main()
