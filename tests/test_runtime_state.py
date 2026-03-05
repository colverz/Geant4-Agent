from __future__ import annotations

import unittest

from ui.web.runtime_state import runtime_config_payload


class RuntimeStateTest(unittest.TestCase):
    def test_runtime_payload_contains_model_preflight(self) -> None:
        payload = runtime_config_payload()
        self.assertIn("model_preflight", payload)
        preflight = payload["model_preflight"]
        self.assertIn("ready", preflight)
        self.assertIn("structure", preflight)
        self.assertIn("ner", preflight)


if __name__ == "__main__":
    unittest.main()

