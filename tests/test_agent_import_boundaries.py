from __future__ import annotations

import importlib
import unittest


class AgentImportBoundaryTest(unittest.TestCase):
    def test_prompt_runtime_and_design_modules_import_independently(self) -> None:
        for module_name in (
            "planner.runtime_intent",
            "core.config.prompt_profiles",
            "core.agent",
            "core.agent.simulation_design_llm",
            "ui.web.request_router",
            "ui.web.strict_api",
        ):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
