# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from tests.rule_test_utils import RuleTestCase, mlinter


class TRF035Test(RuleTestCase):
    # --- TRF035: no # noqa in model files ---

    def test_trf035_flags_noqa(self):
        source = "from ...modeling_utils import PreTrainedModel  # noqa: F401\n"
        violations = self._run(mlinter.TRF035, source)
        self.assertEqual(len(violations), 1)
        self.assertIn("F401", violations[0].message)

    def test_trf035_flags_bare_noqa_and_skips_other_files(self):
        self.assertEqual(len(self._run(mlinter.TRF035, "import torch  # noqa\n")), 1)
        self.assertEqual(self._run(mlinter.TRF035, "import torch  # noqa\n", file_name="processing_foo.py"), [])

    def test_trf035_accepts_modular_undefined_name_codes(self):
        """A modular file trips ruff's undefined-name family by construction, not by mistake."""
        for source in (
            "from ...modeling_utils import PreTrainedModel  # noqa: F401\n",
            '__all__ = ["FooModel"]  # noqa: F822\n',
            "class FooLayer(LlamaDecoderLayer):  # noqa: F821\n    pass\n",
            "import torch  # noqa: F401, F821\n",
            "import torch  # noqa: f401\n",  # code case does not matter
        ):
            self.assertEqual(self._run(mlinter.TRF035, source, file_name="modular_foo.py"), [], source)
            # The same suppression in a shipped file still has something to fix.
            self.assertEqual(len(self._run(mlinter.TRF035, source, file_name="modeling_foo.py")), 1, source)

    def test_trf035_reports_the_codes_a_modular_file_does_not_get_a_pass_on(self):
        # Mixed codes are reported on the ones that are left, so the message says what to fix.
        violations = self._run(mlinter.TRF035, "x = y == True  # noqa: F401, E712\n", file_name="modular_foo.py")
        self.assertEqual(len(violations), 1)
        self.assertIn("E712", violations[0].message)
        self.assertNotIn("F401", violations[0].message)

        # A bare suppression is reported everywhere; in a modular file the ask is the code, not a rewrite.
        bare = self._run(mlinter.TRF035, "import torch  # noqa\n", file_name="modular_foo.py")
        self.assertEqual(len(bare), 1)
        self.assertIn("Name the code", bare[0].message)
        self.assertIn("F822", bare[0].message)

    def test_trf035_respects_suppression(self):
        source = "# trf-ignore: TRF035\nimport torch  # noqa: F401\n"
        self.assertEqual(self._run(mlinter.TRF035, source), [])
