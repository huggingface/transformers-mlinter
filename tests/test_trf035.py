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

    def test_trf035_respects_suppression(self):
        source = "# trf-ignore: TRF035\nimport torch  # noqa: F401\n"
        self.assertEqual(self._run(mlinter.TRF035, source), [])
