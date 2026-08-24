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


class TRF004Test(RuleTestCase):
    # --- TRF004: tie_weights hard ban (reworked old TRF007) ---

    def test_trf004_flags_any_tie_weights_override(self):
        source = """
class FooModel:
    def tie_weights(self):
        super().tie_weights()
"""
        trf004 = self._run(mlinter.TRF004, source)
        self.assertEqual(len(trf004), 1)
        self.assertIn("overrides tie_weights", trf004[0].message)

    def test_trf004_allows_no_tie_weights(self):
        source = """
class FooModel:
    _tied_weights_keys = ["lm_head.weight"]
"""
        trf004 = self._run(mlinter.TRF004, source)
        self.assertEqual(trf004, [])
