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


class TRF002Test(RuleTestCase):
    # --- TRF002: base_model_prefix (old TRF004) ---

    def test_trf002_valid_prefix(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
"""
        trf002 = self._run(mlinter.TRF002, source)
        self.assertEqual(trf002, [])

    def test_trf002_invalid_empty_prefix(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    base_model_prefix = ""
"""
        trf002 = self._run(mlinter.TRF002, source)
        self.assertEqual(len(trf002), 1)
        self.assertIn("non-empty canonical token", trf002[0].message)
