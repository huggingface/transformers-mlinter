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


class TRF001Test(RuleTestCase):
    # --- TRF001: config_class naming consistency (old TRF003) ---

    def test_trf001_valid_config_class(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = FooConfig
"""
        trf001 = self._run(mlinter.TRF001, source)
        self.assertEqual(trf001, [])

    def test_trf001_invalid_config_class(self):
        source = """
class FooPreTrainedModel(PreTrainedModel):
    config_class = BarConfig
"""
        trf001 = self._run(mlinter.TRF001, source)
        self.assertEqual(len(trf001), 1)
        self.assertIn("config_class is BarConfig, expected FooConfig", trf001[0].message)
