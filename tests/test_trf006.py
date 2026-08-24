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


class TRF006Test(RuleTestCase):
    # --- TRF006: cache args usage (old TRF010) ---

    def test_trf006_catches_unused_cache_args(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states, past_key_value=None, use_cache=False):
        return hidden_states
"""
        trf006 = self._run(mlinter.TRF006, source)
        self.assertEqual(len(trf006), 1)
        self.assertIn("past_key_values/use_cache", trf006[0].message)

    def test_trf006_allows_referenced_cache_args(self):
        source = """
class FooPreTrainedModel:
    pass

class FooModel(FooPreTrainedModel):
    def forward(self, hidden_states, past_key_value=None, use_cache=False):
        if use_cache and past_key_value is not None:
            hidden_states = hidden_states + past_key_value[0]
        return hidden_states
"""
        trf006 = self._run(mlinter.TRF006, source)
        self.assertEqual(trf006, [])
