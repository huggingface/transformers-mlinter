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


from tests.rule_test_utils import Path, RuleTestCase, mlinter


class TRF044Test(RuleTestCase):
    # --- TRF044: cache_position must not reappear in modeling signatures ---

    def test_trf044_flags_cache_position_parameter(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states, past_key_values=None, cache_position=None, **kwargs):
        return hidden_states
"""
        trf044 = self._run(mlinter.TRF044, source)
        self.assertEqual(len(trf044), 1)
        self.assertIn("cache_position", trf044[0].message)

    def test_trf044_flags_helper_functions_too(self):
        source = """
def create_causal_mask(attention_mask, cache_position):
    return attention_mask
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF044})
        self.assertEqual(len([v for v in violations if v.rule_id == mlinter.TRF044]), 1)

    def test_trf044_no_violation_without_cache_position(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states, past_key_values=None, **kwargs):
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF044})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF044], [])
