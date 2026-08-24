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


class TRF051Test(RuleTestCase):
    # --- TRF051: no _attn_implementation branching in modeling code ---

    def test_trf051_flags_attn_implementation_comparison(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states):
        if self.config._attn_implementation == "flash_attention_2":
            return flash_path(hidden_states)
        return eager_path(hidden_states)
"""
        trf051 = self._run(mlinter.TRF051, source)
        self.assertEqual(len(trf051), 1)
        self.assertIn("ALL_ATTENTION_FUNCTIONS.get_interface", trf051[0].message)

    def test_trf051_flags_membership_test(self):
        source = """
def helper(config):
    return config._attn_implementation in ("sdpa", "eager")
"""
        file_path = Path("src/transformers/models/foo/modular_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF051})
        self.assertEqual(len([v for v in violations if v.rule_id == mlinter.TRF051]), 1)

    def test_trf051_allows_interface_dispatch(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states):
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        return attention_interface(self, hidden_states)
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF051})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF051], [])
