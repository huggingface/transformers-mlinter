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


class TRF043Test(RuleTestCase):
    # --- TRF043: Attention classes must not declare position_ids in forward ---

    def test_trf043_flags_position_ids_in_attention_forward(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, **kwargs):
        return hidden_states
"""
        trf043 = self._run(mlinter.TRF043, source)
        self.assertEqual(len(trf043), 1)
        self.assertIn("FooAttention.forward declares position_ids", trf043[0].message)

    def test_trf043_allows_kwargs_only_attention(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF043})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF043], [])

    def test_trf043_ignores_non_attention_classes(self):
        source = """
class FooDecoderLayer(GradientCheckpointingLayer):
    def forward(self, hidden_states, position_ids=None, **kwargs):
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF043})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF043], [])

    def test_trf043_respects_suppression_comment(self):
        source = """
class FooAttention(nn.Module):
    # trf-ignore: TRF043
    def forward(self, hidden_states, position_ids=None, **kwargs):
        return hidden_states
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF043})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF043], [])
