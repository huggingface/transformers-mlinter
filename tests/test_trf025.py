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


from tests.rule_test_utils import Path, RuleTestCase, _helpers_mod, mlinter, patch


class TRF025Test(RuleTestCase):
    # --- TRF025: masks must be built once in the model, not per layer ---

    def _trf025(self, source, file_name="modeling_foo.py"):
        file_path = Path(f"src/transformers/models/foo/{file_name}")
        with patch.object(_helpers_mod, "model_contribution_date", return_value=None):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF025})
        return [v for v in violations if v.rule_id == mlinter.TRF025]

    def test_trf025_flags_mask_creation_inside_a_layer(self):
        source = """
class FooDecoderLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        attention_mask = create_causal_mask(config=self.config, attention_mask=attention_mask)
        return self.self_attn(hidden_states, attention_mask, **kwargs)
"""
        violations = self._trf025(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("create_causal_mask", violations[0].message)
        self.assertIn("FooDecoderLayer", violations[0].message)

    def test_trf025_flags_custom_mask_factory_in_attention(self):
        source = """
class FooAttention(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        mask = create_local_causal_valid_mask(hidden_states)
        return hidden_states + mask
"""
        self.assertEqual(len(self._trf025(source)), 1)

    def test_trf025_allows_mask_creation_in_the_model(self):
        source = """
class FooModel(FooPreTrainedModel):
    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        causal_mask = create_causal_mask(config=self.config, attention_mask=attention_mask)
        for layer in self.layers:
            inputs_embeds = layer(inputs_embeds, causal_mask, **kwargs)
        return inputs_embeds


class FooEncoder(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        attention_mask = create_bidirectional_mask(config=self.config, attention_mask=attention_mask)
        return hidden_states
"""
        self.assertEqual(self._trf025(source), [])

    def test_trf025_allows_layer_consuming_a_prepared_mask(self):
        source = """
class FooDecoderLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return self.self_attn(hidden_states, attention_mask, **kwargs)
"""
        self.assertEqual(self._trf025(source), [])

    def test_trf025_respects_suppression_and_file_type(self):
        source = """
# trf-ignore: TRF025
class FooDecoderLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return create_causal_mask(config=self.config, attention_mask=attention_mask)
"""
        self.assertEqual(self._trf025(source), [])
        plain = """
class FooLayer(nn.Module):
    def forward(self, x):
        return create_causal_mask(x)
"""
        self.assertEqual(self._trf025(plain, file_name="processing_foo.py"), [])
