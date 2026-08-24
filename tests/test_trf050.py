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


class TRF050Test(RuleTestCase):
    # --- TRF050: attention classes must not instantiate their own rotary embedding ---

    def test_trf050_flags_rotary_in_attention_init(self):
        source = """
class FooAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.rotary_emb = FooRotaryEmbedding(config)
"""
        trf050 = self._run(mlinter.TRF050, source)
        self.assertEqual(len(trf050), 1)
        self.assertIn("FooAttention.__init__ instantiates FooRotaryEmbedding", trf050[0].message)

    def test_trf050_allows_rotary_on_the_model(self):
        source = """
class FooModel(FooPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.rotary_emb = FooRotaryEmbedding(config)
        self.post_init()
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF050})
        self.assertEqual([v for v in violations if v.rule_id == mlinter.TRF050], [])
